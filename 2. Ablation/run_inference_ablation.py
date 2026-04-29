#!/usr/bin/env python3
"""
Task 7 — Inference: retain vision branch comparison.

After training C-GASTON (Soft-weighted InfoNCE, Technical Guide §4.4 Eq.10-11),
compares two inference strategies applied to the SAME trained model:

  condition='mol_only'  : Standard GASTON inference (Technical Guide §3.3).
                          Uses encoder(S) → d → dp_related breakpoint detection.
                          Vision branch is discarded at inference time.

  condition='mol_vis'   : Vision-retained inference.
                          Fuses the trained molecular embedding z_m and vision
                          embedding z_v by summing their L2-normalised
                          representations (z_fused = z_m/|z_m| + z_v/|z_v|),
                          then reduces to 1-D via PCA.  Equal-count quantile
                          binning (Q = 7 buckets) gives the final domain labels.

Training is identical for both conditions — one C-GASTON model is trained per
slice (5 restarts, best selected by reconstruction loss), and both inference
procedures are evaluated on the same best model.

All training hyperparameters fixed to Task-4 optimal:
    lambda = 0.1,  sigma = 0.5,  tau = 0.07,  5 restarts,  10 000 epochs.

Run: python run_inference_ablation.py

Outputs:
    C_GASTON_ablation_inference/{condition}/{slice_id}/
        model.pt
        isodepth.npy
        labels.npy
        cgaston_best.pt
        cgaston_isodepth.npy
        cgaston_labels.npy
    C_GASTON_ablation_inference/results_summary.csv   (all metrics)
    C_GASTON_ablation_inference/aggregate_summary.csv
    C_GASTON_ablation_inference/run_config.json
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from gaston import dp_related
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
def find_repo_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, 'DLPFC_Datasets')) and os.path.isdir(os.path.join(current, 'glmpca_results')):
            return current.replace('\\', '/')
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError('Could not locate repo root containing DLPFC_Datasets and glmpca_results.')
        current = parent


BASE_DIR         = find_repo_root()
GLMPCA_UNIFIED   = f'{BASE_DIR}/glmpca_results'
HE_BASE_DIR      = f'{BASE_DIR}/DLPFC_Datasets'
SAMPLE1_DATA_DIR = f'{BASE_DIR}/DLPFC_Datasets/Sample1/h5ad_cordinate_data'
SAMPLE3_DATA_DIR = f'{BASE_DIR}/DLPFC_Datasets/Sample3/h5ad_cordinate_data'
ABLATION_DIR     = f'{BASE_DIR}/C_GASTON_ablation_inference'
os.makedirs(ABLATION_DIR, exist_ok=True)

# --- Inference conditions to evaluate ---
CONDITIONS = ['mol_only', 'mol_vis']

# Fixed hyperparameters (optimal from Tasks 3–5)
LAMBDA_CONTRASTIVE = 0.1
SIGMA              = 0.5    # RBF bandwidth σ in isodepth space
NUM_LAYERS         = 7
ISODEPTH_ARCH      = [20, 20]
EXPRESSION_ARCH    = [20, 20]
NUM_DIMS           = 14
EMBEDDING_DIM      = 128
TEMPERATURE        = 0.07
BATCH_SIZE         = 256
WARMUP_EPOCHS      = 2000
TOTAL_EPOCHS       = 10000
NUM_RESTARTS       = 5
LR                 = 1e-3
PATCH_SIZE         = 224

LABEL_TO_INT = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4, 'L6': 5, 'WM': 6}
SUMMARY_FIELDS = ['condition', 'slice', 'ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']

ALL_SLICES = ['151507', '151508', '151509', '151510',
              '151673', '151674', '151675', '151676']

SLICE_DATA_DIRS = {
    '151507': SAMPLE1_DATA_DIR, '151508': SAMPLE1_DATA_DIR,
    '151509': SAMPLE1_DATA_DIR, '151510': SAMPLE1_DATA_DIR,
    '151673': SAMPLE3_DATA_DIR, '151674': SAMPLE3_DATA_DIR,
    '151675': SAMPLE3_DATA_DIR, '151676': SAMPLE3_DATA_DIR,
}

HE_IMAGE_PATHS = {
    '151507': f'{HE_BASE_DIR}/Sample1/H&E image/151507',
    '151508': f'{HE_BASE_DIR}/Sample1/H&E image/151508',
    '151509': f'{HE_BASE_DIR}/Sample1/H&E image/151509',
    '151510': f'{HE_BASE_DIR}/Sample1/H&E image/151510',
    '151673': f'{HE_BASE_DIR}/Sample3/H&E image/151673',
    '151674': f'{HE_BASE_DIR}/Sample3/H&E image/151674',
    '151675': f'{HE_BASE_DIR}/Sample3/H&E image/151675',
    '151676': f'{HE_BASE_DIR}/Sample3/H&E image/151676',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Conditions: {CONDITIONS}")
print(f"Lambda (fixed): {LAMBDA_CONTRASTIVE},  Sigma (fixed): {SIGMA}")
print(f"Restarts: {NUM_RESTARTS}  (one shared training per slice, two inference modes)")

# ============================================================
# Utilities
# ============================================================
def load_rescale_input_data(S, A):
    S_norm = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-8)
    A_norm = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
    return torch.tensor(S_norm, dtype=torch.float32), torch.tensor(A_norm, dtype=torch.float32)


def morans_i(values, coords, k=6):
    """Row-standardised k-NN Moran's I (k=6 for Visium hexagonal grid)."""
    N  = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    W  = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        W[i, idx[i]] = 1.0 / k
    z = values - values.mean()
    I = float((N / W.sum()) * (z @ W @ z) / (z @ z))
    return I


def compute_metrics(isodepth, labels, gt, coords):
    """ARI, NMI (valid mask); Spearman |r|, Moran's I (all spots)."""
    valid = gt >= 0
    ari   = adjusted_rand_score(gt[valid], labels[valid])
    nmi   = normalized_mutual_info_score(gt[valid], labels[valid])
    d     = (isodepth - isodepth.min()) / (isodepth.max() - isodepth.min() + 1e-8)
    sp, _ = spearmanr(d[valid], gt[valid])
    mi    = morans_i(d, coords, k=6)
    return ari, nmi, abs(float(sp)), float(mi)


def population_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def build_model_payload(condition, model_state):
    return {
        'model_type': 'cgaston_inference_ablation',
        'condition': condition,
        'training_shared_across_conditions': True,
        'state_dict': model_state,
        'num_layers': NUM_LAYERS,
        'isodepth_arch': ISODEPTH_ARCH,
        'expression_arch': EXPRESSION_ARCH,
        'num_dims': NUM_DIMS,
        'embedding_dim': EMBEDDING_DIM,
        'vision_backbone': 'resnet50',
        'patch_size': PATCH_SIZE,
        'temperature': TEMPERATURE,
        'lambda_contrastive': LAMBDA_CONTRASTIVE,
        'sigma_soft': SIGMA,
        'warmup_epochs': WARMUP_EPOCHS,
        'total_epochs': TOTAL_EPOCHS,
        'num_restarts': NUM_RESTARTS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
    }


def write_model_outputs(save_dir, condition, model_state):
    torch.save(build_model_payload(condition, model_state), f'{save_dir}/model.pt')
    torch.save(model_state, f'{save_dir}/cgaston_best.pt')


def write_prediction_outputs(save_dir, isodepth, labels):
    np.save(f'{save_dir}/isodepth.npy', isodepth)
    np.save(f'{save_dir}/labels.npy', labels)
    np.save(f'{save_dir}/cgaston_isodepth.npy', isodepth)
    np.save(f'{save_dir}/cgaston_labels.npy', labels)


def write_summary_files(output_dir, rows):
    with open(f'{output_dir}/results_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for condition in CONDITIONS:
        cond_rows = [row for row in rows if row['condition'] == condition]
        aggregate_row = {'condition': condition}
        for metric in ['ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']:
            mean_value, std_value = population_mean_std([row[metric] for row in cond_rows])
            aggregate_row[f'{metric}_mean'] = round(mean_value, 4)
            aggregate_row[f'{metric}_std'] = round(std_value, 4)
        aggregate_rows.append(aggregate_row)

    fieldnames = ['condition'] + [
        field
        for metric in ['ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']
        for field in (f'{metric}_mean', f'{metric}_std')
    ]
    with open(f'{output_dir}/aggregate_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def write_run_config(output_dir):
    config = {
        'base_dir': BASE_DIR,
        'output_dir': output_dir,
        'slices': ALL_SLICES,
        'conditions': CONDITIONS,
        'model_type': 'cgaston_inference_ablation',
        'training_shared_across_conditions': True,
        'vision_backbone': 'resnet50',
        'num_layers': NUM_LAYERS,
        'isodepth_arch': ISODEPTH_ARCH,
        'expression_arch': EXPRESSION_ARCH,
        'num_dims': NUM_DIMS,
        'embedding_dim': EMBEDDING_DIM,
        'temperature': TEMPERATURE,
        'lambda_contrastive': LAMBDA_CONTRASTIVE,
        'sigma_soft': SIGMA,
        'batch_size': BATCH_SIZE,
        'warmup_epochs': WARMUP_EPOCHS,
        'total_epochs': TOTAL_EPOCHS,
        'num_restarts': NUM_RESTARTS,
        'lr': LR,
        'patch_size': PATCH_SIZE,
        'device': device,
    }
    with open(f'{output_dir}/run_config.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)

# ============================================================
# Model
# ============================================================
class CGASTON(nn.Module):
    def __init__(self, K, D_v, D=128, isodepth_arch=[20, 20], expression_arch=[20, 20]):
        super().__init__()
        self.K = K
        self.D = D

        enc_dims = [2] + isodepth_arch + [1]
        enc = []
        for i in range(len(enc_dims) - 1):
            enc.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)

        dec_dims = [1] + expression_arch + [K]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

        self.mol_projection = nn.Linear(1 + K, D)
        self.vis_projection = nn.Sequential(
            nn.Linear(D_v, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, D),
        )

    def molecular_embedding(self, S):
        d     = self.encoder(S)          # (N, 1)
        z_hat = self.decoder(d)          # (N, K)
        z_m   = self.mol_projection(torch.cat([d, z_hat], dim=1))  # (N, D)
        return z_m, d, z_hat

    def vision_embedding(self, v):
        return self.vis_projection(v)    # (N, D)

# ============================================================
# Loss — Soft-weighted InfoNCE (Technical Guide §4.4 Eq.10-11)
# ============================================================
def soft_info_nce_loss(z_m, z_v, isodepth_1d, temperature=0.07, sigma=0.5):
    """
    Negative-downweighting soft InfoNCE (one-way, m→v):
        ω_ij = 1 - exp(-(d_i - d_j)² / σ²),  ω_ii = 1 (forced)
    Loss = -mean_i [ pos_logit_i - log Σ_j ω_ij · exp(logit_ij) ]
    """
    z_m = F.normalize(z_m, dim=1)
    z_v = F.normalize(z_v, dim=1)
    logits  = z_m @ z_v.T / temperature           # (B, B)

    d       = isodepth_1d.view(-1, 1)             # (B, 1)
    dist_sq = (d - d.T) ** 2                       # (B, B)
    weights = 1.0 - torch.exp(-dist_sq / (sigma ** 2))  # Eq.10: no factor of 2
    B = weights.shape[0]
    weights[torch.arange(B), torch.arange(B)] = 1.0     # force diagonal = 1

    pos_logits = torch.diag(logits)                # (B,)
    denom      = (weights * torch.exp(logits)).sum(dim=1)  # (B,) — m→v direction
    return -torch.mean(pos_logits - torch.log(denom + 1e-8))

# ============================================================
# Training
# ============================================================
def train_cgaston(model, S_torch, A_torch, V_torch,
                  lam=0.1, sigma=0.5, total_epochs=10000, warmup_epochs=2000,
                  temperature=0.07, batch_size=256, lr=1e-3,
                  log_interval=2000, seed=0):
    torch.manual_seed(seed)
    N   = S_torch.shape[0]
    S   = S_torch.to(device)
    A   = A_torch.to(device)
    V   = V_torch.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = {'recon': [], 'contrastive': [], 'total': []}

    for epoch in range(1, total_epochs + 1):
        model.train()
        _, _, z_hat = model.molecular_embedding(S)
        loss_recon  = mse(z_hat, A)

        loss_cont = torch.tensor(0.0, device=device)
        if lam > 0 and epoch > warmup_epochs:
            perm           = torch.randperm(N, device=device)[:batch_size]
            z_m_b, d_b, _  = model.molecular_embedding(S[perm])
            z_v_b          = model.vision_embedding(V[perm])
            loss_cont      = soft_info_nce_loss(
                z_m_b, z_v_b, d_b.squeeze(1),
                temperature=temperature, sigma=sigma)

        current_lam = lam if epoch > warmup_epochs else 0.0
        loss_total  = loss_recon + current_lam * loss_cont

        opt.zero_grad()
        loss_total.backward()
        opt.step()

        history['recon'].append(loss_recon.item())
        history['contrastive'].append(loss_cont.item())
        history['total'].append(loss_total.item())

        if epoch % log_interval == 0:
            phase = "WARMUP" if epoch <= warmup_epochs else "JOINT "
            print(f"  [{phase}] Ep {epoch:5d} | "
                  f"Recon {loss_recon.item():.4f} | "
                  f"Cont {loss_cont.item():.4f} | "
                  f"Total {loss_total.item():.4f}")

    return model, history

# ============================================================
# Inference helpers
# ============================================================
class CGASTONWrapper:
    """Thin wrapper for dp_related.get_isodepth_labels compatibility."""
    def __init__(self, m):
        self.spatial_embedding   = m.encoder
        self.expression_function = m.decoder


def infer_mol_only(model, A_np, S_np, num_layers):
    """
    Standard GASTON inference (Technical Guide §3.3).
    Uses encoder(S) → d → dp_related breakpoint detection on the piecewise
    linear decoder.  Vision branch is not used.
    Returns (isodepth, labels) as numpy arrays.
    """
    wrapper = CGASTONWrapper(model)
    isodepth, labels = dp_related.get_isodepth_labels(
        wrapper, A_np, S_np, num_layers, num_buckets=100)
    return isodepth, labels.astype(int)


def infer_mol_vis(model, S_t, V_t, num_layers):
    """
    Vision-retained inference.

    1. Compute z_m = mol_projection(cat(d, z_hat))  and  z_v = vis_projection(v).
    2. L2-normalise both, sum element-wise: z_fused = z_m_n + z_v_n.
       (Summing unit vectors preserves direction; near-aligned embeddings reinforce,
        misaligned ones partially cancel — a natural soft-AND fusion.)
    3. Reduce to 1-D via PCA first component.
    4. Discretise into num_layers equal-count quantile bins.

    Returns (isodepth, labels) where isodepth is the raw PCA score (1-D float).
    """
    model.eval()
    with torch.no_grad():
        z_m, _, _ = model.molecular_embedding(S_t)
        z_v       = model.vision_embedding(V_t)
        z_m_n     = F.normalize(z_m, dim=1).numpy()  # (N, D)
        z_v_n     = F.normalize(z_v, dim=1).numpy()  # (N, D)

    z_fused  = z_m_n + z_v_n                     # (N, D) sum of unit vectors

    # PCA → 1D isodepth proxy
    pca      = PCA(n_components=1, random_state=0)
    isodepth = pca.fit_transform(z_fused).squeeze(1)  # (N,)

    # Equal-count quantile binning into num_layers bins (0 … num_layers-1)
    boundaries = np.percentile(isodepth, np.linspace(0, 100, num_layers + 1))
    boundaries[-1] += 1e-6           # include the exact maximum value
    labels     = np.digitize(isodepth, boundaries[1:-1]).astype(int)

    return isodepth, labels

# ============================================================
# H&E patch extraction
# ============================================================
def extract_patches(adata, he_dir, output_size=224, crop_multiplier=3.0):
    img    = Image.open(f'{he_dir}/tissue_hires_image.png').convert('RGB')
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    with open(f'{he_dir}/scalefactors_json.json') as f:
        sf = json.load(f)
    scale       = sf['tissue_hires_scalef']
    crop_radius = int(np.ceil(sf['spot_diameter_fullres'] * scale * crop_multiplier / 2))

    pos_df = {}
    with open(f'{he_dir}/tissue_positions_list.txt') as f:
        for line in f:
            parts = line.strip().split(',')
            pos_df[parts[0]] = (float(parts[4]), float(parts[5]))

    barcodes = adata.obs_names.tolist()
    patches  = np.zeros((len(barcodes), output_size, output_size, 3), dtype=np.uint8)
    for i, bc in enumerate(barcodes):
        if bc not in pos_df:
            continue
        r, c       = pos_df[bc]
        r_hi, c_hi = r * scale, c * scale
        r1 = max(0, int(r_hi - crop_radius)); r2 = min(H_img, int(r_hi + crop_radius))
        c1 = max(0, int(c_hi - crop_radius)); c2 = min(W_img, int(c_hi + crop_radius))
        crop = img_np[r1:r2, c1:c2]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        patches[i] = np.array(Image.fromarray(crop).resize(
            (output_size, output_size), Image.LANCZOS))
    return patches


patch_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_vision_features(patches, vision_model, D_v, batch_size=64):
    vision_model = vision_model.to(device)
    N        = len(patches)
    features = np.zeros((N, D_v), dtype=np.float32)
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        imgs = torch.stack([patch_transform(Image.fromarray(p))
                            for p in patches[start:end]]).to(device)
        with torch.no_grad():
            features[start:end] = vision_model(imgs).cpu().numpy()
        if (start // batch_size) % 10 == 0:
            print(f"  Extracted {end}/{N} patches...", end='\r')
    print(f"  Extracted {N}/{N} patches.     ")
    return features

# ============================================================
# STEP 1 — Load data for all 8 slices
# ============================================================
print("\n" + "="*60)
print("STEP 1 — Loading data for all 8 slices")
print("="*60)

data = {}
for sid in ALL_SLICES:
    print(f"\n--- {sid} ---")
    adata = ad.read_h5ad(f'{SLICE_DATA_DIRS[sid]}/{sid}.h5ad')
    adata.var_names_make_unique()
    S      = np.asarray(adata.obsm['spatial'])
    gt_str = adata.obs['original_domain'].astype(str).values
    gt     = np.array([LABEL_TO_INT.get(l, -1) for l in gt_str])
    A      = np.load(f'{GLMPCA_UNIFIED}/{sid}/glmpca.npy')
    S_t, A_t = load_rescale_input_data(S, A)
    data[sid] = {'adata': adata, 'coords': S, 'gt': gt, 'S_torch': S_t, 'A_torch': A_t}
    print(f"  S {S_t.shape}  A {A_t.shape}  gt labels: {np.unique(gt_str).tolist()}")

# ============================================================
# STEP 2 — Extract vision features ONCE
# ============================================================
print("\n" + "="*60)
print("STEP 2 — Extracting vision features (ResNet-50, done once)")
print("="*60)

vision_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
vision_model.fc = nn.Identity()
vision_model.eval()
for p in vision_model.parameters():
    p.requires_grad = False
D_v = 2048

for sid in ALL_SLICES:
    print(f"\n--- {sid} ---")
    patches = extract_patches(data[sid]['adata'], HE_IMAGE_PATHS[sid],
                              output_size=PATCH_SIZE, crop_multiplier=3.0)
    data[sid]['V_torch'] = torch.tensor(
        extract_vision_features(patches, vision_model, D_v), dtype=torch.float32)
    print(f"  Vision features: {data[sid]['V_torch'].shape}")

del vision_model

# ============================================================
# STEP 3 — Train one C-GASTON model per slice (shared for both inference modes)
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Training C-GASTON (Soft InfoNCE, shared model per slice)")
print("="*60)

trained_models = {}

for sid in ALL_SLICES:
    print(f"\n{'='*60}")
    print(f"Training: slice {sid}")
    print(f"{'='*60}")

    S_t  = data[sid]['S_torch']
    A_t  = data[sid]['A_torch']
    V_t  = data[sid]['V_torch']

    best_loss        = float('inf')
    best_model_state = None

    for restart in range(NUM_RESTARTS):
        print(f"\n  --- Restart {restart}/{NUM_RESTARTS-1} ---")
        mdl = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                      isodepth_arch=ISODEPTH_ARCH,
                      expression_arch=EXPRESSION_ARCH).to(device)

        mdl, hist = train_cgaston(
            mdl, S_t, A_t, V_t,
            lam=LAMBDA_CONTRASTIVE, sigma=SIGMA,
            total_epochs=TOTAL_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
            temperature=TEMPERATURE, batch_size=BATCH_SIZE,
            lr=LR, log_interval=2000, seed=restart)

        final_recon = hist['recon'][-1]
        if final_recon < best_loss:
            best_loss        = final_recon
            best_model_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            print(f"  >>> New best! (recon={best_loss:.6f})")

    best_mdl = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                       isodepth_arch=ISODEPTH_ARCH,
                       expression_arch=EXPRESSION_ARCH)
    best_mdl.load_state_dict(best_model_state)
    best_mdl.eval()

    # Save the trained model once (shared by both inference conditions)
    shared_save_dir = f'{ABLATION_DIR}/trained_models/{sid}'
    os.makedirs(shared_save_dir, exist_ok=True)
    torch.save(best_model_state, f'{shared_save_dir}/cgaston_best.pt')

    trained_models[sid] = {'model': best_mdl, 'best_loss': best_loss}
    print(f"\n{sid}: Training done (best recon loss = {best_loss:.6f})")

# ============================================================
# STEP 4 — Apply both inference modes and evaluate
# ============================================================
print("\n" + "="*60)
print("STEP 4 — Evaluating both inference modes")
print("="*60)

csv_rows = []
# summary[condition][metric] = list of per-slice values
summary = {c: {'ARI': [], 'NMI': [], 'Spearman': [], 'MoransI': []}
           for c in CONDITIONS}

for condition in CONDITIONS:
    print(f"\n{'='*60}")
    print(f"Inference mode: '{condition}'")
    print(f"{'='*60}")

    out_root = f'{ABLATION_DIR}/{condition}'
    os.makedirs(out_root, exist_ok=True)

    for sid in ALL_SLICES:
        print(f"\n  --- {sid} ---")

        mdl  = trained_models[sid]['model']
        S_t  = data[sid]['S_torch']
        V_t  = data[sid]['V_torch']
        A_np = S_t.detach().numpy()           # normalised coords for dp_related
        S_np = data[sid]['S_torch'].detach().numpy()

        if condition == 'mol_only':
            # Standard GASTON inference: encoder → dp_related breakpoint detection
            # Vision branch is not used (Technical Guide §3.3)
            isodepth, labels = infer_mol_only(mdl, data[sid]['A_torch'].detach().numpy(),
                                              S_np, NUM_LAYERS)

        elif condition == 'mol_vis':
            # Vision-retained inference: PCA on fused (z_m_n + z_v_n) → quantile bins
            isodepth, labels = infer_mol_vis(mdl, S_t, V_t, NUM_LAYERS)

        save_dir = f'{out_root}/{sid}'
        os.makedirs(save_dir, exist_ok=True)
        write_model_outputs(save_dir, condition, best_mdl.state_dict())
        write_prediction_outputs(save_dir, isodepth, labels)

        ari, nmi, sp, mi = compute_metrics(isodepth, labels,
                                           data[sid]['gt'], data[sid]['coords'])
        summary[condition]['ARI'].append(ari)
        summary[condition]['NMI'].append(nmi)
        summary[condition]['Spearman'].append(sp)
        summary[condition]['MoransI'].append(mi)

        print(f"  {sid}: ARI={ari:.4f} NMI={nmi:.4f} Spear={sp:.4f} Moran={mi:.4f}  "
              f"(recon={trained_models[sid]['best_loss']:.4f})")

        csv_rows.append({
            'condition': condition, 'slice': sid,
            'ARI': round(ari, 4), 'NMI': round(nmi, 4),
            'Spearman': round(sp, 4), 'MoransI': round(mi, 4),
            'best_loss': round(trained_models[sid]['best_loss'], 6),
        })

# ============================================================
# STEP 5 — Save CSV and print summary table
# ============================================================
write_summary_files(ABLATION_DIR, csv_rows)
write_run_config(ABLATION_DIR)
print(f"\nPer-slice results saved to: {ABLATION_DIR}/results_summary.csv")

print("\n" + "="*60)
print("INFERENCE ABLATION SUMMARY  (mean ± std over 8 slices)")
print("="*60)
print(f"{'Condition':>10}  {'ARI':>14}  {'NMI':>14}  {'Spearman':>14}  {'Moran I':>14}")
print("-"*72)
for cond in CONDITIONS:
    s = summary[cond]
    print(f"{cond:>10}  "
          f"{np.mean(s['ARI']):>6.4f}±{np.std(s['ARI']):.4f}  "
          f"{np.mean(s['NMI']):>6.4f}±{np.std(s['NMI']):.4f}  "
          f"{np.mean(s['Spearman']):>6.4f}±{np.std(s['Spearman']):.4f}  "
          f"{np.mean(s['MoransI']):>6.4f}±{np.std(s['MoransI']):.4f}")
print("="*60)
print(f"\nDelta (mol_vis − mol_only):")
for metric in ['ARI', 'NMI', 'Spearman', 'MoransI']:
    delta = np.mean(summary['mol_vis'][metric]) - np.mean(summary['mol_only'][metric])
    print(f"  {metric:>10}: {delta:+.4f}")
print(f"\nAll results saved to: {ABLATION_DIR}/")
print("Done!")
