#!/usr/bin/env python3
"""
Task 6 — Predicted z vs. Ground-truth z ablation for C-GASTON.

Compares which 1-D signal is used inside the Soft-Weighted InfoNCE
negative-downweighting term (Technical Guide §4.4, Eq. 10-11):

    ω_ij = 1 - exp(-(d_i - d_j)^2 / σ^2)

  condition='pred_z'  : d_i = encoder(S_i), predicted isodepth (standard soft)
  condition='gt_z'    : d_i = ground-truth layer ordinal (L1→0 … WM→6),
                        globally z-scored — oracle upper-bound signal
  condition='spatial' : d_i = a fixed 1-D spatial proxy obtained from the first
                        principal component of z-scored Visium coordinates,
                        then z-scored within the slice

All three conditions now use the SAME corrected negative-downweighting loss,
so the ablation changes only the proximity signal source rather than the loss
family itself.

Loss formula (negative-downweighting, matches notebook exactly):
    numerator  = exp(sim(z_m_i, z_v_i) / τ)               (positive pair)
    denominator = Σ_j ω_ij · exp(sim(z_m_i, z_v_j) / τ)  (ω_ii forced = 1)
    L = -mean_i [ pos_logit_i - log(denominator_i) ]

Note on σ convention: Technical Guide / notebook use exp(-dist²/σ²), i.e. NO
factor of 2 in the exponent denominator.

All other hyperparameters fixed to Task-4 optimal:
  lambda = 0.1,  sigma = 0.5,  tau = 0.07,  5 restarts,  10 000 epochs.

Run: python run_z_ablation.py

Outputs:
    C_GASTON_ablation_z/{condition}/{slice_id}/
        model.pt
        isodepth.npy
        labels.npy
        cgaston_best.pt
        cgaston_isodepth.npy
        cgaston_labels.npy
    C_GASTON_ablation_z/results_summary.csv
    C_GASTON_ablation_z/aggregate_summary.csv
    C_GASTON_ablation_z/run_config.json
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
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
ABLATION_DIR     = f'{BASE_DIR}/C_GASTON_ablation_z'
os.makedirs(ABLATION_DIR, exist_ok=True)

# --- Ablation conditions ---
CONDITIONS = ['spatial', 'pred_z', 'gt_z']

# Fixed hyperparameters (optimal from Tasks 4 & 5)
LAMBDA_CONTRASTIVE = 0.1
SIGMA              = 0.5    # RBF bandwidth σ (isodepth space for pred_z/gt_z; coordinate space for spatial)
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

# Layer ordinal mapping (L1 = superficial, WM = deep white matter)
LABEL_TO_INT = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4, 'L6': 5, 'WM': 6}

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

SUMMARY_FIELDS = ['condition', 'slice', 'ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Conditions: {CONDITIONS}")
print(f"Lambda (fixed): {LAMBDA_CONTRASTIVE},  Sigma (fixed): {SIGMA}")
print(f"Restarts per condition: {NUM_RESTARTS}")

# ============================================================
# Utilities
# ============================================================
def load_rescale_input_data(S, A):
    S_norm = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-8)
    A_norm = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-8)
    return torch.tensor(S_norm, dtype=torch.float32), torch.tensor(A_norm, dtype=torch.float32)


def make_gt_z_tensor(gt_int):
    """
    Convert integer layer labels (0–6, -1 for invalid) to a globally z-scored
    1-D tensor of shape (N, 1).

    Spots with gt_int == -1 are assigned the mean of valid spots so they do
    not distort the RBF weights (they receive a middle-of-range depth value).
    """
    gt_float = gt_int.astype(np.float32)
    valid     = gt_int >= 0
    mean_val  = gt_float[valid].mean()
    std_val   = gt_float[valid].std() + 1e-8
    gt_float[~valid] = mean_val   # impute invalids before normalisation
    gt_norm = (gt_float - mean_val) / std_val
    return torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(1)  # (N, 1)


def make_spatial_z_tensor(coords):
    """Project 2-D coordinates to a deterministic 1-D spatial proxy."""
    coords = np.asarray(coords, dtype=np.float32)
    coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    spatial_1d = PCA(n_components=1, random_state=0).fit_transform(coords_norm).squeeze(1)
    spatial_1d = (spatial_1d - spatial_1d.mean()) / (spatial_1d.std() + 1e-8)
    return torch.tensor(spatial_1d.astype(np.float32), dtype=torch.float32).unsqueeze(1)


def population_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def build_model_payload(condition, model_state):
    return {
        'model_type': 'cgaston_z_ablation',
        'condition': condition,
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
        'output_dir': ABLATION_DIR,
        'slices': ALL_SLICES,
        'conditions': CONDITIONS,
        'model_type': 'cgaston_z_ablation',
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
        'spatial_signal': 'pc1_of_zscored_coords_then_zscored',
    }
    with open(f'{output_dir}/run_config.json', 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def morans_i(values, coords, k=6):
    """Row-standardised k-NN Moran's I (k=6 for Visium hexagonal grid)."""
    N  = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    W   = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        W[i, idx[i]] = 1.0 / k
    z = values - values.mean()
    I = float((N / W.sum()) * (z @ W @ z) / (z @ z))
    return I


def compute_metrics(isodepth, labels, gt, coords):
    valid = gt >= 0
    ari   = adjusted_rand_score(gt[valid], labels[valid])
    nmi   = normalized_mutual_info_score(gt[valid], labels[valid])
    d     = (isodepth - isodepth.min()) / (isodepth.max() - isodepth.min() + 1e-8)
    sp, _ = spearmanr(d[valid], gt[valid])
    mi    = morans_i(d, coords, k=6)
    return ari, nmi, abs(float(sp)), float(mi)

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
# Loss functions
# ============================================================
def soft_nce_neg_downweight(z_m, z_v, isodepth_1d, temperature=0.07, sigma=0.5):
    """
    Soft-Weighted InfoNCE — negative downweighting (Technical Guide §4.4 Eq.10-11;
    C_GASTON.ipynb soft_weighted_info_nce_loss).

    Weights each negative pair (i,j) by:
        ω_ij = 1 - exp(-(d_i - d_j)^2 / σ^2)
    Same-depth spots (d_i ≈ d_j) → ω_ij ≈ 0  (negative suppressed)
    Different-depth spots         → ω_ij ≈ 1  (negative kept)

    The positive pair diagonal is forced to ω_ii = 1.0 so it is always included
    in the denominator.

    Note: σ convention matches the notebook: exp(-dist²/σ²), NO factor of 2.

    Args:
        z_m:          molecular embeddings  (B, D)
        z_v:          vision embeddings     (B, D)
        isodepth_1d:  isodepth values       (B,) or (B, 1) — 1-D scalar per spot
        temperature:  logit scale τ
        sigma:        RBF bandwidth σ
    """
    z_m = F.normalize(z_m, dim=1)
    z_v = F.normalize(z_v, dim=1)

    d       = isodepth_1d.view(-1, 1)                     # (B, 1)
    dist_sq = (d - d.T) ** 2                               # (B, B)

    # ω_ij = 1 - exp(-dist² / σ²)
    weights = 1.0 - torch.exp(-dist_sq / (sigma ** 2))    # (B, B)

    # Positive pair (diagonal): force ω_ii = 1.0 so it appears in denominator
    B = weights.shape[0]
    weights[torch.arange(B), torch.arange(B)] = 1.0

    logits     = z_m @ z_v.T / temperature                 # (B, B)
    pos_logits = torch.diag(logits)                        # (B,)

    # Denominator: Σ_j ω_ij · exp(sim(z_m_i, z_v_j)/τ)
    denom = (weights * torch.exp(logits)).sum(dim=1)       # (B,)
    return -torch.mean(pos_logits - torch.log(denom + 1e-8))


# ============================================================
# Training
# ============================================================
def train_cgaston(model, S_torch, A_torch, V_torch, gt_z_torch, spatial_z_torch, condition,
                  lam=0.1, sigma=0.5, total_epochs=10000, warmup_epochs=2000,
                  temperature=0.07, batch_size=256, lr=1e-3,
                  log_interval=2000, seed=0):
    """
    condition:
        'pred_z'   — Soft-Weighted InfoNCE (neg. downweighting, Eq.10-11):
                     isodepth = encoder(S), with gradients flowing through
                     the weight term as in the notebook / Technical Guide.
        'gt_z'     — Same formula, but isodepth = GT layer ordinal (globally
                     z-scored oracle). Replaces d_i in Eq.10.
        'spatial'  — Same formula, but isodepth = fixed 1-D spatial proxy
                     from coordinates. Only the signal source changes.
    """
    torch.manual_seed(seed)
    N    = S_torch.shape[0]
    S    = S_torch.to(device)
    A    = A_torch.to(device)
    V    = V_torch.to(device)
    gt_z = gt_z_torch.to(device)   # (N, 1) — only used when condition='gt_z'
    spatial_z = spatial_z_torch.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    mse  = nn.MSELoss()
    history = {'recon': [], 'contrastive': [], 'total': []}

    for epoch in range(1, total_epochs + 1):
        model.train()
        _, _, z_hat = model.molecular_embedding(S)
        loss_recon  = mse(z_hat, A)

        loss_cont   = torch.tensor(0.0, device=device)
        if lam > 0 and epoch > warmup_epochs:
            perm          = torch.randperm(N, device=device)[:batch_size]
            z_m_b, d_b, _ = model.molecular_embedding(S[perm])
            z_v_b          = model.vision_embedding(V[perm])

            if condition == 'pred_z':
                # Use encoder's predicted isodepth as the depth signal.
                # Keep the gradient path through ω_ij so the weighting matches
                # the notebook / Technical Guide behavior.
                # No normalisation: σ operates in the raw encoder output space.
                d_pred    = d_b.squeeze(1)                     # (B,)
                loss_cont = soft_nce_neg_downweight(
                    z_m_b, z_v_b, d_pred,
                    temperature=temperature, sigma=sigma)

            elif condition == 'gt_z':
                # Oracle: replace predicted d with ground-truth layer ordinal.
                # gt_z is globally z-scored once at load time so σ=0.5 has
                # the same scale meaning as in the pred_z condition.
                gt_z_batch = gt_z[perm].squeeze(1)             # (B,)
                loss_cont  = soft_nce_neg_downweight(
                    z_m_b, z_v_b, gt_z_batch,
                    temperature=temperature, sigma=sigma)

            elif condition == 'spatial':
                # Fixed 1-D spatial proxy so the loss family stays identical
                # across all three z-source conditions.
                spatial_z_batch = spatial_z[perm].squeeze(1)    # (B,)
                loss_cont = soft_nce_neg_downweight(
                    z_m_b, z_v_b, spatial_z_batch,
                    temperature=temperature, sigma=sigma)

            else:
                raise ValueError(f"Unknown condition: {condition!r}")

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


class CGASTONWrapper:
    def __init__(self, m):
        self.spatial_embedding   = m.encoder
        self.expression_function = m.decoder

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

    # Ground-truth z: globally z-scored layer ordinal for this slice
    gt_z_t = make_gt_z_tensor(gt)
    spatial_z_t = make_spatial_z_tensor(S)

    data[sid] = {
        'adata':   adata,
        'coords':  S,
        'gt':      gt,
        'S_torch': S_t,
        'A_torch': A_t,
        'gt_z':    gt_z_t,   # (N, 1) — fixed oracle proximity signal
        'spatial_z': spatial_z_t,
    }
    n_valid = int((gt >= 0).sum())
    print(f"  S {S_t.shape}  A {A_t.shape}  gt labels: {np.unique(gt_str).tolist()}")
    print(f"  gt_z range: [{gt_z_t.min():.3f}, {gt_z_t.max():.3f}]  "
          f"valid spots: {n_valid}/{len(gt)}")
    print(f"  spatial_z range: [{spatial_z_t.min():.3f}, {spatial_z_t.max():.3f}]")

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

del vision_model  # free GPU memory before training

# ============================================================
# STEP 3 — Condition sweep
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Proximity signal ablation")
print("="*60)

csv_rows = []
summary  = {}

for condition in CONDITIONS:
    out_root = f'{ABLATION_DIR}/{condition}'
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"condition = '{condition}'")
    print(f"{'='*60}")

    cond_aris, cond_nmis, cond_spears, cond_morans = [], [], [], []

    for sid in ALL_SLICES:
        print(f"\n  --- Slice {sid} ---")
        S_t   = data[sid]['S_torch']
        A_t   = data[sid]['A_torch']
        V_t   = data[sid]['V_torch']
        gt_z_t = data[sid]['gt_z']
        spatial_z_t = data[sid]['spatial_z']

        best_loss        = float('inf')
        best_model_state = None

        for restart in range(NUM_RESTARTS):
            mdl = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                          isodepth_arch=ISODEPTH_ARCH,
                          expression_arch=EXPRESSION_ARCH).to(device)

            mdl, hist = train_cgaston(
                mdl, S_t, A_t, V_t, gt_z_t, spatial_z_t, condition=condition,
                lam=LAMBDA_CONTRASTIVE, sigma=SIGMA,
                total_epochs=TOTAL_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
                temperature=TEMPERATURE, batch_size=BATCH_SIZE,
                lr=LR, log_interval=2000, seed=restart)

            final_recon = hist['recon'][-1]
            if final_recon < best_loss:
                best_loss        = final_recon
                best_model_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}

        # Load best model and evaluate
        best_mdl = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                           isodepth_arch=ISODEPTH_ARCH,
                           expression_arch=EXPRESSION_ARCH).to(device)
        best_mdl.load_state_dict(best_model_state)
        best_mdl.eval()
        best_mdl_cpu = best_mdl.cpu()

        save_dir = f'{out_root}/{sid}'
        os.makedirs(save_dir, exist_ok=True)
        write_model_outputs(save_dir, condition, best_model_state)

        A_np    = A_t.detach().cpu().numpy()
        S_np    = S_t.detach().cpu().numpy()
        wrapper = CGASTONWrapper(best_mdl_cpu)
        isodepth, labels = dp_related.get_isodepth_labels(
            wrapper, A_np, S_np, NUM_LAYERS, num_buckets=100)

        write_prediction_outputs(save_dir, isodepth, labels)

        ari, nmi, sp, mi = compute_metrics(isodepth, labels,
                                           data[sid]['gt'], data[sid]['coords'])
        cond_aris.append(ari);   cond_nmis.append(nmi)
        cond_spears.append(sp);  cond_morans.append(mi)

        print(f"  {sid}: ARI={ari:.4f} NMI={nmi:.4f} Spear={sp:.4f} Moran={mi:.4f}  "
              f"(best_recon={best_loss:.4f})")

        csv_rows.append({
            'condition': condition, 'slice': sid,
            'ARI': round(ari, 4), 'NMI': round(nmi, 4),
            'Spearman': round(sp, 4), 'MoransI': round(mi, 4),
            'best_loss': round(best_loss, 6),
        })

    summary[condition] = {
        'ARI':      (np.mean(cond_aris),   np.std(cond_aris)),
        'NMI':      (np.mean(cond_nmis),   np.std(cond_nmis)),
        'Spearman': (np.mean(cond_spears), np.std(cond_spears)),
        'MoransI':  (np.mean(cond_morans), np.std(cond_morans)),
    }

# ============================================================
# STEP 4 — Save CSV and print summary table
# ============================================================
write_summary_files(ABLATION_DIR, csv_rows)
write_run_config(ABLATION_DIR)
print(f"\nPer-slice results saved to: {ABLATION_DIR}/results_summary.csv")

print("\n" + "="*60)
print("Z-SOURCE ABLATION SUMMARY  (mean ± std over 8 slices)")
print("="*60)
print(f"{'Condition':>10}  {'ARI':>14}  {'NMI':>14}  {'Spearman':>14}  {'Moran I':>14}")
print("-"*72)
for cond in CONDITIONS:
    s = summary[cond]
    print(f"{cond:>10}  "
          f"{s['ARI'][0]:>6.4f}±{s['ARI'][1]:.4f}  "
          f"{s['NMI'][0]:>6.4f}±{s['NMI'][1]:.4f}  "
          f"{s['Spearman'][0]:>6.4f}±{s['Spearman'][1]:.4f}  "
          f"{s['MoransI'][0]:>6.4f}±{s['MoransI'][1]:.4f}")
print("="*60)
print("Done! All results saved to:", ABLATION_DIR)
