#!/usr/bin/env python3
"""
C-GASTON training with Soft-weighted InfoNCE loss (Technical Guide §4.4, Eq.10-11).

Difference from run_cgaston_pipeline.py:
  Standard InfoNCE treats all j≠i as equally penalised negatives. Soft-Weighted
  InfoNCE instead *downweights* negatives whose isodepth is close to the anchor:

      ω_ij = 1 − exp(−(d_i − d_j)² / σ²)

  where d_i is the scalar isodepth produced by the encoder for spot i.
  ω_ii = 1 (positive pair always included). ω_ij → 0 when depths are equal
  (same-layer negatives suppressed); ω_ij → 1 when depths differ (true negatives kept).

  sigma (σ) is the RBF bandwidth in isodepth space (default SIGMA_SOFT = 0.5).

Run: python run_cgaston_soft_pipeline.py
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
from scipy.stats import spearmanr
from gaston import neural_net, dp_related
import warnings
warnings.filterwarnings('ignore')

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


BASE_DIR = find_repo_root()
GLMPCA_UNIFIED   = f'{BASE_DIR}/glmpca_results'
HE_BASE_DIR      = f'{BASE_DIR}/DLPFC_Datasets'
SAMPLE1_DATA_DIR = f'{BASE_DIR}/DLPFC_Datasets/Sample1/h5ad_cordinate_data'
SAMPLE3_DATA_DIR = f'{BASE_DIR}/DLPFC_Datasets/Sample3/h5ad_cordinate_data'
OUTPUT_DIR       = f'{BASE_DIR}/C_GASTON_soft_results'   # separate output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

LABEL_TO_INT = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4, 'L6': 5, 'WM': 6}
SUMMARY_FIELDS = ['slice', 'ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']

# Hyperparameters (identical to standard pipeline except SIGMA_SOFT)
NUM_LAYERS       = 7
ISODEPTH_ARCH    = [20, 20]
EXPRESSION_ARCH  = [20, 20]
NUM_DIMS         = 14
EMBEDDING_DIM    = 128
TEMPERATURE      = 0.07
LAMBDA_CONTRASTIVE = 0.1
SIGMA_SOFT       = 0.5   # RBF bandwidth in isodepth space (Technical Guide §4.4)
BATCH_SIZE       = 256
WARMUP_EPOCHS    = 2000
TOTAL_EPOCHS     = 10000
NUM_RESTARTS     = 10
LR               = 1e-3
PATCH_SIZE       = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# Metrics helpers
# ============================================================
def morans_i(values, coords, k=6):
    """Row-standardised k-NN Moran's I (k=6 for Visium hexagonal grid)."""
    N = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    W = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        W[i, idx[i]] = 1.0 / k
    z = values - values.mean()
    I = float((N / W.sum()) * (z @ W @ z) / (z @ z))
    return I


def population_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def build_model_payload(model_state):
    return {
        'model_type': 'cgaston_soft_weighted',
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
        'sigma_soft': SIGMA_SOFT,
        'warmup_epochs': WARMUP_EPOCHS,
        'total_epochs': TOTAL_EPOCHS,
        'num_restarts': NUM_RESTARTS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
    }


def write_model_outputs(save_dir, model_state, legacy_state):
    torch.save(build_model_payload(model_state), f'{save_dir}/model.pt')
    torch.save(legacy_state, f'{save_dir}/cgaston_soft_best.pt')


def write_prediction_outputs(save_dir, isodepth, labels):
    np.save(f'{save_dir}/isodepth.npy', isodepth)
    np.save(f'{save_dir}/labels.npy', labels)
    np.save(f'{save_dir}/cgaston_soft_isodepth.npy', isodepth)
    np.save(f'{save_dir}/cgaston_soft_labels.npy', labels)


def write_summary_files(output_dir, rows):
    with open(f'{output_dir}/results_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_row = {}
    for metric in ['ARI', 'NMI', 'Spearman', 'MoransI', 'best_loss']:
        mean_value, std_value = population_mean_std([row[metric] for row in rows])
        aggregate_row[f'{metric}_mean'] = round(mean_value, 4)
        aggregate_row[f'{metric}_std'] = round(std_value, 4)

    with open(f'{output_dir}/aggregate_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_row.keys()))
        writer.writeheader()
        writer.writerow(aggregate_row)


def write_run_config(output_dir):
    config = {
        'base_dir': BASE_DIR,
        'output_dir': OUTPUT_DIR,
        'slices': ALL_SLICES,
        'model_type': 'cgaston_soft_weighted',
        'vision_backbone': 'resnet50',
        'num_layers': NUM_LAYERS,
        'isodepth_arch': ISODEPTH_ARCH,
        'expression_arch': EXPRESSION_ARCH,
        'num_dims': NUM_DIMS,
        'embedding_dim': EMBEDDING_DIM,
        'temperature': TEMPERATURE,
        'lambda_contrastive': LAMBDA_CONTRASTIVE,
        'sigma_soft': SIGMA_SOFT,
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
# z-score normalization
# ============================================================
def load_rescale_input_data(S, A):
    S_mean, S_std = S.mean(axis=0), S.std(axis=0)
    A_mean, A_std = A.mean(axis=0), A.std(axis=0)
    S_norm = (S - S_mean) / (S_std + 1e-8)
    A_norm = (A - A_mean) / (A_std + 1e-8)
    return torch.tensor(S_norm, dtype=torch.float32), torch.tensor(A_norm, dtype=torch.float32)

# ============================================================
# C-GASTON Model (identical to standard pipeline)
# ============================================================
class CGASTON(nn.Module):
    def __init__(self, K, D_v, D=128, isodepth_arch=[20, 20], expression_arch=[20, 20]):
        super().__init__()
        self.K = K
        self.D = D

        enc_layers = []
        dims = [2] + isodepth_arch + [1]
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        dims = [1] + expression_arch + [K]
        for i in range(len(dims) - 1):
            dec_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        self.mol_projection = nn.Linear(1 + K, D)
        self.vis_projection = nn.Sequential(
            nn.Linear(D_v, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, D),
        )

    def encode_isodepth(self, S):
        return self.encoder(S)

    def decode_expression(self, d):
        return self.decoder(d)

    def molecular_embedding(self, S):
        d = self.encode_isodepth(S)
        z_hat = self.decode_expression(d)
        concat = torch.cat([d, z_hat], dim=1)
        z_m = self.mol_projection(concat)
        return z_m, d, z_hat

    def vision_embedding(self, v):
        return self.vis_projection(v)

    def forward(self, S, v):
        z_m, d, z_hat = self.molecular_embedding(S)
        z_v = self.vision_embedding(v)
        return z_m, z_v, d, z_hat

# ============================================================
# Loss functions
# ============================================================
def info_nce_loss(z_m, z_v, temperature=0.07):
    """Standard InfoNCE: one-hot positive pairs (diagonal only)."""
    z_m = F.normalize(z_m, dim=1)
    z_v = F.normalize(z_v, dim=1)
    logits = z_m @ z_v.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_m2v = F.cross_entropy(logits, labels)
    loss_v2m = F.cross_entropy(logits.T, labels)
    return (loss_m2v + loss_v2m) / 2


def soft_info_nce_loss(z_m, z_v, isodepth_1d, temperature=0.07, sigma=0.5):
    """
    Soft-weighted InfoNCE with negative downweighting (Technical Guide §4.4 Eq.10-11).

    Negative pairs close in isodepth space are downweighted rather than fully
    penalised:
        ω_ij = 1 - exp(-(d_i - d_j)² / σ²)   (0 for identical depth, 1 for far)
        ω_ii = 1  (diagonal forced — positive pair always included)

    The loss denominator sums ω_ij * exp(sim/τ) instead of exp(sim/τ),
    which suppresses contributions from nearby negatives.

    Args:
        z_m          : (B, D) molecular embeddings (unnormalized)
        z_v          : (B, D) vision embeddings (unnormalized)
        isodepth_1d  : (B,) scalar isodepth values from the encoder
        temperature  : softmax temperature τ
        sigma        : RBF bandwidth for isodepth distance (no factor of 2)
    """
    z_m = F.normalize(z_m, dim=1)
    z_v = F.normalize(z_v, dim=1)
    logits = z_m @ z_v.T / temperature                          # (B, B)

    d = isodepth_1d.view(-1, 1)                                 # (B, 1)
    dist_sq = (d - d.T) ** 2                                    # (B, B)
    weights = 1.0 - torch.exp(-dist_sq / (sigma ** 2))         # Eq.10: no factor of 2
    B = weights.shape[0]
    weights[torch.arange(B), torch.arange(B)] = 1.0            # force diagonal = 1

    pos_logits = torch.diag(logits)                             # (B,)
    denom = (weights * torch.exp(logits)).sum(dim=1)            # (B,)  m→v direction
    return -torch.mean(pos_logits - torch.log(denom + 1e-8))

# ============================================================
# Training loop
# ============================================================
def train_cgaston_soft(model, S_torch, A_torch, V_torch,
                       total_epochs=10000, warmup_epochs=2000,
                       lam=0.1, temperature=0.07, sigma=0.5,
                       batch_size=256, lr=1e-3, log_interval=2000, seed=0):
    torch.manual_seed(seed)
    N = S_torch.shape[0]
    S = S_torch.to(device)
    A = A_torch.to(device)
    V = V_torch.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()
    history = {'recon': [], 'contrastive': [], 'total': []}

    for epoch in range(1, total_epochs + 1):
        model.train()
        z_m_full, d_full, z_hat_full = model.molecular_embedding(S)
        loss_recon = mse_fn(z_hat_full, A)

        loss_cont = torch.tensor(0.0, device=device)
        if epoch > warmup_epochs:
            perm = torch.randperm(N, device=device)[:batch_size]
            z_m_batch, d_batch, _ = model.molecular_embedding(S[perm])
            z_v_batch = model.vision_embedding(V[perm])
            # Pass scalar isodepth values for negative downweighting (Technical Guide Eq.10)
            loss_cont = soft_info_nce_loss(
                z_m_batch, z_v_batch, d_batch.squeeze(1),
                temperature=temperature, sigma=sigma)

        current_lam = lam if epoch > warmup_epochs else 0.0
        loss_total = loss_recon + current_lam * loss_cont

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        history['recon'].append(loss_recon.item())
        history['contrastive'].append(loss_cont.item())
        history['total'].append(loss_total.item())

        if epoch % log_interval == 0:
            phase = "WARMUP" if epoch <= warmup_epochs else "JOINT "
            print(f"[{phase}] Epoch {epoch:5d}/{total_epochs} | "
                  f"Recon: {loss_recon.item():.4f} | "
                  f"Cont: {loss_cont.item():.4f} | "
                  f"Total: {loss_total.item():.4f}")

    return model, history

# ============================================================
# H&E patch extraction (unchanged)
# ============================================================
def extract_patches(slice_id, adata, he_dir, output_size=224, crop_multiplier=3.0):
    img = Image.open(f'{he_dir}/tissue_hires_image.png').convert('RGB')
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    with open(f'{he_dir}/scalefactors_json.json') as f:
        sf = json.load(f)
    scale = sf['tissue_hires_scalef']
    spot_d_fullres = sf['spot_diameter_fullres']
    spot_d_hires = spot_d_fullres * scale
    crop_radius = int(np.ceil(spot_d_hires * crop_multiplier / 2))

    print(f"  Hires image: {W_img}x{H_img}, scale={scale:.4f}")
    print(f"  Spot diameter (hires): {spot_d_hires:.1f} px")
    print(f"  Crop size: {2*crop_radius}x{2*crop_radius} -> {output_size}x{output_size}")

    pos_df = {}
    with open(f'{he_dir}/tissue_positions_list.txt') as f:
        for line in f:
            parts = line.strip().split(',')
            pos_df[parts[0]] = (float(parts[4]), float(parts[5]))

    barcodes = adata.obs_names.tolist()
    patches = np.zeros((len(barcodes), output_size, output_size, 3), dtype=np.uint8)

    for i, bc in enumerate(barcodes):
        if bc not in pos_df:
            continue
        row_fullres, col_fullres = pos_df[bc]
        row_hires = row_fullres * scale
        col_hires = col_fullres * scale
        r1 = max(0, int(row_hires - crop_radius))
        r2 = min(H_img, int(row_hires + crop_radius))
        c1 = max(0, int(col_hires - crop_radius))
        c2 = min(W_img, int(col_hires + crop_radius))
        crop = img_np[r1:r2, c1:c2]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        patches[i] = np.array(Image.fromarray(crop).resize((output_size, output_size), Image.LANCZOS))

    return patches

# ============================================================
# Vision feature extraction (unchanged)
# ============================================================
patch_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vision_features(patches, model, D_v, batch_size=64):
    model = model.to(device)
    N = len(patches)
    features = np.zeros((N, D_v), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_tensors = torch.stack([
            patch_transform(Image.fromarray(p)) for p in patches[start:end]
        ]).to(device)
        with torch.no_grad():
            feats = model(batch_tensors)
        features[start:end] = feats.cpu().numpy()
        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{N} patches...", end='\r')
    print(f"  Processed {N}/{N} patches.     ")
    return features

# ============================================================
# CGASTONWrapper for dp_related compatibility (unchanged)
# ============================================================
class CGASTONWrapper:
    def __init__(self, cgaston_model):
        self.spatial_embedding = cgaston_model.encoder
        self.expression_function = cgaston_model.decoder

# ============================================================
# MAIN PIPELINE
# ============================================================
print("="*60)
print("C-GASTON Soft-weighted InfoNCE — all 8 slices")
print(f"sigma={SIGMA_SOFT}, lambda={LAMBDA_CONTRASTIVE}, tau={TEMPERATURE}")
print("="*60)

# --- Step 1: Load pre-computed GLM-PCA + adata ---
print("\n[Step 1] Loading pre-computed GLM-PCA...")
data = {}
for sid in ALL_SLICES:
    print(f"\n--- {sid} ---")
    adata = ad.read_h5ad(f'{SLICE_DATA_DIRS[sid]}/{sid}.h5ad')
    adata.var_names_make_unique()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    S = np.asarray(adata.obsm['spatial'])
    gt_str = adata.obs['original_domain'].astype(str).values
    gt_int = np.array([LABEL_TO_INT.get(l, -1) for l in gt_str])

    A = np.load(f'{GLMPCA_UNIFIED}/{sid}/glmpca.npy')
    print(f"  Loaded GLM-PCA: A {A.shape} from {GLMPCA_UNIFIED}/{sid}/")

    S_torch, A_torch = load_rescale_input_data(S, A)
    data[sid] = {
        'adata': adata, 'coords': S, 'gt_labels': gt_int,
        'A_torch': A_torch, 'S_torch': S_torch,
    }
    print(f"  S_torch: {S_torch.shape}, A_torch: {A_torch.shape}")

# --- Step 2: H&E patches ---
print("\n[Step 2] Extracting H&E patches...")
for sid in ALL_SLICES:
    print(f"\n--- {sid} ---")
    patches = extract_patches(sid, data[sid]['adata'], HE_IMAGE_PATHS[sid],
                              output_size=PATCH_SIZE, crop_multiplier=3.0)
    data[sid]['patches'] = patches
    print(f"  Patches: {patches.shape}")

# --- Step 3: Vision features ---
print("\n[Step 3] Extracting vision features (ResNet-50)...")
vision_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
vision_model.fc = nn.Identity()
vision_model.eval()
for param in vision_model.parameters():
    param.requires_grad = False
D_v = 2048

for sid in ALL_SLICES:
    print(f"\n--- {sid} ---")
    features = extract_vision_features(data[sid]['patches'], vision_model, D_v)
    data[sid]['vision_features'] = features
    del data[sid]['patches']
    print(f"  Vision features: {features.shape}")

# --- Step 4: Train C-GASTON (soft InfoNCE) ---
print("\n[Step 4] Training C-GASTON with Soft-weighted InfoNCE...")
all_results = {}
for sid in ALL_SLICES:
    print(f"\n{'='*60}")
    print(f"Training C-GASTON (soft) on slice {sid}")
    print(f"{'='*60}")

    S_t = data[sid]['S_torch']
    A_t = data[sid]['A_torch']
    V_t = torch.tensor(data[sid]['vision_features'], dtype=torch.float32)

    best_loss = float('inf')
    best_model_state = None
    best_history = None

    for restart in range(NUM_RESTARTS):
        print(f"\n--- Restart {restart}/{NUM_RESTARTS-1} ---")
        mdl = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                      isodepth_arch=ISODEPTH_ARCH,
                      expression_arch=EXPRESSION_ARCH).to(device)

        mdl, hist = train_cgaston_soft(
            mdl, S_t, A_t, V_t,
            total_epochs=TOTAL_EPOCHS, warmup_epochs=WARMUP_EPOCHS,
            lam=LAMBDA_CONTRASTIVE, temperature=TEMPERATURE, sigma=SIGMA_SOFT,
            batch_size=BATCH_SIZE, lr=LR, log_interval=2000, seed=restart)

        final_recon = hist['recon'][-1]
        print(f"  Final recon loss: {final_recon:.6f}")
        if final_recon < best_loss:
            best_loss = final_recon
            best_model_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            best_history = hist
            print(f"  >>> New best! (loss={best_loss:.6f})")

    best_model = CGASTON(K=NUM_DIMS, D_v=D_v, D=EMBEDDING_DIM,
                         isodepth_arch=ISODEPTH_ARCH,
                         expression_arch=EXPRESSION_ARCH).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    save_dir = f'{OUTPUT_DIR}/{sid}'
    os.makedirs(save_dir, exist_ok=True)
    write_model_outputs(save_dir, best_model_state, best_model.state_dict())

    all_results[sid] = {'model': best_model, 'history': best_history, 'best_loss': best_loss}
    print(f"\n{sid}: Best model saved (recon loss = {best_loss:.6f})")

# --- Step 5: Evaluate ---
print("\n[Step 5] Evaluating...")
eval_results = {}
summary_rows = []
for sid in ALL_SLICES:
    print(f"\n--- Evaluating {sid} ---")
    mdl = all_results[sid]['model']
    mdl.eval()
    mdl_cpu = mdl.cpu()

    A_np = data[sid]['A_torch'].detach().numpy()
    S_np = data[sid]['S_torch'].detach().numpy()
    wrapper = CGASTONWrapper(mdl_cpu)

    gaston_isodepth, gaston_labels = dp_related.get_isodepth_labels(
        wrapper, A_np, S_np, NUM_LAYERS, num_buckets=100)

    gt = data[sid]['gt_labels']
    valid = gt >= 0
    ari = adjusted_rand_score(gt[valid], gaston_labels[valid])
    nmi = normalized_mutual_info_score(gt[valid], gaston_labels[valid])
    d_norm = (gaston_isodepth - gaston_isodepth.min()) / (gaston_isodepth.max() - gaston_isodepth.min() + 1e-8)
    sp, _ = spearmanr(d_norm[valid], gt[valid])
    mi = morans_i(d_norm, data[sid]['coords'], k=6)
    labels_int = gaston_labels.astype(int)

    eval_results[sid] = {
        'isodepth': gaston_isodepth, 'labels': labels_int,
        'ari': ari, 'nmi': nmi, 'spearman': abs(float(sp)), 'morans_i': float(mi),
    }

    summary_rows.append({
        'slice': sid,
        'ARI': round(ari, 4),
        'NMI': round(nmi, 4),
        'Spearman': round(abs(float(sp)), 4),
        'MoransI': round(float(mi), 4),
        'best_loss': round(all_results[sid]['best_loss'], 6),
    })

    write_prediction_outputs(f'{OUTPUT_DIR}/{sid}', gaston_isodepth, labels_int)
    mdl.to(device)

    print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}, Spearman: {abs(float(sp)):.4f}, Moran's I: {float(mi):.4f}")

write_summary_files(OUTPUT_DIR, summary_rows)
write_run_config(OUTPUT_DIR)

# --- Summary ---
print("\n" + "="*60)
print("RESULTS SUMMARY  (Soft-weighted InfoNCE)")
print("="*60)
for sid in ALL_SLICES:
    r = eval_results[sid]
    print(f"  {sid}: ARI={r['ari']:.4f}  NMI={r['nmi']:.4f}  Spearman={r['spearman']:.4f}  MoranI={r['morans_i']:.4f}  ReconLoss={all_results[sid]['best_loss']:.4f}")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("Done!")
