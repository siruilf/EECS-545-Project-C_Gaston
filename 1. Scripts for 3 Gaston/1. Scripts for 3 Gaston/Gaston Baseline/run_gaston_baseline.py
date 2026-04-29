#!/usr/bin/env python3
"""
Standalone original GASTON baseline runner.

This script trains an independent GASTON model on the 8 DLPFC slices using the
official gaston.neural_net.GASTON / gaston.neural_net.train API. It is intended
to produce an apples-to-apples baseline for the main C-GASTON experiments:

- same 8 slices
- same GLM-PCA targets
- same evaluation metrics
- configurable restarts / epochs for smoke tests or full runs

Outputs per slice:
    GASTON_baseline_results/{slice_id}/
        model.pt
        isodepth.npy
        labels.npy
        gaston_best.pt
        gaston_isodepth.npy
        gaston_labels.npy
        gaston_loss_history.npy

Workspace-level outputs:
    GASTON_baseline_results/results_summary.csv
    GASTON_baseline_results/aggregate_summary.csv
    GASTON_baseline_results/run_config.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from gaston import dp_related, neural_net
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


ALL_SLICES = [
    "151507", "151508", "151509", "151510",
    "151673", "151674", "151675", "151676",
]

SAMPLE_BY_SLICE = {
    "151507": "Sample1",
    "151508": "Sample1",
    "151509": "Sample1",
    "151510": "Sample1",
    "151673": "Sample3",
    "151674": "Sample3",
    "151675": "Sample3",
    "151676": "Sample3",
}

LABEL_TO_INT = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "L6": 5, "WM": 6}
SUMMARY_FIELDS = ["slice", "ARI", "NMI", "Spearman", "MoransI", "best_loss"]


def find_repo_root(start_path: Path) -> Path:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / "DLPFC_Datasets").is_dir() and (current / "glmpca_results").is_dir():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root containing DLPFC_Datasets and glmpca_results.")
        current = current.parent


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Train and evaluate an original GASTON baseline on the DLPFC slices.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=find_repo_root(script_dir),
        help="Workspace root containing DLPFC_Datasets, glmpca_results, and output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory. Defaults to BASE_DIR/GASTON_baseline_results.",
    )
    parser.add_argument(
        "--slices",
        nargs="*",
        default=ALL_SLICES,
        help="Subset of slice IDs to run. Default: all 8 slices.",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Training epochs per restart.")
    parser.add_argument("--restarts", type=int, default=10, help="Random restarts per slice.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size for official gaston.neural_net.train. Use 0 for full-batch.",
    )
    parser.add_argument("--checkpoint", type=int, default=2000, help="Progress print interval in epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--optim",
        choices=["adam", "sgd", "adagrad"],
        default="adam",
        help="Optimizer passed to gaston.neural_net.train.",
    )
    parser.add_argument(
        "--s-hidden",
        nargs="*",
        type=int,
        default=[20, 20],
        help="Hidden layer sizes for the spatial embedding f_S.",
    )
    parser.add_argument(
        "--a-hidden",
        nargs="*",
        type=int,
        default=[20, 20],
        help="Hidden layer sizes for the expression function f_A.",
    )
    parser.add_argument("--num-layers", type=int, default=7, help="Number of cortical domains Q.")
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=100,
        help="Bucket count for dp_related.get_isodepth_labels.",
    )
    parser.add_argument(
        "--embed-size",
        type=int,
        default=4,
        help="Positional encoding embed_size passed to gaston.GASTON.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Positional encoding sigma passed to gaston.GASTON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override. Default: auto-detect.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Base random seed added to each restart index.",
    )
    return parser.parse_args()


def validate_slices(slices: list[str]) -> list[str]:
    invalid = sorted(set(slices) - set(ALL_SLICES))
    if invalid:
        raise ValueError(f"Unsupported slice IDs: {invalid}")
    return slices


def resolve_paths(base_dir: Path) -> dict[str, Path]:
    base_dir = find_repo_root(base_dir)
    return {
        "glmpca_dir": base_dir / "glmpca_results",
        "dataset_dir": base_dir / "DLPFC_Datasets",
    }


def h5ad_path_for_slice(dataset_dir: Path, slice_id: str) -> Path:
    sample = SAMPLE_BY_SLICE[slice_id]
    return dataset_dir / sample / "h5ad_cordinate_data" / f"{slice_id}.h5ad"


def compute_morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]

    weights = np.zeros((len(values), len(values)), dtype=np.float32)
    for row_idx in range(len(values)):
        weights[row_idx, idx[row_idx]] = 1.0 / k

    centered = values - values.mean()
    return float((len(values) / weights.sum()) * (centered @ weights @ centered) / (centered @ centered))


def compute_metrics(
    isodepth: np.ndarray,
    labels: np.ndarray,
    gt_labels: np.ndarray,
    coords: np.ndarray,
) -> tuple[float, float, float, float]:
    valid = gt_labels >= 0
    ari = adjusted_rand_score(gt_labels[valid], labels[valid])
    nmi = normalized_mutual_info_score(gt_labels[valid], labels[valid])
    normalized_depth = (isodepth - isodepth.min()) / (isodepth.max() - isodepth.min() + 1e-8)
    spearman_r, _ = spearmanr(normalized_depth[valid], gt_labels[valid])
    morans_i = compute_morans_i(normalized_depth, coords, k=6)
    return float(ari), float(nmi), abs(float(spearman_r)), float(morans_i)


def load_slice_data(slice_id: str, paths: dict[str, Path]) -> dict[str, object]:
    h5ad_path = h5ad_path_for_slice(paths["dataset_dir"], slice_id)
    glmpca_path = paths["glmpca_dir"] / slice_id / "glmpca.npy"

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad file for slice {slice_id}: {h5ad_path}")
    if not glmpca_path.exists():
        raise FileNotFoundError(f"Missing GLM-PCA file for slice {slice_id}: {glmpca_path}")

    adata = ad.read_h5ad(str(h5ad_path))
    adata.var_names_make_unique()

    coords = np.asarray(adata.obsm["spatial"])
    gt_str = adata.obs["original_domain"].astype(str).values
    gt_int = np.array([LABEL_TO_INT.get(label, -1) for label in gt_str])
    glmpca = np.load(glmpca_path)
    coords_torch, glmpca_torch = neural_net.load_rescale_input_data(coords, glmpca)

    return {
        "coords": coords,
        "gt_labels": gt_int,
        "coords_torch": coords_torch,
        "glmpca_torch": glmpca_torch,
    }


def train_best_restart(
    slice_data: dict[str, object],
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, np.ndarray, float]:
    coords_torch = slice_data["coords_torch"]
    glmpca_torch = slice_data["glmpca_torch"]
    num_genes = glmpca_torch.shape[1]

    best_restart_loss = float("inf")
    best_state_dict = None
    best_loss_history = None

    batch_size = None if args.batch_size <= 0 else args.batch_size

    for restart in range(args.restarts):
        seed = args.seed_offset + restart
        print(f"    restart {restart + 1}/{args.restarts} (seed={seed})")
        model, loss_history = neural_net.train(
            coords_torch,
            glmpca_torch,
            gaston_model=None,
            S_hidden_list=args.s_hidden,
            A_hidden_list=args.a_hidden,
            epochs=args.epochs,
            batch_size=batch_size,
            checkpoint=args.checkpoint,
            save_dir=None,
            loss_reduction="mean",
            optim=args.optim,
            lr=args.lr,
            seed=seed,
            save_final=False,
            embed_size=args.embed_size,
            sigma=args.sigma,
            device=args.device,
        )

        restart_loss = float(np.min(loss_history))
        final_loss = float(loss_history[-1])
        print(f"      min loss: {restart_loss:.6f} | final loss: {final_loss:.6f}")

        if restart_loss < best_restart_loss:
            best_restart_loss = restart_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_loss_history = np.asarray(loss_history, dtype=np.float32)
            print(f"      new best restart")

    if best_state_dict is None or best_loss_history is None:
        raise RuntimeError("No GASTON restart completed successfully.")

    best_model = neural_net.GASTON(
        num_genes,
        args.s_hidden,
        args.a_hidden,
        embed_size=args.embed_size,
        sigma=args.sigma,
    )
    best_model.load_state_dict(best_state_dict)
    best_model = best_model.cpu()
    best_model.eval()
    return best_model, best_loss_history, best_restart_loss


def evaluate_slice(
    model: torch.nn.Module,
    slice_data: dict[str, object],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    glmpca_np = slice_data["glmpca_torch"].detach().cpu().numpy()
    coords_np = slice_data["coords_torch"].detach().cpu().numpy()

    isodepth, labels = dp_related.get_isodepth_labels(
        model,
        glmpca_np,
        coords_np,
        args.num_layers,
        num_buckets=args.num_buckets,
    )
    labels = labels.astype(int)
    metrics = compute_metrics(isodepth, labels, slice_data["gt_labels"], slice_data["coords"])
    return isodepth, labels, metrics


def write_per_slice_outputs(
    output_dir: Path,
    slice_id: str,
    model: torch.nn.Module,
    isodepth: np.ndarray,
    labels: np.ndarray,
    loss_history: np.ndarray,
    args: argparse.Namespace,
) -> None:
    slice_dir = output_dir / slice_id
    slice_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "s_hidden": args.s_hidden,
        "a_hidden": args.a_hidden,
        "embed_size": args.embed_size,
        "sigma": args.sigma,
    }
    model_payload = {
        "model_type": "gaston_baseline",
        "state_dict": model.state_dict(),
        "s_hidden": args.s_hidden,
        "a_hidden": args.a_hidden,
        "embed_size": args.embed_size,
        "sigma": args.sigma,
        "num_layers": args.num_layers,
        "num_buckets": args.num_buckets,
    }
    torch.save(model_payload, slice_dir / "model.pt")
    torch.save(payload, slice_dir / "gaston_best.pt")
    np.save(slice_dir / "isodepth.npy", isodepth)
    np.save(slice_dir / "labels.npy", labels)
    np.save(slice_dir / "gaston_isodepth.npy", isodepth)
    np.save(slice_dir / "gaston_labels.npy", labels)
    np.save(slice_dir / "gaston_loss_history.npy", loss_history)


def population_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_summary_files(output_dir: Path, rows: list[dict[str, float]]) -> None:
    with open(output_dir / "results_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_row = {}
    for metric in ["ARI", "NMI", "Spearman", "MoransI", "best_loss"]:
        mean_value, std_value = population_mean_std([row[metric] for row in rows])
        aggregate_row[f"{metric}_mean"] = round(mean_value, 4)
        aggregate_row[f"{metric}_std"] = round(std_value, 4)

    with open(output_dir / "aggregate_summary.csv", "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(aggregate_row.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(aggregate_row)


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    config = {
        "base_dir": str(args.base_dir),
        "output_dir": str(args.output_dir),
        "slices": args.slices,
        "epochs": args.epochs,
        "restarts": args.restarts,
        "batch_size": None if args.batch_size <= 0 else args.batch_size,
        "checkpoint": args.checkpoint,
        "lr": args.lr,
        "optim": args.optim,
        "s_hidden": args.s_hidden,
        "a_hidden": args.a_hidden,
        "num_layers": args.num_layers,
        "num_buckets": args.num_buckets,
        "embed_size": args.embed_size,
        "sigma": args.sigma,
        "device": args.device,
        "seed_offset": args.seed_offset,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.base_dir = find_repo_root(args.base_dir)
    args.slices = validate_slices(args.slices)
    args.output_dir = (args.output_dir or (args.base_dir / "GASTON_baseline_results")).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_paths(args.base_dir)

    print("=" * 60)
    print("Original GASTON baseline")
    print("=" * 60)
    print(f"Base dir:    {args.base_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Slices:      {args.slices}")
    print(f"Epochs:      {args.epochs}")
    print(f"Restarts:    {args.restarts}")
    print(f"Optimizer:   {args.optim}")
    print(f"LR:          {args.lr}")
    print(f"S hidden:    {args.s_hidden}")
    print(f"A hidden:    {args.a_hidden}")
    print(f"Buckets:     {args.num_buckets}")
    print("=" * 60)

    all_rows = []

    for slice_id in args.slices:
        print(f"\n--- Loading slice {slice_id} ---")
        slice_data = load_slice_data(slice_id, paths)
        print(
            f"  coords: {slice_data['coords_torch'].shape} | "
            f"glmpca: {slice_data['glmpca_torch'].shape}")

        print(f"--- Training slice {slice_id} ---")
        best_model, best_loss_history, best_loss = train_best_restart(slice_data, args)

        print(f"--- Evaluating slice {slice_id} ---")
        isodepth, labels, metrics = evaluate_slice(best_model, slice_data, args)
        ari, nmi, spearman, morans_i = metrics
        print(
            f"  ARI={ari:.4f} NMI={nmi:.4f} "
            f"Spearman={spearman:.4f} Moran={morans_i:.4f} "
            f"best_loss={best_loss:.6f}")

        write_per_slice_outputs(
            args.output_dir,
            slice_id,
            best_model,
            isodepth,
            labels,
            best_loss_history,
            args,
        )

        all_rows.append({
            "slice": slice_id,
            "ARI": round(ari, 4),
            "NMI": round(nmi, 4),
            "Spearman": round(spearman, 4),
            "MoransI": round(morans_i, 4),
            "best_loss": round(best_loss, 6),
        })

    write_summary_files(args.output_dir, all_rows)
    write_run_config(args.output_dir, args)

    print("\n" + "=" * 60)
    print("Aggregate summary (population mean +/- std)")
    print("=" * 60)
    for metric in ["ARI", "NMI", "Spearman", "MoransI"]:
        mean_value, std_value = population_mean_std([row[metric] for row in all_rows])
        print(f"{metric:>10}: {mean_value:.4f} +/- {std_value:.4f}")
    print("=" * 60)
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()