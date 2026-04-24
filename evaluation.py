"""
evaluate_dielectric.py
----------------------
Loads the test_results.pkl produced by train.py and computes:
  - MSE  : mean squared error over the full spectrum
  - MAE  : mean absolute error over the full spectrum
  - SC   : spectral similarity coefficient

Similarity Coefficient (SC) per sample i:
    SC_i = 1 - trapz(|pred_i - target_i|) / trapz(|target_i|)

The script breaks MSE/MAE/SC down separately for the imaginary (epsI)
and real (epsR) halves of y (first and second 2001 elements respectively).

Usage:
    python evaluation.py --results runs/exp_number/logs/test_results.pkl
"""

import argparse
import pickle
import numpy as np


L = 2001  # number of frequency points per spectral component


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """
    Mean MSE (averaged over all samples and freq points) and
    median of per-sample MSEs (mean over freq points per sample).
    """
    mean_val   = float(np.mean((preds - targets) ** 2))
    median_val = float(np.median(np.mean((preds - targets) ** 2, axis=1)))
    return mean_val, median_val


def mae(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """
    Mean MAE (averaged over all samples and freq points) and
    median of per-sample MAEs (mean over freq points per sample).
    """
    mean_val   = float(np.mean(np.abs(preds - targets)))
    median_val = float(np.median(np.mean(np.abs(preds - targets), axis=1)))
    return mean_val, median_val


def similarity_coefficient(preds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Per-sample SC = 1 - trapz(|pred - target|) / trapz(|target|)

    Args:
        preds   : [N, L]
        targets : [N, L]

    Returns:
        per_sample_sc : [N]
        mean_sc       : float
        median_sc     : float
    """
    per_sample_sc = (
        1 - np.trapz(np.abs(preds - targets), axis=1)
          / np.trapz(np.abs(targets), axis=1)
    )
    return per_sample_sc, float(np.mean(per_sample_sc)), float(np.median(per_sample_sc))


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_metrics(label: str, preds: np.ndarray, targets: np.ndarray) -> None:
    mse_mean, mse_median = mse(preds, targets)
    mae_mean, mae_median = mae(preds, targets)
    per_sc, mean_sc, median_sc = similarity_coefficient(preds, targets)

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  MAE  : {mae_mean:.6f}  (mean)")
    print(f"  MAE  : {mae_median:.6f}  (median)")
    print(f"  MSE  : {mse_mean:.6f}  (mean)")
    print(f"  MSE  : {mse_median:.6f}  (median)")
    print(f"  SC   : {mean_sc:.6f}  (mean)")
    print(f"  SC   : {median_sc:.6f}  (median)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate dielectric spectrum predictions.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to test_results.pkl produced by train_dielectric.py",
    )
    args = parser.parse_args()

    with open(args.results, "rb") as f:
        results = pickle.load(f)

    preds   = results["predictions"]   # [N, 4002]
    targets = results["targets"]       # [N, 4002]

    assert preds.shape == targets.shape, (
        f"Shape mismatch: predictions {preds.shape} vs targets {targets.shape}"
    )
    assert preds.shape[1] == 2 * L, (
        f"Expected spectrum length {2 * L}, got {preds.shape[1]}"
    )

    N = preds.shape[0]
    print(f"\nEvaluating {N} test samples  |  spectrum length = {2 * L} ({L} epsI + {L} epsR)")

    # epsI (imaginary part) — first L elements
    print_metrics("epsI  (imaginary, [:2001])", preds[:, :L], targets[:, :L])

    # epsR (real part) — last L elements
    print_metrics("epsR  (real,     [2001:])", preds[:, L:], targets[:, L:])

    print(f"\n{'─' * 50}\n")


if __name__ == "__main__":
    main()