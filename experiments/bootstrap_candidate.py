"""Paired bootstrap uncertainty for the final candidate versus v2 baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_validation import RESULTS_DIR, weighted_rank_average


N_BOOT = 1000
SEED = 20260715
FINAL_XGB_CONFIG = "regularized_subsample_08"
FINAL_CANDIDATE_WEIGHT = 0.75


def stratified_bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
) -> float:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    sample = np.concatenate(
        [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
    )
    return roc_auc_score(y[sample], candidate[sample]) - roc_auc_score(
        y[sample], baseline[sample]
    )


def main() -> None:
    original = pd.read_pickle(RESULTS_DIR / "temporal_predictions.pkl.gz")
    original = original[original["variant"] == "uniform"]
    feature = pd.read_pickle(RESULTS_DIR / "feature_predictions_uniform.pkl.gz")
    feature = feature[feature["variant"] == "drop_start_week"]
    xgb = pd.read_pickle(RESULTS_DIR / "xgb_predictions.pkl.gz")
    xgb = xgb[xgb["config"] == FINAL_XGB_CONFIG]

    rng = np.random.default_rng(SEED)
    boot_by_split: dict[str, np.ndarray] = {}
    rows = []
    for split in original["split"].unique():
        old = original[original["split"] == split].reset_index(drop=True)
        feat = feature[feature["split"] == split].reset_index(drop=True)
        tuned_xgb = xgb[xgb["split"] == split].reset_index(drop=True)
        if not old["target"].equals(feat["target"]) or not old["target"].equals(
            tuned_xgb["target"]
        ):
            raise RuntimeError(f"target/order mismatch for {split}")

        baseline = old["equal_rank_blend"].to_numpy()
        candidate = weighted_rank_average(
            {
                "lgbm": feat["lgbm"].to_numpy(),
                "xgb": tuned_xgb["xgb"].to_numpy(),
                "cat": feat["cat"].to_numpy(),
            }
        )
        baseline_rank = rankdata(baseline) / len(baseline)
        candidate_rank = rankdata(candidate) / len(candidate)
        candidate = (1 - FINAL_CANDIDATE_WEIGHT) * baseline_rank + (
            FINAL_CANDIDATE_WEIGHT * candidate_rank
        )
        y = old["target"].to_numpy()
        observed = roc_auc_score(y, candidate) - roc_auc_score(y, baseline)
        deltas = np.array(
            [stratified_bootstrap_delta(y, baseline, candidate, rng) for _ in range(N_BOOT)]
        )
        boot_by_split[split] = deltas
        rows.append(
            {
                "scope": split,
                "observed_delta": observed,
                "ci_2_5": np.quantile(deltas, 0.025),
                "ci_5": np.quantile(deltas, 0.05),
                "ci_95": np.quantile(deltas, 0.95),
                "ci_97_5": np.quantile(deltas, 0.975),
                "probability_positive": np.mean(deltas > 0),
            }
        )

    aggregate = np.mean(np.column_stack(list(boot_by_split.values())), axis=1)
    rows.append(
        {
            "scope": "mean_across_windows",
            "observed_delta": np.mean([row["observed_delta"] for row in rows]),
            "ci_2_5": np.quantile(aggregate, 0.025),
            "ci_5": np.quantile(aggregate, 0.05),
            "ci_95": np.quantile(aggregate, 0.95),
            "ci_97_5": np.quantile(aggregate, 0.975),
            "probability_positive": np.mean(aggregate > 0),
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "candidate_bootstrap.csv", index=False)
    np.savez_compressed(
        RESULTS_DIR / "candidate_bootstrap_deltas.npz",
        **boot_by_split,
        mean_across_windows=aggregate,
    )
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
