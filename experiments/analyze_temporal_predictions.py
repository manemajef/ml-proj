"""Analyze saved temporal predictions without refitting models."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from temporal_validation import RESULTS_DIR, weighted_rank_average


PREDICTIONS_PATH = RESULTS_DIR / "temporal_predictions.pkl.gz"
GRID_PATH = RESULTS_DIR / "blend_weight_grid.csv"
MODEL_NAMES = ("lgbm", "xgb", "cat")
RECENT_SPLIT = "2017_jan_apr"


def evaluate_weight_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, variant_df in predictions.groupby("variant", sort=False):
        for raw_weights in product(range(5), repeat=len(MODEL_NAMES)):
            if sum(raw_weights) == 0 or sum(w > 0 for w in raw_weights) < 2:
                continue
            weights = dict(zip(MODEL_NAMES, raw_weights, strict=True))
            split_aucs = {}
            for split, split_df in variant_df.groupby("split", sort=False):
                pred = weighted_rank_average(
                    {name: split_df[name].to_numpy() for name in MODEL_NAMES},
                    weights,
                )
                split_aucs[split] = roc_auc_score(split_df["target"], pred)
            rows.append(
                {
                    "variant": variant,
                    "lgbm_weight": weights["lgbm"],
                    "xgb_weight": weights["xgb"],
                    "cat_weight": weights["cat"],
                    "mean_auc": sum(split_aucs.values()) / len(split_aucs),
                    "worst_auc": min(split_aucs.values()),
                    "recent_auc": split_aucs[RECENT_SPLIT],
                    **{f"auc_{name}": value for name, value in split_aucs.items()},
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        ["variant", "lgbm_weight", "xgb_weight", "cat_weight"]
    )


def main() -> None:
    predictions = pd.read_pickle(PREDICTIONS_PATH)
    grid = evaluate_weight_grid(predictions)
    grid.to_csv(GRID_PATH, index=False)

    columns = [
        "variant",
        "lgbm_weight",
        "xgb_weight",
        "cat_weight",
        "mean_auc",
        "worst_auc",
        "recent_auc",
    ]
    print("Top mean AUC")
    print(
        grid.sort_values(["mean_auc", "worst_auc"], ascending=False)[columns]
        .head(15)
        .round(6)
        .to_string(index=False)
    )
    print("\nTop worst-split AUC")
    print(
        grid.sort_values(["worst_auc", "mean_auc"], ascending=False)[columns]
        .head(15)
        .round(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
