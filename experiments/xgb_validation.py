"""Small rolling-window XGBoost search around the scored configuration."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
from temporal_validation import RESULTS_DIR, SPLITS, TemporalSplit, weighted_rank_average


TRAIN_PATH = ROOT / pipeline.TRAIN_PATH
MEMORY = joblib.Memory(ROOT / "cache" / "xgb_validation", verbose=0)

CONFIGS = {
    "current": {},
    "depth_4": {"max_depth": 4},
    "depth_5": {"max_depth": 5},
    "depth_7": {"max_depth": 7},
    "child_10": {"min_child_weight": 10},
    "lambda_3": {"reg_lambda": 3.0},
    "regularized": {"min_child_weight": 10, "reg_lambda": 3.0, "reg_alpha": 0.1},
    "regularized_subsample_08": {
        "min_child_weight": 10,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "subsample": 0.8,
    },
    "regularized_subsample_08_seed17": {
        "min_child_weight": 10,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "subsample": 0.8,
        "random_state": 17,
    },
    "regularized_subsample_08_seed73": {
        "min_child_weight": 10,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1,
        "subsample": 0.8,
        "random_state": 73,
    },
    "budget_1000_lr_002": {"n_estimators": 1000, "learning_rate": 0.02},
    "budget_500_lr_004": {"n_estimators": 500, "learning_rate": 0.04},
    "subsample_08": {"subsample": 0.8},
}


@MEMORY.cache
def fit_one(split: TemporalSplit, config_name: str) -> dict:
    started = time.perf_counter()
    raw = pipeline.load_raw(str(TRAIN_PATH))
    start = pd.Timestamp(split.valid_start)
    end = pd.Timestamp(split.valid_end)
    train_raw = raw[raw["Course_Start_Date"] < start].copy()
    valid_raw = raw[
        (raw["Course_Start_Date"] >= start) & (raw["Course_Start_Date"] < end)
    ].copy()

    freq_maps = pipeline.make_freq_maps(train_raw)
    X_train = pipeline.build_features(train_raw, freq_maps).drop(columns=["start_week"])
    X_valid = pipeline.build_features(valid_raw, freq_maps).drop(columns=["start_week"])
    pipeline.align_categories(X_train, X_valid)

    model = pipeline.get_xgb(**CONFIGS[config_name])
    model.fit(X_train, train_raw[pipeline.TARGET].to_numpy())
    prediction = model.predict_proba(X_valid)[:, 1]
    return {
        "prediction": prediction,
        "target": valid_raw[pipeline.TARGET].to_numpy(),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    rows = []
    pred_rows = []
    for config_name in CONFIGS:
        for split in SPLITS:
            print(f"fit {config_name:>20} | {split.name}", flush=True)
            result = fit_one(split, config_name)
            rows.append(
                {
                    "config": config_name,
                    "split": split.name,
                    "xgb_auc": roc_auc_score(result["target"], result["prediction"]),
                    "fit_seconds": result["seconds"],
                }
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "config": config_name,
                        "split": split.name,
                        "target": result["target"],
                        "xgb": result["prediction"],
                    }
                )
            )

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(pred_rows, ignore_index=True)

    # Reuse the already fitted LightGBM and CatBoost predictions for the same
    # drop-start-week feature matrix; only the XGBoost component changes.
    base = pd.read_pickle(RESULTS_DIR / "feature_predictions_uniform.pkl.gz")
    base = base[base["variant"] == "drop_start_week"]
    blend_rows = []
    for config_name, config_df in predictions.groupby("config", sort=False):
        for split, xgb_df in config_df.groupby("split", sort=False):
            companion = base[base["split"] == split]
            if not xgb_df["target"].reset_index(drop=True).equals(
                companion["target"].reset_index(drop=True)
            ):
                raise RuntimeError(f"target/order mismatch for {config_name} {split}")
            blend = weighted_rank_average(
                {
                    "lgbm": companion["lgbm"].to_numpy(),
                    "xgb": xgb_df["xgb"].to_numpy(),
                    "cat": companion["cat"].to_numpy(),
                }
            )
            blend_rows.append(
                {
                    "config": config_name,
                    "split": split,
                    "blend_auc": roc_auc_score(xgb_df["target"], blend),
                }
            )

    blend_metrics = pd.DataFrame(blend_rows)
    metrics = metrics.merge(blend_metrics, on=["config", "split"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "xgb_metrics.csv", index=False)
    predictions.to_pickle(RESULTS_DIR / "xgb_predictions.pkl.gz")
    (RESULTS_DIR / "xgb_configs.json").write_text(
        json.dumps(CONFIGS, indent=2) + "\n", encoding="utf-8"
    )

    summary = metrics.groupby("config").agg(
        xgb_mean=("xgb_auc", "mean"),
        xgb_worst=("xgb_auc", "min"),
        blend_mean=("blend_auc", "mean"),
        blend_worst=("blend_auc", "min"),
    )
    recent = metrics[metrics["split"] == "2017_jan_apr"].set_index("config")
    summary["blend_recent"] = recent["blend_auc"]
    print("\nSummary")
    print(
        summary.sort_values(["blend_mean", "blend_worst"], ascending=False)
        .round(6)
        .to_string()
    )


if __name__ == "__main__":
    main()
