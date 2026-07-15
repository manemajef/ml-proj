"""Focused rolling-temporal validation for the scored submission pipeline.

The script deliberately imports feature and model code from ``pipeline.py`` so
the baseline measured here is the same implementation that produced the scored
submission. Expensive fits are cached under ``cache/temporal_validation`` and
compact metrics/predictions are written to ``experiments/results``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline


TRAIN_PATH = ROOT / pipeline.TRAIN_PATH
CACHE_DIR = ROOT / "cache" / "temporal_validation"
RESULTS_DIR = ROOT / "experiments" / "results"
MEMORY = joblib.Memory(CACHE_DIR, verbose=0)


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    valid_start: str
    valid_end: str


SPLITS = (
    TemporalSplit("2016_may_aug", "2016-05-01", "2016-09-01"),
    TemporalSplit("2016_sep_dec", "2016-09-01", "2017-01-01"),
    TemporalSplit("2017_jan_apr", "2017-01-01", "2017-04-27"),
)

WEIGHTING_VARIANTS = {
    "uniform": None,
    "half_life_720d": 720,
    "half_life_365d": 365,
    "half_life_180d": 180,
}

MODEL_NAMES = ("lgbm", "xgb", "cat")


def recency_weights(dates: pd.Series, half_life_days: int | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp2(-age_days / half_life_days)
    return weights / weights.mean()


def weighted_rank_average(
    predictions: dict[str, np.ndarray], weights: dict[str, float] | None = None
) -> np.ndarray:
    from scipy.stats import rankdata

    if weights is None:
        weights = {name: 1.0 for name in predictions}
    total = sum(weights[name] for name in predictions)
    return sum(
        weights[name] * rankdata(pred) / len(pred)
        for name, pred in predictions.items()
    ) / total


@MEMORY.cache
def fit_one(
    split: TemporalSplit,
    model_name: str,
    half_life_days: int | None,
) -> dict:
    started = time.perf_counter()
    raw = pipeline.load_raw(str(TRAIN_PATH))
    start = pd.Timestamp(split.valid_start)
    end = pd.Timestamp(split.valid_end)
    train_raw = raw[raw["Course_Start_Date"] < start].copy()
    valid_raw = raw[
        (raw["Course_Start_Date"] >= start) & (raw["Course_Start_Date"] < end)
    ].copy()

    # Validation-time frequencies must be learned from the past only.
    freq_maps = pipeline.make_freq_maps(train_raw)
    X_train = pipeline.build_features(train_raw, freq_maps)
    X_valid = pipeline.build_features(valid_raw, freq_maps)
    pipeline.align_categories(X_train, X_valid)

    sample_weight = recency_weights(train_raw["Course_Start_Date"], half_life_days)
    prediction = pipeline.fit_predict(
        model_name,
        X_train,
        train_raw[pipeline.TARGET].to_numpy(),
        X_valid,
        sample_weight=sample_weight,
    )
    return {
        "prediction": prediction,
        "target": valid_raw[pipeline.TARGET].to_numpy(),
        "client_id": valid_raw["Client_ID"].to_numpy(),
        "valid_date": valid_raw["Course_Start_Date"].to_numpy(),
        "n_train": len(train_raw),
        "n_valid": len(valid_raw),
        "valid_rate": float(valid_raw[pipeline.TARGET].mean()),
        "seconds": time.perf_counter() - started,
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for variant, half_life_days in WEIGHTING_VARIANTS.items():
        for split in SPLITS:
            predictions: dict[str, np.ndarray] = {}
            split_result = None
            for model_name in MODEL_NAMES:
                print(f"fit {variant:>14} | {split.name} | {model_name}", flush=True)
                result = fit_one(split, model_name, half_life_days)
                split_result = result
                predictions[model_name] = result["prediction"]
                metric_rows.append(
                    {
                        "variant": variant,
                        "split": split.name,
                        "model": model_name,
                        "auc": roc_auc_score(result["target"], result["prediction"]),
                        "n_train": result["n_train"],
                        "n_valid": result["n_valid"],
                        "valid_rate": result["valid_rate"],
                        "fit_seconds": result["seconds"],
                    }
                )

            assert split_result is not None
            blend = weighted_rank_average(predictions)
            metric_rows.append(
                {
                    "variant": variant,
                    "split": split.name,
                    "model": "equal_rank_blend",
                    "auc": roc_auc_score(split_result["target"], blend),
                    "n_train": split_result["n_train"],
                    "n_valid": split_result["n_valid"],
                    "valid_rate": split_result["valid_rate"],
                    "fit_seconds": sum(
                        fit_one(split, name, half_life_days)["seconds"]
                        for name in MODEL_NAMES
                    ),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "split": split.name,
                        "Client_ID": split_result["client_id"],
                        "Course_Start_Date": split_result["valid_date"],
                        "target": split_result["target"],
                        **predictions,
                        "equal_rank_blend": blend,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "temporal_metrics.csv", index=False)
    predictions.to_pickle(RESULTS_DIR / "temporal_predictions.pkl.gz")

    metadata = {
        "splits": [asdict(split) for split in SPLITS],
        "weighting_variants": WEIGHTING_VARIANTS,
        "models": MODEL_NAMES,
        "metric": "roc_auc",
        "frequency_map_policy": "past training window only",
    }
    (RESULTS_DIR / "temporal_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, predictions


def print_summary(metrics: pd.DataFrame) -> None:
    blend = metrics[metrics["model"] == "equal_rank_blend"]
    summary = blend.groupby("variant")["auc"].agg(
        mean="mean", worst="min", best="max"
    )
    recent = (
        blend[blend["split"] == "2017_jan_apr"]
        .set_index("variant")["auc"]
        .rename("recent")
    )
    summary = summary.join(recent).sort_values(["mean", "worst"], ascending=False)
    print("\nEqual-rank blend summary")
    print(summary.round(6).to_string())
    print("\nPer-split blend AUC")
    print(
        blend.pivot(index="variant", columns="split", values="auc")
        .round(6)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear-cache", action="store_true", help="remove cached model fits first"
    )
    args = parser.parse_args()
    if args.clear_cache:
        MEMORY.clear(warn=False)
    metrics, _ = evaluate()
    print_summary(metrics)


if __name__ == "__main__":
    main()
