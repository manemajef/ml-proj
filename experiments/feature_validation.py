"""Screen small feature changes across rolling temporal windows.

This is intentionally separate from ``temporal_validation.py`` so the cached
baseline fits remain valid while feature candidates evolve.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
from temporal_validation import RESULTS_DIR, SPLITS, TemporalSplit, weighted_rank_average


TRAIN_PATH = ROOT / pipeline.TRAIN_PATH
MEMORY = joblib.Memory(ROOT / "cache" / "feature_validation", verbose=0)
MODEL_NAMES = ("lgbm", "xgb", "cat")

FEATURE_VARIANTS = (
    "add_assigned_lab",
    "add_company_id",
    "add_cyclic_calendar",
    "cyclic_calendar_no_week",
    "drop_start_week",
    "drop_frequency_encodings",
)


def apply_feature_variant(
    X: pd.DataFrame, raw: pd.DataFrame, variant: str
) -> pd.DataFrame:
    X = X.copy()
    normalized = pipeline.normalize_cats(raw)
    if variant == "add_assigned_lab":
        X["Assigned_Lab_Config"] = (
            normalized["Assigned_Lab_Config"].fillna("missing").astype("category")
        )
    elif variant == "add_company_id":
        X["Company_ID"] = normalized["Company_ID"].fillna("missing").astype("category")
    elif variant in {"add_cyclic_calendar", "cyclic_calendar_no_week"}:
        day = raw["Course_Start_Date"].dt.dayofyear.astype(float)
        angle = 2 * np.pi * day / 365.25
        X["year_sin"] = np.sin(angle)
        X["year_cos"] = np.cos(angle)
        if variant == "cyclic_calendar_no_week":
            X = X.drop(columns=["start_week"])
    elif variant == "drop_start_week":
        X = X.drop(columns=["start_week"])
    elif variant == "drop_frequency_encodings":
        X = X.drop(
            columns=["Agent_ID_freq", "Company_ID_freq", "Origin_Country_freq"]
        )
    else:
        raise ValueError(f"unknown feature variant: {variant}")
    return X


def recency_weights(dates: pd.Series, half_life_days: int | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp2(-age_days / half_life_days)
    return weights / weights.mean()


@MEMORY.cache
def fit_one(
    split: TemporalSplit,
    variant: str,
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

    freq_maps = pipeline.make_freq_maps(train_raw)
    X_train = apply_feature_variant(
        pipeline.build_features(train_raw, freq_maps), train_raw, variant
    )
    X_valid = apply_feature_variant(
        pipeline.build_features(valid_raw, freq_maps), valid_raw, variant
    )
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
        "seconds": time.perf_counter() - started,
    }


def evaluate(variants: list[str], models: tuple[str, ...], half_life_days: int | None):
    rows = []
    pred_rows = []
    for variant in variants:
        for split in SPLITS:
            predictions = {}
            target = None
            for model_name in models:
                print(f"fit {variant:>27} | {split.name} | {model_name}", flush=True)
                result = fit_one(split, variant, model_name, half_life_days)
                predictions[model_name] = result["prediction"]
                target = result["target"]
                rows.append(
                    {
                        "variant": variant,
                        "weighting": "uniform" if half_life_days is None else f"half_life_{half_life_days}d",
                        "split": split.name,
                        "model": model_name,
                        "auc": roc_auc_score(target, result["prediction"]),
                        "fit_seconds": result["seconds"],
                    }
                )
            assert target is not None
            if len(predictions) > 1:
                blend = weighted_rank_average(predictions)
                rows.append(
                    {
                        "variant": variant,
                        "weighting": "uniform" if half_life_days is None else f"half_life_{half_life_days}d",
                        "split": split.name,
                        "model": "equal_rank_blend",
                        "auc": roc_auc_score(target, blend),
                        "fit_seconds": sum(
                            fit_one(split, variant, name, half_life_days)["seconds"]
                            for name in models
                        ),
                    }
                )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "weighting": "uniform" if half_life_days is None else f"half_life_{half_life_days}d",
                        "split": split.name,
                        "target": target,
                        **predictions,
                    }
                )
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    suffix = "uniform" if half_life_days is None else f"half_life_{half_life_days}d"
    metrics.to_csv(RESULTS_DIR / f"feature_metrics_{suffix}.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_pickle(
        RESULTS_DIR / f"feature_predictions_{suffix}.pkl.gz"
    )
    print("\nSummary")
    print(
        metrics.groupby(["variant", "model"])["auc"]
        .agg(mean="mean", worst="min")
        .sort_values(["mean", "worst"], ascending=False)
        .round(6)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", choices=FEATURE_VARIANTS, default=list(FEATURE_VARIANTS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=["xgb"]
    )
    parser.add_argument("--half-life-days", type=int)
    args = parser.parse_args()
    evaluate(args.variants, tuple(args.models), args.half_life_days)


if __name__ == "__main__":
    main()
