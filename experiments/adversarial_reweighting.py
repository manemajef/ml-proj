"""Hidden-test similarity diagnostics using label-free adversarial validation.

The propensity-weighted AUC is a stress test, not the primary selector: it
assumes covariate shift (stable P(y|x)) and clips extreme density ratios.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline
from temporal_validation import RESULTS_DIR, weighted_rank_average


SEED = 42
N_SPLITS = 3
WEIGHT_CLIP = 10.0
FINAL_XGB_CONFIG = "regularized_subsample_08"


def source_matrix(train: pd.DataFrame, test: pd.DataFrame):
    train = train.drop(columns=[pipeline.TARGET])
    combined = pd.concat(
        [train.assign(is_test=0), test.assign(is_test=1)], ignore_index=True
    )
    source = combined.pop("is_test").to_numpy()
    X = combined.drop(columns=["Client_ID", "Course_Start_Date"], errors="ignore")
    for col in X.select_dtypes(include=["object", "string"]).columns:
        cleaned = X[col].astype("string").str.strip().str.lower().fillna("missing")
        X[col] = cleaned.astype("category").cat.codes
    return X.fillna(-1), source


def adversarial_oof(X: pd.DataFrame, source: np.ndarray) -> np.ndarray:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    prediction = np.zeros(len(X), dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, source), start=1):
        print(f"adversarial fold {fold}/{N_SPLITS}", flush=True)
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], source[train_idx])
        prediction[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return prediction


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def main() -> None:
    train = pipeline.load_raw(str(ROOT / pipeline.TRAIN_PATH))
    test = pipeline.load_raw(str(ROOT / pipeline.TEST_PATH))
    X_source, source = source_matrix(train, test)
    source_probability = adversarial_oof(X_source, source)
    source_auc = roc_auc_score(source, source_probability)
    print(f"OOF adversarial AUC: {source_auc:.6f}")

    n_train = int((source == 0).sum())
    n_test = int((source == 1).sum())
    p = np.clip(source_probability[:n_train], 1e-6, 1 - 1e-6)
    density_ratio = (p / (1 - p)) * (n_train / n_test)
    weights = np.clip(density_ratio, 0, WEIGHT_CLIP)

    original = pd.read_pickle(RESULTS_DIR / "temporal_predictions.pkl.gz")
    original = original[original["variant"] == "uniform"]
    feature = pd.read_pickle(RESULTS_DIR / "feature_predictions_uniform.pkl.gz")
    feature = feature[feature["variant"] == "drop_start_week"]
    xgb = pd.read_pickle(RESULTS_DIR / "xgb_predictions.pkl.gz")
    xgb = xgb[xgb["config"] == FINAL_XGB_CONFIG]

    weight_by_client = pd.Series(weights, index=train["Client_ID"])
    prob_by_client = pd.Series(p, index=train["Client_ID"])
    rows = []
    for split in original["split"].unique():
        old = original[original["split"] == split].reset_index(drop=True)
        feat = feature[feature["split"] == split].reset_index(drop=True)
        tuned_xgb = xgb[xgb["split"] == split].reset_index(drop=True)
        y = old["target"].to_numpy()
        baseline = old["equal_rank_blend"].to_numpy()
        candidate = weighted_rank_average(
            {
                "lgbm": feat["lgbm"].to_numpy(),
                "xgb": tuned_xgb["xgb"].to_numpy(),
                "cat": feat["cat"].to_numpy(),
            }
        )
        split_weights = old["Client_ID"].map(weight_by_client).to_numpy()
        split_prob = old["Client_ID"].map(prob_by_client).to_numpy()
        baseline_auc = roc_auc_score(y, baseline, sample_weight=split_weights)
        candidate_auc = roc_auc_score(y, candidate, sample_weight=split_weights)
        rows.append(
            {
                "split": split,
                "mean_test_probability": split_prob.mean(),
                "median_test_probability": np.median(split_prob),
                "mean_clipped_weight": split_weights.mean(),
                "effective_n": effective_sample_size(split_weights),
                "weighted_baseline_auc": baseline_auc,
                "weighted_candidate_auc": candidate_auc,
                "weighted_delta": candidate_auc - baseline_auc,
            }
        )

    results = pd.DataFrame(rows)
    results.insert(0, "adversarial_oof_auc", source_auc)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "adversarial_weighted_metrics.csv", index=False)
    pd.DataFrame(
        {
            "Client_ID": train["Client_ID"],
            "test_probability_oof": p,
            "clipped_density_ratio": weights,
        }
    ).to_pickle(RESULTS_DIR / "adversarial_train_propensity.pkl.gz")
    print(results.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
