"""Rebuild v2/v3 predictions and compare both with the scored submission."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline
import pipeline_v3


RESULTS_DIR = ROOT / "experiments" / "results"
OFFICIAL_PATH = ROOT / "data" / "Group_27_Submission.csv"
V3_PATH = ROOT / pipeline_v3.SUBMISSION_PATH


def compare(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    if list(reference.columns) != ["Client_ID", "Drop_Probability"]:
        raise RuntimeError("unexpected reference schema")
    if list(candidate.columns) != ["Client_ID", "Drop_Probability"]:
        raise RuntimeError("unexpected candidate schema")
    if not reference["Client_ID"].equals(candidate["Client_ID"]):
        raise RuntimeError("Client_ID order mismatch")
    ref = reference["Drop_Probability"].to_numpy()
    cand = candidate["Drop_Probability"].to_numpy()
    return {
        "spearman": float(spearmanr(ref, cand).statistic),
        "max_abs_score_difference": float(np.max(np.abs(ref - cand))),
        "mean_abs_score_difference": float(np.mean(np.abs(ref - cand))),
        "exactly_equal": bool(np.array_equal(ref, cand)),
    }


def assert_feature_delta_only() -> None:
    train = pipeline.load_raw(str(ROOT / pipeline.TRAIN_PATH))
    test = pipeline.load_raw(str(ROOT / pipeline.TEST_PATH))
    freq = pipeline.make_freq_maps(train, test)
    v2 = pipeline.build_features(train.head(1000), freq)
    v3 = pipeline_v3.build_features(train.head(1000), freq)
    if not v2.drop(columns=["start_week"]).equals(v3):
        raise RuntimeError("v3 feature matrix differs beyond removal of start_week")


def main() -> None:
    assert_feature_delta_only()
    official = pd.read_csv(OFFICIAL_PATH)
    existing_v3 = pd.read_csv(V3_PATH) if V3_PATH.exists() else None
    print("rebuilding v2", flush=True)
    rebuilt_v2 = pipeline.run_final(write=False)
    print("building and writing v3", flush=True)
    rebuilt_v3 = pipeline_v3.run_final(str(V3_PATH), write=True)

    summary = {
        "official_rows": len(official),
        "v3_rows": len(rebuilt_v3),
        "v3_score_min": float(rebuilt_v3["Drop_Probability"].min()),
        "v3_score_max": float(rebuilt_v3["Drop_Probability"].max()),
        "v3_sha256": hashlib.sha256(V3_PATH.read_bytes()).hexdigest(),
        "rebuilt_v2_vs_official": compare(official, rebuilt_v2),
        "v3_vs_rebuilt_v2": compare(rebuilt_v2, rebuilt_v3),
        "v3_vs_official": compare(official, rebuilt_v3),
        "feature_delta": "start_week removed only",
        "xgb_delta": {
            "min_child_weight": 10,
            "subsample": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
        },
        "anchor": {
            "path": str(OFFICIAL_PATH.relative_to(ROOT)),
            "candidate_weight": pipeline_v3.CANDIDATE_WEIGHT,
        },
    }
    if existing_v3 is not None:
        summary["v3_rerun_vs_existing"] = compare(existing_v3, rebuilt_v3)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "final_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
