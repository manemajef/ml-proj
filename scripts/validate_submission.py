"""Reproduce and compare the final notebook submission with a protected target.

The notebook writes ``data/Group_27_Submission.csv``. The canonical prediction
vector is stored separately at ``data/val/desired_submission.csv``, so running
the notebook can never silently replace the reference used by this test.

This validates model configuration and prediction reproducibility. It cannot
validate hidden-test ROC-AUC because the hidden labels are unavailable.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
from io import StringIO
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


# Marimo normally selects an interactive backend. Validation runs headlessly.
os.environ.setdefault("MPLBACKEND", "Agg")


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook.py"
NOTEBOOK_OUTPUT_CSV = ROOT / "data" / "Group_27_Submission.csv"
REFERENCE_CSV = ROOT / "data" / "val" / "desired_submission.csv"
REFERENCE_SHA256 = "41a8bd3fd0098ae957e96ab5a787cbbbe1bf3fd529b6ba5b58142f78fc3d5104"

EXPECTED_XGB = {
    "min_child_weight": 10,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 700,
}
EXPECTED_FEATURES = {"start_week", "days_since_epoch"}
EXPECTED_SELECTION = {
    "selected_depth": 6,
    "selected_learning_rate": 0.03,
    "selected_n_trees": 700,
    "selected_profile_name": "Stronger",
}

# Rank agreement matters most because the project metric is ROC-AUC. The value
# tolerances allow small implementation/hardware differences while rejecting
# the historical v2 and no-week v3 vectors.
MIN_SPEARMAN = 0.9999
MAX_MEAN_ABS_DIFF = 0.005
MAX_ABS_DIFF = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = ["Client_ID", "Drop_Probability"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"{path} has columns {list(frame.columns)!r}; expected {expected_columns!r}"
        )
    if frame["Client_ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate Client_ID values")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    scores = frame["Drop_Probability"].to_numpy()
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f"{path} contains invalid scores")
    return frame


def import_notebook():
    spec = importlib.util.spec_from_file_location("submission_notebook", NOTEBOOK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {NOTEBOOK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_workdir(workdir: Path, *, fresh: bool) -> None:
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True)
    for name in ("Train_Data.csv", "Test_Data_No_Target.csv"):
        (data_dir / name).symlink_to(ROOT / "data" / name)

    if not fresh:
        shared_cache = ROOT / ".cache" / "joblib"
        if shared_cache.exists():
            cache_parent = workdir / ".cache"
            cache_parent.mkdir()
            (cache_parent / "joblib").symlink_to(shared_cache, target_is_directory=True)


def run_notebook_isolated(
    *, fresh: bool
) -> tuple[Path, object, str, TemporaryDirectory[str]]:
    module = import_notebook()
    temporary = TemporaryDirectory(prefix="ml-proj-submission-")
    workdir = Path(temporary.name)
    prepare_workdir(workdir, fresh=fresh)

    previous_cwd = Path.cwd()
    captured = StringIO()
    try:
        os.chdir(workdir)
        with redirect_stdout(captured), redirect_stderr(captured):
            _, definitions = module.app.run()
    except Exception:
        print(captured.getvalue(), file=sys.stderr)
        temporary.cleanup()
        raise
    finally:
        os.chdir(previous_cwd)

    candidate = workdir / "data" / "Group_27_Submission.csv"
    if not candidate.exists():
        temporary.cleanup()
        raise RuntimeError("The notebook completed without producing the submission CSV")

    # Return the handle so the caller can read/copy the candidate before cleanup.
    return candidate, definitions, captured.getvalue(), temporary


def validate_model_configuration(definitions: object) -> list[str]:
    errors: list[str] = []
    for name, expected in EXPECTED_SELECTION.items():
        actual = definitions.get(name)
        if actual != expected:
            errors.append(f"{name}={actual!r}; expected {expected!r}")

    regularization_comparison = definitions.get("regularization_comparison")
    if regularization_comparison is None:
        errors.append("notebook definitions do not contain regularization_comparison")
    else:
        selected_profiles = regularization_comparison.loc[
            regularization_comparison["selected"], "profile"
        ].tolist()
        if selected_profiles != ["Stronger"]:
            errors.append(
                f"regularization comparison selected {selected_profiles!r}; "
                "expected ['Stronger']"
            )

    blend = definitions.get("selected_blend")
    if blend is None:
        errors.append("notebook definitions do not contain selected_blend")
        return errors

    feature_names = set(getattr(blend, "feature_names_in_", []))
    missing_features = EXPECTED_FEATURES - feature_names
    if missing_features:
        errors.append(f"selected blend is missing features: {sorted(missing_features)}")

    component_params = getattr(blend, "component_params", {})
    xgb_params = component_params.get("xgb", {})
    for name, expected in EXPECTED_XGB.items():
        actual = xgb_params.get(name)
        if actual != expected:
            errors.append(f"XGBoost {name}={actual!r}; expected {expected!r}")
    return errors


def compare_submissions(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, float | bool]:
    same_ids = reference["Client_ID"].equals(candidate["Client_ID"])
    if not same_ids:
        return {"same_ids": False}

    expected = reference["Drop_Probability"]
    actual = candidate["Drop_Probability"]
    difference = (actual - expected).abs()
    return {
        "same_ids": True,
        "exact_values": expected.equals(actual),
        "spearman": float(expected.corr(actual, method="spearman")),
        "mean_abs_diff": float(difference.mean()),
        "max_abs_diff": float(difference.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore the shared notebook cache and refit everything (much slower).",
    )
    parser.add_argument(
        "--keep-candidate",
        type=Path,
        help="Copy the isolated generated CSV to this path after validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REFERENCE_CSV.resolve() == NOTEBOOK_OUTPUT_CSV.resolve():
        raise RuntimeError(
            "Unsafe validator configuration: the protected reference and notebook "
            "output paths must be different"
        )

    reference_hash = sha256(REFERENCE_CSV)
    reference_hash_matches = reference_hash == REFERENCE_SHA256
    print(f"reference: {REFERENCE_CSV.relative_to(ROOT)}")
    print(f"reference SHA-256: {reference_hash}")
    if not reference_hash_matches:
        print(f"ERROR: expected reference SHA-256 {REFERENCE_SHA256}")

    reference = load_submission(REFERENCE_CSV)
    print(
        "running notebook in an isolated temporary directory "
        f"({'fresh models' if args.fresh else 'shared source-keyed cache'})...",
        flush=True,
    )
    candidate_path, definitions, _, temporary = run_notebook_isolated(
        fresh=args.fresh
    )
    try:
        candidate_hash = sha256(candidate_path)
        candidate = load_submission(candidate_path)
        config_errors = validate_model_configuration(definitions)
        comparison = compare_submissions(reference, candidate)
        print(
            "selected XGBoost: "
            f"depth={definitions.get('selected_depth')}, "
            f"learning_rate={definitions.get('selected_learning_rate')}, "
            f"n_estimators={definitions.get('selected_n_trees')}, "
            f"regularization={definitions.get('selected_profile_name')}"
        )

        if args.keep_candidate:
            destination = args.keep_candidate.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, destination)
            print(f"kept generated candidate: {destination}")
    finally:
        temporary.cleanup()

    reference_hash_after_run = sha256(REFERENCE_CSV)
    reference_unchanged = reference_hash_after_run == reference_hash
    if not reference_unchanged:
        print(
            "ERROR: protected reference changed while the notebook was running: "
            f"{reference_hash_after_run}"
        )

    print(f"candidate SHA-256: {candidate_hash}")
    for name, value in comparison.items():
        print(f"{name}: {value}")
    for error in config_errors:
        print(f"ERROR: {error}")

    passes_similarity = bool(
        comparison.get("same_ids")
        and comparison.get("spearman", 0.0) >= MIN_SPEARMAN
        and comparison.get("mean_abs_diff", float("inf")) <= MAX_MEAN_ABS_DIFF
        and comparison.get("max_abs_diff", float("inf")) <= MAX_ABS_DIFF
    )
    passed = (
        reference_hash_matches
        and reference_unchanged
        and not config_errors
        and passes_similarity
    )
    print(f"validation: {'PASS' if passed else 'FAIL'}")
    print("note: this validates reproducibility, not hidden-test ROC-AUC")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
