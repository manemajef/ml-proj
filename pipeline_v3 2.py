"""Downside-protected v3 submission pipeline.

Compared with the verified v2 pipeline, this version removes the redundant
``start_week`` feature and uses a slightly more regularized XGBoost component.
Both changes were selected on three expanding chronological validation windows.
The final score anchors 25% of the verified v2 ranking for downside protection.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
SUBMISSION_PATH = "data/Group_27_Submission_v3.csv"
ANCHOR_PATH = "data/Group_27_Submission.csv"
CANDIDATE_WEIGHT = 0.75
TARGET = "Dropped_Course"
SEED = 42

COMMON_NANS = {
    "",
    "-",
    "--",
    ".",
    "?",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unknonwn",
}
COUNTRY_ALIASES = {"cn": "chn"}

CAT_COLS = [
    "Origin_Country",
    "Catering_Package",
    "Welcome_Gift_Type",
    "Requested_Lab_Config",
    "Assigned_Lab_Config",
    "Enrollment_Type",
    "Lanyard_Color",
    "Client_Category",
    "Submission_Source",
    "Payment_Terms",
    "Agent_ID",
    "Company_ID",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Course_Start_Date"])
    for col in ("Agent_ID", "Company_ID"):
        df[col] = df[col].astype("string")
    return df


def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        s = df[col].astype("string").str.strip().str.lower()
        s = (
            s.str.replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() .+-]+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        s = s.mask(s.isin(COMMON_NANS))
        df[col] = s
    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    return df


def build_features(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    df = normalize_cats(df)
    out = pd.DataFrame(index=df.index)

    # Numeric passthrough with the same sanity caps as the verified pipeline.
    out["Professionals_Count"] = df["Professionals_Count"]
    out["Students_Count"] = df["Students_Count"].clip(upper=10)
    out["Observers_Count"] = df["Observers_Count"]
    out["Practical_Hours"] = df["Practical_Hours"].clip(0, 12)
    out["Theory_Hours"] = df["Theory_Hours"]
    out["Registration_Days_Before"] = df["Registration_Days_Before"]
    out["Prev_Course_Dropouts"] = df["Prev_Course_Dropouts"]
    out["Prev_Course_Attended"] = df["Prev_Course_Attended"]
    out["Pre_Course_Supports_Tickets"] = df["Pre_Course_Supports_Tickets"]
    out["Physical_Course_Kits"] = df["Physical_Course_Kits"]
    out["Waiting_List_Days"] = df["Waiting_List_Days"]
    out["Registration_Changes"] = df["Registration_Changes"]
    out["Returning_Client"] = df["Returning_Client"]
    out["Daily_Tuition_Cost"] = df["Daily_Tuition_Cost"].clip(upper=600)

    # Keep stable seasonality/trend features. ISO week was removed after it
    # underperformed consistently in rolling future-window validation.
    d = df["Course_Start_Date"]
    out["start_month"] = d.dt.month
    out["start_dow"] = d.dt.dayofweek
    out["days_since_epoch"] = (d - pd.Timestamp("2015-01-01")).dt.days

    total = (
        df["Professionals_Count"].fillna(0)
        + df["Students_Count"].clip(upper=10).fillna(0)
        + df["Observers_Count"].fillna(0)
    )
    out["total_participants"] = total
    out["prof_share"] = df["Professionals_Count"] / total.replace(0, np.nan)
    out["total_hours"] = df["Practical_Hours"].clip(0, 12) + df["Theory_Hours"]
    out["practical_share"] = df["Practical_Hours"].clip(0, 12) / out[
        "total_hours"
    ].replace(0, np.nan)
    out["cost_x_days"] = df["Daily_Tuition_Cost"].clip(upper=600) * out["total_hours"]

    out["prev_drop_rate"] = df["Prev_Course_Dropouts"] / (
        df["Prev_Course_Attended"] + 1
    )
    out["kits_per_participant"] = df["Physical_Course_Kits"] / total.replace(
        0, np.nan
    )
    out["tickets_per_participant"] = df[
        "Pre_Course_Supports_Tickets"
    ] / total.replace(0, np.nan)

    out["got_requested_lab"] = (
        df["Requested_Lab_Config"] == df["Assigned_Lab_Config"]
    ).astype(float)
    out["has_company_id"] = df["Company_ID"].notna().astype(int)
    out["has_agent_id"] = df["Agent_ID"].notna().astype(int)

    for col in ("Agent_ID", "Company_ID", "Origin_Country"):
        out[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0).astype(float)

    for col in (
        "Origin_Country",
        "Catering_Package",
        "Welcome_Gift_Type",
        "Requested_Lab_Config",
        "Enrollment_Type",
        "Lanyard_Color",
        "Client_Category",
        "Submission_Source",
        "Payment_Terms",
        "Agent_ID",
    ):
        out[col] = df[col].fillna("missing").astype("category")

    return out


def make_freq_maps(*dfs: pd.DataFrame) -> dict:
    """Frequency of each ID value across supplied frames, without labels."""
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True)
        for col in ("Agent_ID", "Company_ID", "Origin_Country")
    }


def align_categories(train_X: pd.DataFrame, *others: pd.DataFrame):
    for col in train_X.select_dtypes("category").columns:
        cats = train_X[col].cat.categories
        for other in others:
            cats = cats.union(other[col].cat.categories)
        train_X[col] = train_X[col].cat.set_categories(cats)
        for other in others:
            other[col] = other[col].cat.set_categories(cats)


def get_lgbm(**kw):
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    params.update(kw)
    return LGBMClassifier(**params)


def get_xgb(**kw):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )
    params.update(kw)
    return XGBClassifier(**params)


def get_cat(**kw):
    from catboost import CatBoostClassifier

    params = dict(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        verbose=False,
        eval_metric="AUC",
    )
    params.update(kw)
    return CatBoostClassifier(**params)


def fit_predict(name, X_tr, y_tr, X_va, sample_weight=None):
    if name == "cat":
        cat_idx = [
            i for i, col in enumerate(X_tr.columns) if str(X_tr[col].dtype) == "category"
        ]
        X_tr2, X_va2 = X_tr.copy(), X_va.copy()
        for col in X_tr2.columns[cat_idx]:
            X_tr2[col] = X_tr2[col].astype(str)
            X_va2[col] = X_va2[col].astype(str)
        model = get_cat(cat_features=cat_idx)
        model.fit(X_tr2, y_tr, sample_weight=sample_weight)
        return model.predict_proba(X_va2)[:, 1]

    model = get_lgbm() if name == "lgbm" else get_xgb()
    if name == "lgbm":
        cat_cols = X_tr.select_dtypes("category").columns.tolist()
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            categorical_feature=cat_cols,
        )
    else:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    return model.predict_proba(X_va)[:, 1]


def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata

    return np.mean([rankdata(pred) / len(pred) for pred in preds], axis=0)


def anchor_to_verified_submission(
    candidate: pd.DataFrame,
    anchor_path: str = ANCHOR_PATH,
    candidate_weight: float = CANDIDATE_WEIGHT,
) -> pd.DataFrame:
    """Blend v3 ranks with the verified v2 submission for downside protection."""
    from scipy.stats import rankdata

    if not 0 <= candidate_weight <= 1:
        raise ValueError("candidate_weight must be between 0 and 1")
    anchor = pd.read_csv(anchor_path)
    expected = ["Client_ID", "Drop_Probability"]
    if list(anchor.columns) != expected or anchor["Client_ID"].duplicated().any():
        raise ValueError(f"invalid anchor submission: {anchor_path}")
    aligned = candidate[["Client_ID"]].merge(
        anchor, on="Client_ID", how="left", validate="one_to_one"
    )
    if aligned["Drop_Probability"].isna().any():
        raise ValueError("anchor is missing one or more test Client_ID values")

    n = len(candidate)
    old_rank = rankdata(aligned["Drop_Probability"].to_numpy()) / n
    new_rank = rankdata(candidate["Drop_Probability"].to_numpy()) / n
    return pd.DataFrame(
        {
            "Client_ID": candidate["Client_ID"],
            "Drop_Probability": (1 - candidate_weight) * old_rank
            + candidate_weight * new_rank,
        }
    )


def run_final(
    out_path: str = SUBMISSION_PATH,
    write: bool = False,
    anchor_path: str = ANCHOR_PATH,
    candidate_weight: float = CANDIDATE_WEIGHT,
) -> pd.DataFrame:
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    freq_maps = make_freq_maps(train_raw, test_raw)

    X_train = build_features(train_raw, freq_maps)
    X_test = build_features(test_raw, freq_maps)
    align_categories(X_train, X_test)
    y_train = train_raw[TARGET].to_numpy()

    predictions = []
    for name in ("lgbm", "xgb", "cat"):
        predictions.append(fit_predict(name, X_train, y_train, X_test))
        print(f"fitted {name} on {len(X_train)} rows")

    candidate = pd.DataFrame(
        {
            "Client_ID": test_raw["Client_ID"],
            "Drop_Probability": rank_avg(predictions),
        }
    )
    submission = anchor_to_verified_submission(
        candidate, anchor_path=anchor_path, candidate_weight=candidate_weight
    )
    if write:
        submission.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({len(submission)} rows)")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the validated v3 blend and optionally write its submission."
    )
    parser.add_argument(
        "--out",
        default=SUBMISSION_PATH,
        help=f"output CSV path (default: {SUBMISSION_PATH})",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the submission CSV; the default is a non-writing dry run",
    )
    parser.add_argument(
        "--anchor",
        default=ANCHOR_PATH,
        help=f"verified v2 submission used as a rank anchor (default: {ANCHOR_PATH})",
    )
    parser.add_argument(
        "--candidate-weight",
        type=float,
        default=CANDIDATE_WEIGHT,
        help="v3 rank weight in the final anchored blend; use 1.0 for pure v3",
    )
    args = parser.parse_args()
    run_final(
        args.out,
        write=args.write,
        anchor_path=args.anchor,
        candidate_weight=args.candidate_weight,
    )
    if not args.write:
        print("dry run complete; use --write to create the v3 submission file")
