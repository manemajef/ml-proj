# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Pipeline v2 — production pipeline for the drop-probability model
#
# This is the **clean, runnable pipeline** distilled from the v2 exploration in [`notebook_v2.py`](notebook_v2.py). All plots, experiments, and rejected ideas live in that notebook; here we keep only the steps that produce the submission, plus the markdown that explains _why_ each step exists.
#
# **The one idea behind every choice:** the official test window (2017-04-26 → 2017-08-31) is strictly _later_ than the training window (2015-07-01 $\to$ 2017-04-26). Random-split CV therefore overstates real performance (it scored 0.944 while the leaderboard gave 0.886). Model and feature selection were done against a **chronological holdout** — see `notebook_v2.py` for the evidence. This file just applies the winners.
#
# **Result:** the blend below scored **0.889314 AUC** on the official
# leaderboard (1st of 32 groups), up from v1's 0.886408.
#
# Usage:
#
# ```bash
# python pipeline_v2.py # fit on all data, write the submission
# python pipeline_v2.py --out PATH # write to a custom path instead
# ```
#

# %%
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
SUBMISSION_PATH = "data/Group_27_Submission.csv"  # the official v2 submission
TARGET = "Dropped_Course"
SEED = 42

# %% [markdown]
# ## 1. Cleaning
#
# The categorical columns are deliberately dirty: mixed case, padded whitespace, junk placeholder strings ("unknown", "-", "n/a", ...). Normalize everything to a canonical lowercase form, map junk to NaN, and unify known aliases (`cn` and `chn` are both China).
#

# %%
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
            s.str
            .replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() .+-]+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        s = s.mask(s.isin(COMMON_NANS))
        df[col] = s
    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    return df


# %% [markdown]
# ## 2. Feature engineering
#
# - **Seasonality**: month / day-of-week / week-of-year from the start date.
# - **A linear time index** (`days_since_epoch`): counterintuitive but it
#   _improved_ every model on the future-window holdout — future rows land in
#   the most-recent leaf, so the model scores them like the latest regime
#   instead of averaging over 2015-2016. See `notebook_v2.py` for the ablation.
# - **Composition & ratios**: group size, professional share, hours split,
#   cost×hours, previous drop rate, per-participant kits/tickets.
# - **Frequency encoding** for the high-cardinality IDs (Agent/Company/Country),
#   counted on train+test combined — label-free, so no leakage.
# - **Sanity caps** for the known corrupted values (Students_Count=9999,
#   negative Practical_Hours, Daily_Tuition_Cost=5400).
# - Categoricals are kept as native `category` dtype for the boosters rather
#   than one-hot encoded.
#


# %%
def build_features(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    df = normalize_cats(df)
    out = pd.DataFrame(index=df.index)

    # numeric passthrough with sanity caps
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

    # date: seasonality parts + linear time index (validated in notebook_v2)
    d = df["Course_Start_Date"]
    out["start_month"] = d.dt.month
    out["start_dow"] = d.dt.dayofweek
    out["start_week"] = d.dt.isocalendar().week.astype(float)
    out["days_since_epoch"] = (d - pd.Timestamp("2015-01-01")).dt.days

    # group composition
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

    # client history
    out["prev_drop_rate"] = df["Prev_Course_Dropouts"] / (
        df["Prev_Course_Attended"] + 1
    )
    out["kits_per_participant"] = df["Physical_Course_Kits"] / total.replace(0, np.nan)
    out["tickets_per_participant"] = df["Pre_Course_Supports_Tickets"] / total.replace(
        0, np.nan
    )

    # lab config: what matters is the request and whether it was honored
    out["got_requested_lab"] = (
        df["Requested_Lab_Config"] == df["Assigned_Lab_Config"]
    ).astype(float)

    # missingness / presence flags for IDs
    out["has_company_id"] = df["Company_ID"].notna().astype(int)
    out["has_agent_id"] = df["Agent_ID"].notna().astype(int)

    # frequency encodings (train+test combined counts, label-free)
    for col in ("Agent_ID", "Company_ID", "Origin_Country"):
        out[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0).astype(float)

    # native categoricals for the boosters
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
    """Frequency of each ID value across all supplied frames (label-free)."""
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True)
        for col in ("Agent_ID", "Company_ID", "Origin_Country")
    }


def align_categories(train_X: pd.DataFrame, *others: pd.DataFrame):
    """Give every frame identical category levels so the boosters agree."""
    for col in train_X.select_dtypes("category").columns:
        cats = train_X[col].cat.categories
        for o in others:
            cats = cats.union(o[col].cat.categories)
        train_X[col] = train_X[col].cat.set_categories(cats)
        for o in others:
            o[col] = o[col].cat.set_categories(cats)


# %% [markdown]
# ## 3. Models
#
# Three gradient boosters with native categorical handling, blended. Hyper-parameters were tuned lightly against the chronological holdout (not random CV). The blend beats every single model on the future window — see `notebook_v2.py`.
#


# %%
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
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
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
    """Fit one booster by name ('lgbm' | 'xgb' | 'cat') and return P(drop)."""
    if name == "cat":
        cat_idx = [
            i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) == "category"
        ]
        X_tr2, X_va2 = X_tr.copy(), X_va.copy()
        for c in X_tr2.columns[cat_idx]:
            X_tr2[c] = X_tr2[c].astype(str)
            X_va2[c] = X_va2[c].astype(str)
        m = get_cat(cat_features=cat_idx)
        m.fit(X_tr2, y_tr, sample_weight=sample_weight)
        return m.predict_proba(X_va2)[:, 1]
    m = get_lgbm() if name == "lgbm" else get_xgb()
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(X_va)[:, 1]


def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    """Average of per-model rank-percentiles. Preserves AUC ordering while
    ignoring calibration differences between the models."""
    from scipy.stats import rankdata

    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


# %% [markdown]
# ## 4. Fit on all data and write the submission
#
# Retrain all three boosters on every labeled row, rank-average their test predictions, and write the CSV.
#


# %%
def run_final(out_path: str = SUBMISSION_PATH) -> pd.DataFrame:
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    freq_maps = make_freq_maps(train_raw, test_raw)

    X_tr = build_features(train_raw, freq_maps)
    X_te = build_features(test_raw, freq_maps)
    align_categories(X_tr, X_te)
    y_tr = train_raw[TARGET].values

    preds = []
    for name in ("lgbm", "xgb", "cat"):
        preds.append(fit_predict(name, X_tr, y_tr, X_te))
        print(f"fitted {name} on {len(X_tr)} rows")

    submission = pd.DataFrame({
        "Client_ID": test_raw["Client_ID"],
        "Drop_Probability": rank_avg(preds),
    })

    WRITE_CSV = False  # dont ovverride file everytime it runs
    if not WRITE_CSV:
        return submission
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(submission)} rows)")
    return submission


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the v2 blend and write the submission."
    )
    parser.add_argument(
        "--out",
        default=SUBMISSION_PATH,
        help=f"output CSV path (default: {SUBMISSION_PATH})",
    )
    args = parser.parse_args()
    run_final(args.out)
