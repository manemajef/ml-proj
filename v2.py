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
# # V2 pipeline — fresh rediscovery of the drop-prediction problem
#
# Goal: predict `Drop_Probability` for the official test set, beating the
# previous leaderboard AUC (~0.886). Key insight driving every choice here:
# the official test window (2017-04-26 .. 2017-08-31) is strictly LATER than
# the train window (2015-07-01 .. 2017-04-26). Random-split validation gave
# 0.944 while the leaderboard gave 0.886, so random CV is optimistic under
# temporal drift. All model selection below therefore uses a CHRONOLOGICAL
# holdout (last ~4 months of train) that mimics the leaderboard setup.
#
# Usage:
# python v2.py experiments # run chrono-holdout experiments, print table
# python v2.py final # fit chosen config on all data, write submission
#
# Output: data/Group_27_Submission.csv (Client_ID, Drop_Probability)
#

# %%
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
SUBMISSION_PATH = "data/Group_27_Submission_02.csv"
TARGET = "Dropped_Course"
CHRONO_CUTOFF = "2017-01-01"  # val window ~4 months, same length as test window
SEED = 42

# %% [markdown]
# ## 1. Cleaning
#
# The categorical columns are deliberately dirty: mixed case, padded
# whitespace, junk placeholder strings. Normalize everything to a canonical
# lowercase form and unify known aliases (e.g. `cn` vs `chn` for China).
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
# - Date parts: month / day-of-week / week-of-year for seasonality. No raw
#   date or linear time index by default — trees extrapolate the last leaf,
#   which is exactly what burned the previous submission (tested below).
# - Group sizes and ratios, previous-history rates, lab-config match,
#   missing-ness flags for the ID columns.
# - Frequency encoding for the high-cardinality IDs, computed on
#   train+test combined (no label involved, so no leakage).
# - Numeric sanity caps for the known corrupted values
#   (Students_Count=9999, negative Practical_Hours, tuition=5400).
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

    # date parts (seasonality only)
    d = df["Course_Start_Date"]
    out["start_month"] = d.dt.month
    out["start_dow"] = d.dt.dayofweek
    out["start_week"] = d.dt.isocalendar().week.astype(float)

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
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True)
        for col in ("Agent_ID", "Company_ID", "Origin_Country")
    }


def align_categories(train_X: pd.DataFrame, *others: pd.DataFrame):
    """Give every frame identical category levels so boosters agree."""
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
# Three gradient boosters with native categorical handling. Moderate depth,
# enough trees, mild regularization — tuned lightly against the chrono
# holdout, not against random CV.
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
    if name == "cat":
        cat_idx = [
            i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) == "category"
        ]
        X_tr2 = X_tr.copy()
        X_va2 = X_va.copy()
        for c in X_tr2.columns[cat_idx]:
            X_tr2[c] = X_tr2[c].astype(str)
            X_va2[c] = X_va2[c].astype(str)
        m = get_cat(cat_features=cat_idx)
        m.fit(X_tr2, y_tr, sample_weight=sample_weight)
        return m.predict_proba(X_va2)[:, 1]
    m = get_lgbm() if name == "lgbm" else get_xgb()
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(X_va)[:, 1]


# %% [markdown]
# ## 4. Chronological experiments
#
# Fit on rows before 2017-01-01, validate on 2017 rows (~11.6k). This mirrors
# "train on the past, score on the future" exactly like the leaderboard.
#

# %%
def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata

    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def run_experiments():
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    freq_maps = make_freq_maps(train_raw, test_raw)

    cutoff = pd.Timestamp(CHRONO_CUTOFF)
    tr_mask = train_raw["Course_Start_Date"] < cutoff
    tr_raw, va_raw = train_raw[tr_mask], train_raw[~tr_mask]
    print(
        f"chrono split: fit={len(tr_raw)}  val={len(va_raw)}  "
        f"val drop rate={va_raw[TARGET].mean():.3f}"
    )

    X_tr = build_features(tr_raw, freq_maps)
    X_va = build_features(va_raw, freq_maps)
    align_categories(X_tr, X_va)
    y_tr, y_va = tr_raw[TARGET].values, va_raw[TARGET].values

    # recency weights: half-life of 365 days
    age_days = (tr_raw["Course_Start_Date"].max() - tr_raw["Course_Start_Date"]).dt.days
    w_recency = np.power(0.5, age_days / 365.0).values

    results = {}
    for name in ("lgbm", "xgb", "cat"):
        p = fit_predict(name, X_tr, y_tr, X_va)
        results[name] = p
        print(f"{name:>12}: chrono AUC = {roc_auc_score(y_va, p):.4f}")

    p = fit_predict("lgbm", X_tr, y_tr, X_va, sample_weight=w_recency)
    results["lgbm_recency"] = p
    print(f"{'lgbm_recency':>12}: chrono AUC = {roc_auc_score(y_va, p):.4f}")

    blend = rank_avg([results["lgbm"], results["xgb"], results["cat"]])
    print(f"{'blend3':>12}: chrono AUC = {roc_auc_score(y_va, blend):.4f}")
    blend_w = rank_avg([results["lgbm_recency"], results["xgb"], results["cat"]])
    print(f"{'blend3_rec':>12}: chrono AUC = {roc_auc_score(y_va, blend_w):.4f}")

    # ablation: does adding a linear time index help or hurt the future window?
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    epoch = pd.Timestamp("2015-01-01")
    X_tr2["days_since_epoch"] = (tr_raw["Course_Start_Date"] - epoch).dt.days.values
    X_va2["days_since_epoch"] = (va_raw["Course_Start_Date"] - epoch).dt.days.values
    time_preds = {}
    for name in ("lgbm", "xgb", "cat"):
        p = fit_predict(name, X_tr2, y_tr, X_va2)
        time_preds[name] = p
        print(f"{name + '+time':>12}: chrono AUC = {roc_auc_score(y_va, p):.4f}")
    blend_t = rank_avg(list(time_preds.values()))
    print(f"{'blend3+time':>12}: chrono AUC = {roc_auc_score(y_va, blend_t):.4f}")

    # recency weighting on top of the time feature, shorter half-life
    age2 = (tr_raw["Course_Start_Date"].max() - tr_raw["Course_Start_Date"]).dt.days
    w180 = np.power(0.5, age2 / 180.0).values
    p = fit_predict("lgbm", X_tr2, y_tr, X_va2, sample_weight=w180)
    print(f"{'lgbm+t+w180':>12}: chrono AUC = {roc_auc_score(y_va, p):.4f}")

    # reference: random-split score, to document the optimism gap
    from sklearn.model_selection import train_test_split

    tr_r, va_r = train_test_split(
        train_raw, test_size=0.2, random_state=SEED, stratify=train_raw[TARGET]
    )
    X_trr = build_features(tr_r, freq_maps)
    X_var = build_features(va_r, freq_maps)
    align_categories(X_trr, X_var)
    p = fit_predict("lgbm", X_trr, tr_r[TARGET].values, X_var)
    print(
        f"{'lgbm random':>12}: AUC = {roc_auc_score(va_r[TARGET].values, p):.4f}"
        f"   <- optimistic, do not trust"
    )


# %% [markdown]
# ## 5. Final fit and submission
#
# Retrain the winning configuration on ALL labeled rows and rank-average the
# three boosters. Rank averaging preserves AUC ordering while washing out
# calibration differences between the models.
#

# %%
def run_final():
    train_raw = load_raw(TRAIN_PATH)
    test_raw = load_raw(TEST_PATH)
    freq_maps = make_freq_maps(train_raw, test_raw)

    X_tr = build_features(train_raw, freq_maps)
    X_te = build_features(test_raw, freq_maps)

    # linear time index: validated on the chrono holdout, where it improved
    # every model (future rows land in the most-recent leaf)
    epoch = pd.Timestamp("2015-01-01")
    X_tr["days_since_epoch"] = (train_raw["Course_Start_Date"] - epoch).dt.days.values
    X_te["days_since_epoch"] = (test_raw["Course_Start_Date"] - epoch).dt.days.values

    align_categories(X_tr, X_te)
    y_tr = train_raw[TARGET].values

    preds = []
    for name in ("lgbm", "xgb", "cat"):
        p = fit_predict(name, X_tr, y_tr, X_te)
        preds.append(p)
        print(f"fitted {name} on {len(X_tr)} rows")

    blend = rank_avg(preds)
    submission = pd.DataFrame({
        "Client_ID": test_raw["Client_ID"],
        "Drop_Probability": blend,
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {SUBMISSION_PATH}  ({len(submission)} rows)")


# %%
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    if mode == "experiments":
        run_experiments()
    elif mode == "final":
        run_final()
