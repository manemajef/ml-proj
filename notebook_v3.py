# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Group 27 — Course-Drop Prediction (Nova Academy)
#
# **Submitters:** Rotem David Semah (ID: `211396593`) · Ron Drach (ID: `213915499`)
#
# ---
#
# This is the **final, unified CRISP-DM notebook** for the project. It tells the whole story end to end: business understanding → data exploration → cleaning → feature engineering → modelling → evaluation → interpretation → conclusions.
#
# It is a polished merge of two lines of exploratory work (`notebook_v1.py` and `Project_Ron_V3.ipynb`) around the winning model, whose clean runnable form lives in `pipeline_v2.py`. Running this notebook top to bottom reproduces our official submission.
#
# **The one idea that shaped every decision:** the hidden test set is the**future** — it begins exactly where training ends and runs four months further. So the task is _forecasting_, not interpolating, and any validation that mixes past and future rows (a random split) is measuring the wrong thing. Fixing the validation to respect time, then engineering and blending around it, lifted our leaderboard AUC from **0.886 (v1) to 0.889314 — 1st of 32 groups**.
#

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# `display` is injected by Jupyter. Define a fallback so the notebook also runs
# as a plain script (e.g. for CI / reproducibility checks).
try:
    display  # noqa: B018
except NameError:  # pragma: no cover

    def display(*objs, **_):
        for o in objs:
            print(o)


TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
TARGET = "Dropped_Course"
CHRONO_CUTOFF = "2017-01-01"  # validation window ~ 4 months, matching the test window
SEED = 42
GITHUB_RAW_BASES = [
    "https://raw.githubusercontent.com/manemajef/ml-proj/main",
]

pd.set_option("display.max_columns", None)


def load_raw(path: str) -> pd.DataFrame:
    local_path = Path(path)
    read_errors = []

    if local_path.exists():
        source = local_path
    else:
        source = None
        for base_url in GITHUB_RAW_BASES:
            url = f"{base_url}/{path}"
            try:
                df = pd.read_csv(url, parse_dates=["Course_Start_Date"])
                print(f"{path} not found locally; loaded from {url}")
                break
            except Exception as exc:
                read_errors.append(f"{url}: {exc}")
        else:
            tried = "\n".join(read_errors) or "No GitHub fallback URL configured."
            raise FileNotFoundError(
                f"Could not find {path} locally and could not fetch it from GitHub.\n{tried}"
            )

    if source is not None:
        df = pd.read_csv(source, parse_dates=["Course_Start_Date"])

    # Agent/Company IDs are categorical labels, not quantities.
    for col in ("Agent_ID", "Company_ID"):
        df[col] = df[col].astype("string")
    return df


# %% [markdown]
# # 1. Business understanding
#
# Nova Academy runs paid, in-person B2B technical trainings. Preparing a course is expensive and largely **sunk before it starts** — cloud environments, catering, physical equipment kits, room capacity. When a registered group cancels (`Dropped_Course = 1`), the company loses that spend _and_ the empty seats block other groups from being scheduled.
#
# **Goal.** Given a new registration, predict the **probability** that it will be cancelled, so operations can manage risk (overbook cautiously, follow up with high-risk groups, hold back irreversible spend).
#
# **Why probability, not a hard label.** Operations needs to _rank_ and _size_ risk, not receive a yes/no. The submission is therefore a calibrated-ish score, and the grading metric is **AUC** — a threshold-free measure of how well the score ranks droppers above non-droppers. The passing bar is AUC ≥ 0.70.
#
# **CRISP-DM framing.** The rest of the notebook follows the standard cycle: understand the data, prepare it, model, evaluate, and translate results back into business insight.
#

# %% [markdown]
# # 2. Data loading & first look
#
# Two files are provided:
#
# - `Train_Data.csv` — historical registrations **with** the `Dropped_Course`
#   label.
# - `Test_Data_No_Target.csv` — registrations to score, **without** the label.
#
# If the local `data/` files are missing, `load_raw` falls back to the raw CSVs
# in the GitHub repository. This keeps the notebook runnable when shared as a
# single `.ipynb`, as long as the machine has internet access and the repository
# data files are reachable.
#
# Each row is one order (`Client_ID`). We load both and immediately build a data dictionary: dtype, missingness, cardinality, mode, and a zero-count (some "zeros" are really missing-in-disguise).
#

# %%
train_raw = load_raw(TRAIN_PATH)
test_raw = load_raw(TEST_PATH)

print(f"train: {train_raw.shape[0]:,} rows x {train_raw.shape[1]} cols")
print(f"test : {test_raw.shape[0]:,} rows x {test_raw.shape[1]} cols")

data_dictionary = pd.DataFrame({
    "dtype": train_raw.dtypes.astype(str),
    "n_missing": train_raw.isna().sum(),
    "missing_%": (train_raw.isna().mean() * 100).round(2),
    "n_unique": train_raw.nunique(dropna=True),
    "n_zero": (train_raw == 0).sum(numeric_only=False),
    "most_frequent": train_raw.mode(dropna=True).iloc[0],
})
display(data_dictionary)

# %% [markdown]
# **What the dictionary tells us.**
#
# - `Client_ID` is unique per row — an identifier, never a feature.
# - `Agent_ID`, `Company_ID`, `Origin_Country` are **high-cardinality**
#   identifiers (many distinct values). `Company_ID` is missing for most rows.
# - A handful of numeric columns show impossible extremes in `describe()` below
#   (e.g. `Students_Count = 9999`, `Practical_Hours = 10000`) — flagged for the
#   outlier section.
# - The categorical text columns are visibly _dirty_ (casing, punctuation,
#   placeholder strings) — handled in cleaning.
#

# %%
display(train_raw.describe())

# %% [markdown]
# ## 2.1 Target balance
#
# Before anything else: how (im)balanced is the target? A heavily skewed target would change how we read metrics.
#

# %%
target_counts = train_raw[TARGET].value_counts().sort_index()
target_rate = train_raw[TARGET].value_counts(normalize=True).sort_index()
balance = pd.DataFrame({
    "count": target_counts,
    "rate_%": (target_rate * 100).round(1),
})
balance.index = ["0 = completed", "1 = dropped"]
display(balance)

ax = target_rate.mul(100).plot.bar(color=["#4c72b0", "#c44e52"], figsize=(5, 3.5))
ax.set_xticklabels(["completed (0)", "dropped (1)"], rotation=0)
ax.set_ylabel("share of orders (%)")
ax.set_title("Target balance — Dropped_Course")
plt.tight_layout()
plt.show()

# %% [markdown]
# The classes are **roughly balanced** (~59% completed / ~41% dropped). No resampling is needed, and AUC is a sensible, stable choice of metric.
#

# %% [markdown]
# # 3. Exploratory Data Analysis
#
# The EDA has one headline finding that reorganises the whole project (the time
# structure), plus the usual per-feature analysis. We lead with the headline.
#

# %% [markdown]
# ## 3.1 The headline: **the test set is the future**
#
# We plot the monthly drop rate across the _training_ period and overlay where training ends and where the hidden test window ends.
#

# %%
train_end = train_raw["Course_Start_Date"].max()
test_start = test_raw["Course_Start_Date"].min()
test_end = test_raw["Course_Start_Date"].max()

print(
    f"train dates: {train_raw['Course_Start_Date'].min().date()} -> {train_end.date()}"
)
print(f"test  dates: {test_start.date()} -> {test_end.date()}")

monthly = (
    train_raw.set_index("Course_Start_Date").resample("MS")[TARGET].mean().mul(100)
)
ax = monthly.plot(marker="o", figsize=(12, 4))
ax.axhline(train_raw[TARGET].mean() * 100, ls="--", color="grey", label="train average")
ax.axvline(train_end, ls="--", color="green", label=f"train ends ({train_end.date()})")
ax.axvline(test_end, ls=":", color="red", label=f"test ends ({test_end.date()})")
ax.set_xlim(train_raw["Course_Start_Date"].min(), test_end)
ax.set_ylabel("drop rate (%)")
ax.set_title("Drop rate over time — training period and the hidden test horizon")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the plot.** Training runs `2015-07 → 2017-04`; the test window starts exactly where training ends and continues to `2017-08`, with essentially **zero overlap in time**. The drop rate also **drifts year to year** — it is not a stationary process.
#
# **Consequence.** The real task is "train on the past, predict the future". A random train/validation split leaks future rows into training and produces an _optimistic_ score that does not transfer to the leaderboard. This single fact dictates our validation strategy (Section 6) and even a feature choice (the time index, Section 5). It is the main reason v2 beat v1.
#

# %% [markdown]
# ## 3.2 Missing values
#
# We compare missingness in train vs the official test file (the pipeline must handle both identically), then ask whether _the fact of being missing_ is itself predictive.
#

# %%
missing_compare = pd.DataFrame({
    "train_missing_%": train_raw.isna().mean().mul(100).round(2),
    "test_missing_%": test_raw.isna().mean().mul(100).round(2),
})
missing_compare = missing_compare[
    (missing_compare["train_missing_%"] > 0) | (missing_compare["test_missing_%"] > 0)
].sort_values("train_missing_%", ascending=False)
display(missing_compare)

# %% [markdown]
# Missingness patterns are **consistent between train and test**, so a single imputation policy is safe to reuse for scoring. Next: is missingness itself a signal?
#

# %%
missingness_cols = [
    "Company_ID",
    "Agent_ID",
    "Registration_Days_Before",
    "Physical_Course_Kits",
    "Daily_Tuition_Cost",
    "Payment_Terms",
]
rows = []
for col in missingness_cols:
    stats = (
        train_raw
        .assign(is_missing=train_raw[col].isna())
        .groupby("is_missing")[TARGET]
        .agg(count="size", drop_rate="mean")
    )
    for is_missing, r in stats.iterrows():
        rows.append({
            "column": col,
            "is_missing": is_missing,
            "count": int(r["count"]),
            "drop_rate_%": round(r["drop_rate"] * 100, 1),
        })
display(pd.DataFrame(rows))

# %% [markdown]
# **Missingness is informative.** Rows _without_ a `Company_ID` drop at a noticeably higher rate than rows with one — a registration made through a known company is a more committed order. `Agent_ID` missingness shows a different profile too. This justifies explicit **presence flags** (`has_company_id`, `has_agent_id`) rather than silently imputing these away.
#

# %% [markdown]
# ## 3.3 Categorical data quality (and why cleaning is mandatory)
#
# The categorical columns are deliberately corrupted: mixed case, injected punctuation (`blu#e` → `blue`), padded whitespace, and placeholder junk (`unknown`, `?`, `-`, `n/a`). Left raw, the same real category splits into many fake ones, inflating cardinality and diluting signal. We normalise to a canonical lowercase form and map junk to missing.
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
COUNTRY_ALIASES = {"cn": "chn"}  # both codes mean China (found in EDA)
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


def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise dirty categorical text and map junk placeholders to NaN."""
    df = df.copy()
    for col in CAT_COLS:
        s = df[col].astype("string").str.strip().str.lower()
        s = (
            s.str
            .replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() .+-]+", "", regex=True)  # strip # ! * ? etc.
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df[col] = s.mask(s.isin(COMMON_NANS))
    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    return df


def cat_cardinality(df, cols):
    return (
        pd
        .DataFrame({
            "raw_unique": {c: df[c].nunique(dropna=True) for c in cols},
            "clean_unique": {
                c: normalize_cats(df)[c].nunique(dropna=True) for c in cols
            },
        })
        .assign(collapsed=lambda t: t["raw_unique"] - t["clean_unique"])
        .sort_values("collapsed", ascending=False)
    )


low_card = [c for c in CAT_COLS if c not in ("Agent_ID", "Company_ID")]
display(cat_cardinality(train_raw, low_card))

# %% [markdown]
# **Before/after normalisation**, dozens of spurious variants collapse into their true categories (the `collapsed` column). This is data cleaning, not feature engineering, and the _exact same_ function is reused for the test set so the category vocabularies line up.
#

# %% [markdown]
# ## 3.4 Which categories actually relate to dropping?
#
# For the cleaned categoricals we plot the drop rate of the most frequent levels against the dataset mean. A level far from the dashed mean line carries signal.
#

# %%
clean_train = normalize_cats(train_raw)


def plot_dropout_by_category(df, col, min_count=50, top_n=10, ax=None):
    stats = df.groupby(col, dropna=False)[TARGET].agg(drop_rate="mean", count="size")
    stats = (
        stats[stats["count"] >= min_count]
        .sort_values("count", ascending=False)
        .head(top_n)
        .sort_values("drop_rate")
    )
    labels = [f"{i} (n={int(r['count'])})" for i, r in stats.iterrows()]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, stats["drop_rate"] * 100, color="#4c72b0")
    overall = df[TARGET].mean() * 100
    ax.axvline(overall, ls="--", color="red", label=f"mean ({overall:.1f}%)")
    ax.set_xlabel("drop rate (%)")
    ax.set_title(f"Drop rate by {col}")
    ax.legend()
    return stats


fig, axes = plt.subplots(2, 2, figsize=(14, 9))
plot_dropout_by_category(
    clean_train, "Payment_Terms", min_count=20, top_n=5, ax=axes[0, 0]
)
plot_dropout_by_category(
    clean_train, "Client_Category", min_count=100, top_n=8, ax=axes[0, 1]
)
plot_dropout_by_category(
    clean_train, "Submission_Source", min_count=100, top_n=6, ax=axes[1, 0]
)
plot_dropout_by_category(
    clean_train, "Enrollment_Type", min_count=100, top_n=6, ax=axes[1, 1]
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Findings.**
#
# - **`Payment_Terms` is the single strongest categorical signal.** _Prepaid
#   (non-refundable)_ orders drop far more often than _pay-on-start_ ones. This is
#   counter-intuitive (why cancel something you can't refund?) and strong enough
#   that we flag it for a leakage check during interpretation (Section 9). We keep
#   it, but watch it.
# - **`Client_Category`**: big-tech / multinational segments drop above average;
#   fintech/banking and industrial/IoT below.
# - **`Submission_Source`**: direct-website and dedicated-sales orders are lower
#   risk than B2B-platform / reseller traffic.
# - **`Enrollment_Type`**: organisational / affiliated arrangements are lower risk
#   than general or one-off contractual admissions.
#
# By contrast, `Lanyard_Color` and `Welcome_Gift_Type` show no stable pattern and have no business reason to matter — candidates to drop as noise.
#

# %% [markdown]
# ### The identifier columns carry signal too
#
# `Agent_ID` and `Company_ID` are labels, not numbers. Frequent agents have very different drop rates, and _having_ a company id lowers risk — so identity here is predictive and we must keep it without exploding the feature space (Section 5.2).
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_dropout_by_category(clean_train, "Agent_ID", min_count=150, top_n=12, ax=axes[0])
company_presence = train_raw.groupby(train_raw["Company_ID"].notna())[TARGET].agg(
    count="size", drop_rate="mean"
)
company_presence.index = ["no company_id", "has company_id"]
axes[1].bar(
    company_presence.index,
    company_presence["drop_rate"] * 100,
    color=["#c44e52", "#55a868"],
)
axes[1].set_ylabel("drop rate (%)")
axes[1].set_title("Drop rate by Company_ID presence")
plt.tight_layout()
plt.show()
display(company_presence)

# %% [markdown]
# ## 3.5 Numeric features: summary, correlation, and suspects
#
# For the numeric columns we tabulate central tendency, spread, skew, and correlation with the target, then look at the correlation structure between features.
#

# %%
ID_LIKE = ["Client_ID", "Agent_ID", "Company_ID"]
num_cols = [
    c
    for c in train_raw.select_dtypes(include=["int64", "float64"]).columns
    if c not in ID_LIKE + [TARGET]
]


def numeric_summary(df, cols):
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "column": c,
            "missing_%": round(s.isna().mean() * 100, 1),
            "corr_target": round(s.corr(df[TARGET]), 3),
            "mean": round(s.mean(), 2),
            "median": round(s.median(), 2),
            "std": round(s.std(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
            "skew": round(s.skew(), 2),
        })
    return pd.DataFrame(rows).sort_values("corr_target", key=abs, ascending=False)


display(numeric_summary(train_raw, num_cols))

# %% [markdown]
# The `max` column already exposes the corrupted values: `Students_Count` maxes at 9999 and `Practical_Hours` at 10000, with a negative minimum. We handle these in Section 4. First, the correlation picture.
#

# %%
corr = train_raw[num_cols + [TARGET]].corr()
plt.figure(figsize=(13, 10))
sns.heatmap(
    corr, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm", center=0
)
plt.title("Numeric correlation heatmap (incl. target)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Reading the heatmap.** No single raw numeric feature correlates strongly with the target — the signal is spread out and largely **non-linear / interaction driven**, which is exactly where gradient-boosted trees shine and where a plain linear model struggles. Inter-feature correlations are mild, so there is no severe multicollinearity forcing us to drop columns; the dimensionality problem lives in the _categoricals_, not here (Section 5.2).
#

# %% [markdown]
# ## 3.6 Numeric drop-rate profiles
#
# Binning a couple of the more predictive numeric features shows _how_ risk moves with them (not just whether they correlate linearly).
#

# %%
def plot_dropout_by_bins(df, col, bins=8, ax=None):
    tmp = df[[col, TARGET]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[col], q=bins, duplicates="drop")
    stats = tmp.groupby("bin", observed=True)[TARGET].mean().mul(100)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    stats.plot.bar(ax=ax, color="#4c72b0")
    ax.axhline(df[TARGET].mean() * 100, ls="--", color="red", label="mean")
    ax.set_ylabel("drop rate (%)")
    ax.set_title(f"Drop rate by {col} bins")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    return stats


fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
plot_dropout_by_bins(train_raw, "Registration_Days_Before", bins=8, ax=axes[0])
plot_dropout_by_bins(train_raw, "Pre_Course_Supports_Tickets", bins=6, ax=axes[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# - **`Registration_Days_Before`**: the earlier a group registers relative to the
#   course, the more likely it is to drop — plausibly because plans change over a
#   longer horizon.
# - **`Pre_Course_Supports_Tickets`**: more pre-course engagement is associated
#   with _lower_ dropping — a group that is actively preparing is committed.
#
# **EDA takeaways carried forward:** time structure (headline), payment terms, registration timing, company/agent identity, client segment & channel, and support engagement are the strongest visible signals.
#

# %% [markdown]
# # 4. Missing-value handling & outlier analysis
#
# Guided by the EDA, we now fix the corrupted values and decide the imputation policy. Both are applied _inside_ the feature builder (Section 5) so train and test are transformed identically.
#

# %% [markdown]
# ## 4.1 Outliers: identify, justify, cap
#
# We look for values that are physically impossible or absurdly far from the bulk.
#

# %%
def suspect_report(df, cols, max_mult=10):
    out = []
    for c in cols:
        s = df[c].dropna()
        q99 = s.quantile(0.99)
        iqr = s.quantile(0.75) - s.quantile(0.25)
        scale = max(q99, iqr, 1.0)
        why = []
        if s.min() < 0:
            why.append("negative values")
        if s.max() > max_mult * scale:
            why.append(f"max={s.max():g} >> q99={q99:g}")
        if why:
            out.append({
                "column": c,
                "min": s.min(),
                "max": s.max(),
                "q99": round(q99, 1),
                "why": "; ".join(why),
            })
    return pd.DataFrame(out)


print("Suspect columns — TRAIN")
display(suspect_report(train_raw, num_cols))
print("Suspect columns — TEST")
display(suspect_report(test_raw, num_cols))

# %% [markdown]
# The test set introduces **no new kinds** of corruption, so caps learned from domain reasoning on train transfer safely. We also check a logical constraint:
#

# %%
impossible = train_raw[
    train_raw["Prev_Course_Dropouts"] > train_raw["Prev_Course_Attended"]
]
print(f"rows where prev dropouts > prev attended (impossible): {len(impossible)}")

# %% [markdown]
# **Decisions and justification.**
#
# | Column               | Problem             | Action            | Why                                         |
# | -------------------- | ------------------- | ----------------- | ------------------------------------------- |
# | `Students_Count`     | jumps to `9999`     | clip to ≤ 10      | bulk is single-digit; 9999 is a placeholder |
# | `Practical_Hours`    | negatives & `10000` | clip to `[0, 12]` | course hours can't be negative or 5-digit   |
# | `Daily_Tuition_Cost` | one `5400` value    | clip to ≤ 600     | ~30× the typical rate; a data-entry error   |
#
# We **clip (winsorize) rather than drop rows**: the _other_ fields in a corrupted row are still valid and informative, and clipping keeps train and test aligned. The plot below shows a representative before/after (log scale).
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(train_raw["Students_Count"].dropna(), bins=50, color="#c44e52")
axes[0].set_yscale("log")
axes[0].set_title(f"Students_Count BEFORE (max={train_raw['Students_Count'].max():g})")
axes[1].hist(
    train_raw["Students_Count"].clip(upper=10).dropna(), bins=20, color="#55a868"
)
axes[1].set_yscale("log")
axes[1].set_title("Students_Count AFTER (clipped ≤ 10)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.2 Missing-value policy
#
# Different column types get different treatment, each justified:
#
# - **Categoricals** → keep an explicit `"missing"` level. For tree models,
#   "missing" is just another category the model can split on; the EDA showed
#   missingness is itself predictive, so we must not erase it.
# - **High-cardinality IDs** (`Agent_ID`, `Company_ID`) → represented via
#   **presence flags** and **frequency encoding** (Section 5.2), not imputed.
# - **Numerics** → the gradient-boosting libraries we use (LightGBM, XGBoost,
#   CatBoost) handle `NaN` natively by learning a default split direction, which
#   is strictly more informative than median-filling. We therefore **pass numeric
#   NaNs through** to the models rather than imputing them, and only compute
#   fill-values inside engineered _ratios_ to avoid divide-by-zero.
#
# This is a deliberate change from v1, which median-imputed everything for a one-hot + linear/forest pipeline. With native-NaN boosters, imputation throws away the "was it missing?" signal for no benefit.
#

# %% [markdown]
# # 5. Feature engineering & dimensionality
#
# Every feature below is justified by the EDA or by domain logic. This is the exact transform that `pipeline_v2.py` applies to produce the submission.
#

# %% [markdown]
# ## 5.1 The engineered features and their rationale
#
# **Seasonality & a linear time index.** From `Course_Start_Date` we take `month`, `day-of-week`, `week-of-year` (seasonality), **and** a linear index
#
# $$
# \text{days\_since\_epoch} = (\text{start date}) - \text{2015-01-01}.
# $$
#
# Dropping the date is the intuitive move (trees can't extrapolate past their training range) and is what v1 did. But on a _future_ holdout the time index **helped every model** (proven in Section 6.2): future rows fall past every split threshold and land in the most-recent leaf, so they are scored like the latest regime instead of an average over 2015-2016.
#
# **Composition & ratios** — turn raw counts into rates the model can compare
# across group sizes:
#
# $$
# \text{prof\_share}=\frac{\text{Professionals}}{\text{total participants}},\quad
# \text{practical\_share}=\frac{\text{Practical hours}}{\text{total hours}},
# $$
#
# $$
# \text{prev\_drop\_rate}=\frac{\text{Prev dropouts}}{\text{Prev attended}+1},\quad
# \text{kits/tickets per participant}=\frac{\cdot}{\text{total participants}}.
# $$
#
# The `+1` in `prev_drop_rate` is Laplace smoothing — it keeps the ratio defined for groups with no history and shrinks noisy estimates from tiny denominators toward 0.
#
# **Interaction**: `cost_x_days = Daily_Tuition_Cost × total_hours` approximates the total contract value at stake.
#
# **Lab config**: the raw requested/assigned pair matters only through _was the request honoured?_ → a single boolean `got_requested_lab`.
#
# **Presence flags**: `has_company_id`, `has_agent_id` capture the predictive missingness found in EDA.
#

# %% [markdown]
# ## 5.2 Dimensionality: taming high-cardinality IDs without one-hot
#
# The curse of dimensionality here comes from the **identifiers**, not the numerics. Naive one-hot encoding creates one sparse column per category value.
#

# %%
def make_freq_maps(*dfs):
    """Label-free frequency of each ID value across all frames (no leakage)."""
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True)
        for col in ("Agent_ID", "Company_ID", "Origin_Country")
    }


freq_maps = make_freq_maps(train_raw, test_raw)


def build_features(
    df: pd.DataFrame, freq_maps: dict, add_time: bool = True
) -> pd.DataFrame:
    """Cleaning + feature engineering. Identical transform for train and test.

    Mirrors ``pipeline_v2.build_features`` exactly (source of truth)."""
    df = normalize_cats(df)
    out = pd.DataFrame(index=df.index)

    # numeric passthrough with sanity caps (Section 4)
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

    # date: seasonality + linear time index (validated in Section 6.2)
    d = df["Course_Start_Date"]
    out["start_month"] = d.dt.month
    out["start_dow"] = d.dt.dayofweek
    out["start_week"] = d.dt.isocalendar().week.astype(float)
    if add_time:
        out["days_since_epoch"] = (d - pd.Timestamp("2015-01-01")).dt.days

    # group composition & ratios
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
    out["kits_per_participant"] = df["Physical_Course_Kits"] / total.replace(0, np.nan)
    out["tickets_per_participant"] = df["Pre_Course_Supports_Tickets"] / total.replace(
        0, np.nan
    )

    # lab config: only "was the request honoured?" matters
    out["got_requested_lab"] = (
        df["Requested_Lab_Config"] == df["Assigned_Lab_Config"]
    ).astype(float)

    # predictive missingness of the IDs
    out["has_company_id"] = df["Company_ID"].notna().astype(int)
    out["has_agent_id"] = df["Agent_ID"].notna().astype(int)

    # frequency encodings for high-cardinality IDs (compact, label-free)
    for col in ("Agent_ID", "Company_ID", "Origin_Country"):
        out[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0).astype(float)

    # native categoricals for the boosters (no one-hot expansion).
    # Note: Company_ID (highest cardinality) is intentionally NOT kept raw —
    # only its frequency + presence flag survive.
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


def align_categories(train_X, *others):
    """Give every frame identical category levels so the boosters agree."""
    for col in train_X.select_dtypes("category").columns:
        cats = train_X[col].cat.categories
        for o in others:
            cats = cats.union(o[col].cat.categories)
        train_X[col] = train_X[col].cat.set_categories(cats)
        for o in others:
            o[col] = o[col].cat.set_categories(cats)


# Dimensionality comparison: native categorical vs a naive one-hot expansion.
X_all = build_features(train_raw, freq_maps)
cat_cols = X_all.select_dtypes("category").columns
native_dim = X_all.shape[1]
onehot_dim = X_all.drop(columns=cat_cols).shape[1] + sum(
    X_all[c].nunique(dropna=False) for c in cat_cols
)
print(f"features with native categorical handling : {native_dim}")
print(f"estimated dims after naive one-hot        : {onehot_dim}")
print(f"dummy columns avoided                     : {onehot_dim - native_dim}")
print("\ncategory cardinalities:")
display(X_all[cat_cols].nunique(dropna=False).sort_values(ascending=False))

# %% [markdown]
# **Our dimensionality strategy.** Rather than one-hot (which would add hundreds of sparse columns) or a hard top-$k$ collapse (v1's approach, which loses the identity of rare agents/countries), v2:
#
# 1. keeps categoricals in **native `category` dtype** — the boosters split on
#    category identity directly, no dummy columns;
# 2. adds a compact **frequency encoding** per ID (how common is this value);
# 3. **drops raw `Company_ID`** (the highest-cardinality field) entirely, keeping
#    only its frequency and presence flag.
#
# This keeps the informative signal while avoiding the sparse-matrix blow-up — a principled answer to the curse of dimensionality.
#

# %% [markdown]
# # 6. Validation methodology
#
# Everything in modelling hinges on measuring performance the _right_ way.
#

# %% [markdown]
# ## 6.1 Adversarial validation — quantifying the drift
#
# We train a classifier to tell **test rows from train rows** using the features (label removed, raw date and `Client_ID` dropped). If it separates them well above AUC 0.5, the feature distributions have genuinely drifted.
#

# %%
def adversarial_validation():
    tr = load_raw(TRAIN_PATH).drop(columns=[TARGET])
    te = load_raw(TEST_PATH)
    combined = pd.concat(
        [tr.assign(is_test=0), te.assign(is_test=1)], ignore_index=True
    )
    y = combined.pop("is_test")
    X = combined.drop(columns=["Client_ID", "Course_Start_Date"], errors="ignore")
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("string").str.strip().str.lower().fillna("missing")
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(-1)

    from xgboost import XGBClassifier

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    m = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    m.fit(Xtr, ytr)
    a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    print(
        f"adversarial AUC (train vs test): {a:.3f}  (0.5=identical, 1.0=trivially separable)"
    )
    top = (
        pd
        .Series(m.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(8)
    )
    print("\ntop drift drivers:")
    print(top)


adversarial_validation()

# %% [markdown]
# The classifier separates test from train **comfortably above chance**, driven by the ID / frequency-style columns — the client population shifts over time. Confirmed: **do not trust a random split.**
#
# ## 6.2 The chronological holdout
#
# We select every model and feature on a **chronological holdout**: fit on rows before `2017-01-01`, validate on the 2017 rows (~4 months, matching the real test window). This reproduces the leaderboard's "train on the past, score the future" setup, so improvements here move in the same direction as the real score.
#

# %%
cutoff = pd.Timestamp(CHRONO_CUTOFF)
tr_raw = train_raw[train_raw["Course_Start_Date"] < cutoff]
va_raw = train_raw[train_raw["Course_Start_Date"] >= cutoff]
y_tr = tr_raw[TARGET].values
y_va = va_raw[TARGET].values
print(
    f"chrono split -> fit={len(tr_raw):,}  validate={len(va_raw):,}  "
    f"(val drop rate={va_raw[TARGET].mean():.3f})"
)

# feature matrices reused across all experiments below
Xtr_t = build_features(tr_raw, freq_maps, add_time=True)
Xva_t = build_features(va_raw, freq_maps, add_time=True)
align_categories(Xtr_t, Xva_t)
Xtr_n = build_features(tr_raw, freq_maps, add_time=False)
Xva_n = build_features(va_raw, freq_maps, add_time=False)
align_categories(Xtr_n, Xva_n)

# %% [markdown]
# # 7. Model experiments & tuning
#
# We follow the assignment's requirement of **at least three models from different families**, each with a short description and its hyper-parameters, and tune on the chronological holdout.
#

# %% [markdown]
# ## 7.1 The model families
#
# - **Logistic Regression** — a linear baseline. Fast, interpretable, but can only
#   fit linear decision boundaries (in the encoded space); expected to lag on this
#   interaction-heavy data. Key hyper-parameter: inverse-regularisation `C`.
# - **Random Forest** — bagged decision trees; captures non-linearities, robust,
#   but averages many deep trees. Key hyper-parameters: `n_estimators`, `max_depth`.
# - **Gradient-boosted trees** — **LightGBM, XGBoost, CatBoost**. Trees are built
#   _sequentially_, each correcting the last. State of the art for tabular data,
#   with **native categorical / missing handling**. Key hyper-parameters:
#   `n_estimators`/`iterations`, `learning_rate`, tree size (`num_leaves` /
#   `max_depth`), and regularisation (`reg_lambda`, `min_child_*`).
#
# XGBoost is included per the assignment's recommendation; LightGBM and CatBoost are added (the brief encourages tools beyond the lectures) because their differing inductive biases make a **blend** more robust than any single model.
#

# %%
def get_lgbm(**kw):
    from lightgbm import LGBMClassifier

    p = dict(
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
    p.update(kw)
    return LGBMClassifier(**p)


def get_xgb(**kw):
    from xgboost import XGBClassifier

    p = dict(
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
    p.update(kw)
    return XGBClassifier(**p)


def get_cat(**kw):
    from catboost import CatBoostClassifier

    p = dict(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        verbose=False,
        eval_metric="AUC",
    )
    p.update(kw)
    return CatBoostClassifier(**p)


def fit_predict(name, X_tr, y_tr, X_va, sample_weight=None):
    """Fit one booster ('lgbm'|'xgb'|'cat') and return P(drop) on X_va."""
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
    if name == "lgbm":
        m = get_lgbm()
        m.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            categorical_feature=X_tr.select_dtypes("category").columns.tolist(),
        )
        return m.predict_proba(X_va)[:, 1]
    m = get_xgb()
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(X_va)[:, 1]


def rank_avg(preds):
    """Average of per-model rank-percentiles: preserves AUC ordering while
    ignoring calibration differences between models."""
    from scipy.stats import rankdata

    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def numeric_encode(X_tr, X_va):
    """Label-encode categoricals + impute for the non-native models (LR/RF)."""
    Xt, Xv = X_tr.copy(), X_va.copy()
    for c in Xt.select_dtypes("category").columns:
        codes = dict(zip(Xt[c].cat.categories, range(len(Xt[c].cat.categories))))
        Xt[c] = Xt[c].map(codes).astype(float)
        Xv[c] = Xv[c].map(codes).astype(float)
    return Xt.fillna(-1), Xv.fillna(-1)


# %% [markdown]
# ## 7.2 Family comparison on the chronological holdout
#
# All models are trained on the _same_ (with-time) feature set and scored on the 2017 window. The linear/forest models get a label-encoded + imputed copy; the boosters use native categoricals.
#

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

Xtr_enc, Xva_enc = numeric_encode(Xtr_t, Xva_t)

# Logistic Regression (scaled)
scaler = StandardScaler()
lr = LogisticRegression(C=1.0, max_iter=2000)
lr.fit(scaler.fit_transform(Xtr_enc), y_tr)
pred_lr = lr.predict_proba(scaler.transform(Xva_enc))[:, 1]

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
)
rf.fit(Xtr_enc, y_tr)
pred_rf = rf.predict_proba(Xva_enc)[:, 1]

# Boosters (native categoricals) — reused later for evaluation & the blend
pred_t = {
    name: fit_predict(name, Xtr_t, y_tr, Xva_t) for name in ("lgbm", "xgb", "cat")
}
blend_t = rank_avg(list(pred_t.values()))

family_scores = (
    pd
    .DataFrame({
        "model": [
            "Logistic Regression",
            "Random Forest",
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Blend (LGBM+XGB+Cat)",
        ],
        "family": ["linear", "bagging", "boosting", "boosting", "boosting", "ensemble"],
        "chrono_AUC": [
            roc_auc_score(y_va, pred_lr),
            roc_auc_score(y_va, pred_rf),
            roc_auc_score(y_va, pred_t["lgbm"]),
            roc_auc_score(y_va, pred_t["xgb"]),
            roc_auc_score(y_va, pred_t["cat"]),
            roc_auc_score(y_va, blend_t),
        ],
    })
    .sort_values("chrono_AUC", ascending=False)
    .reset_index(drop=True)
)
display(family_scores)

# %% [markdown]
# **The boosters dominate the linear/bagging baselines**, confirming the EDA's hint that the signal is non-linear and interaction-driven. The three boosters score within a whisker of each other but make _different_ errors — so their **rank-average blend edges out every single model**. That blend is our final model.
#

# %% [markdown]
# ## 7.3 The key ablation: does the linear time index help?
#
# This is the experiment that separates v2 from v1. We compare the blend **with** and **without** `days_since_epoch` on the future holdout, and also test a rejected idea (recency sample-weighting).
#

# %%
pred_n = {
    name: fit_predict(name, Xtr_n, y_tr, Xva_n) for name in ("lgbm", "xgb", "cat")
}
blend_n = rank_avg(list(pred_n.values()))

# rejected idea: down-weight old rows by a 1-year half-life
age = (tr_raw["Course_Start_Date"].max() - tr_raw["Course_Start_Date"]).dt.days
w = np.power(0.5, age / 365.0).values
pred_recency = fit_predict("lgbm", Xtr_t, y_tr, Xva_t, sample_weight=w)

# reference: optimistic random split (leaks the future)
tr_r, va_r = train_test_split(
    train_raw, test_size=0.2, random_state=SEED, stratify=train_raw[TARGET]
)
fm_r = make_freq_maps(tr_r, va_r)
Xtr_r = build_features(tr_r, fm_r)
Xva_r = build_features(va_r, fm_r)
align_categories(Xtr_r, Xva_r)
pred_random = fit_predict("lgbm", Xtr_r, tr_r[TARGET].values, Xva_r)

ablation = pd.DataFrame({
    "configuration": [
        "Blend, no time index",
        "Blend, + time index (chosen)",
        "LightGBM + recency weights (rejected)",
        "LightGBM, random split (optimistic — do NOT trust)",
    ],
    "AUC": [
        roc_auc_score(y_va, blend_n),
        roc_auc_score(y_va, blend_t),
        roc_auc_score(y_va, pred_recency),
        roc_auc_score(va_r[TARGET].values, pred_random),
    ],
    "validation": ["chrono", "chrono", "chrono", "random"],
})
display(ablation)

# %% [markdown]
# **What the table shows.**
#
# - Adding the **time index improves the blend on the future holdout** — the
#   counter-intuitive win. It only shows up _because_ we validate on the future; a
#   random split would have hidden (or reversed) it.
# - **Recency weighting is rejected**: the time index already captures the trend,
#   so re-weighting adds nothing.
# - The **random split scores far higher (~0.96)** than any honest chrono number —
#   exactly the trap v1 fell into (random CV 0.944 → leaderboard 0.886). We ignore
#   it.
#
# ### v1 vs v2, on the same footing
#
# Scored on identical splits, v2 beats v1's approach (date dropped, top-$k$ collapse, one-hot, single XGBoost) on **both** protocols — so the gain is real, not an artefact of changing metrics:
#
# | Pipeline                            | Random 80/20 | Chrono holdout |
# | ----------------------------------- | ------------ | -------------- |
# | v1 (drop date, one-hot, single XGB) | ≈ 0.940      | ≈ 0.905        |
# | **v2 (this notebook)**              | ≈ 0.962      | **≈ 0.916**    |
#
# Calibrating against the known v1 gap (0.905 chrono → 0.886 real) maps v2's ~0.916 to roughly **0.89 on the leaderboard — which is what we scored (0.889314, 1st of 32).**
#

# %% [markdown]
# ## 7.4 Hyper-parameter tuning (on the chronological holdout)
#
# Tuning must be done against the _future_ holdout, not random CV — otherwise we would optimise for the wrong distribution. We illustrate with a focused search over LightGBM's two most important capacity knobs, `num_leaves` (tree complexity) and `min_child_samples` (regularisation via minimum leaf size), watching for the bias/variance sweet spot.
#

# %%
tuning_rows = []
for num_leaves in (31, 63, 127):
    for min_child in (20, 40, 80):
        m = get_lgbm(num_leaves=num_leaves, min_child_samples=min_child)
        m.fit(
            Xtr_t,
            y_tr,
            categorical_feature=Xtr_t.select_dtypes("category").columns.tolist(),
        )
        tuning_rows.append({
            "num_leaves": num_leaves,
            "min_child_samples": min_child,
            "chrono_AUC": round(roc_auc_score(y_va, m.predict_proba(Xva_t)[:, 1]), 4),
        })
tuning = (
    pd
    .DataFrame(tuning_rows)
    .sort_values("chrono_AUC", ascending=False)
    .reset_index(drop=True)
)
display(tuning)
print(
    "\nChosen for the pipeline: num_leaves=63, min_child_samples=40 "
    "(near-best AUC without over-growing the trees)."
)

# %% [markdown]
# The grid is flat near the top — the model is not fragile to these choices — and the pipeline's `num_leaves=63, min_child_samples=40` sits at the sweet spot: large trees (127 leaves) with small leaves start to overfit the past and do not improve the future score. XGBoost and CatBoost were tuned the same way (depth 6, moderate `learning_rate`, `reg_lambda`/`l2_leaf_reg` for regularisation).
#

# %% [markdown]
# # 8. Model evaluation
#
# AUC is the competition metric, but operations act on a **threshold**. We evaluate the chosen blend on the chronological holdout with the full confusion-matrix family of metrics, plus ROC and precision–recall curves.
#

# %% [markdown]
# ## 8.1 ROC & precision–recall curves
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for pred, name in [
    (pred_lr, "Logistic Regression"),
    (pred_rf, "Random Forest"),
    (pred_t["lgbm"], "LightGBM"),
    (pred_t["xgb"], "XGBoost"),
    (pred_t["cat"], "CatBoost"),
    (blend_t, "Blend (final)"),
]:
    fpr, tpr, _ = roc_curve(y_va, pred)
    axes[0].plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc(fpr, tpr):.3f})",
        lw=2.2 if name.startswith("Blend") else 1.2,
    )
    prec, rec, _ = precision_recall_curve(y_va, pred)
    axes[1].plot(
        rec,
        prec,
        label=f"{name} (AP={average_precision_score(y_va, pred):.3f})",
        lw=2.2 if name.startswith("Blend") else 1.2,
    )
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set(
    xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curve"
)
axes[0].legend(loc="lower right", fontsize=8)
axes[1].axhline(y_va.mean(), ls="--", color="grey", alpha=0.7)
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve")
axes[1].legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# The blend's ROC curve sits on or above every individual model across the range, which is why it wins on AUC.
#

# %% [markdown]
# ## 8.2 Confusion matrix & threshold metrics
#
# At the default 0.5 threshold we turn the blend's scores into hard decisions and read off the operational metrics.
#

# %%
y_hat = (blend_t >= 0.5).astype(int)
cm = confusion_matrix(y_va, y_hat)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt=",d",
    cmap="Blues",
    cbar=False,
    xticklabels=["pred completed", "pred dropped"],
    yticklabels=["true completed", "true dropped"],
    ax=ax,
)
ax.set_title("Confusion matrix — blend @ 0.5 (chrono holdout)")
plt.tight_layout()
plt.show()

print(
    classification_report(y_va, y_hat, target_names=["completed", "dropped"], digits=3)
)
print(f"AUC (threshold-free): {roc_auc_score(y_va, blend_t):.4f}")

# %% [markdown]
# **What each metric means here.**
#
# - **Precision (dropped)** — of the orders we flag as high-risk, how many really
#   cancel. Low precision ⇒ we waste follow-up effort / overbook wrongly.
# - **Recall (dropped)** — of the orders that really cancel, how many we catch.
#   Low recall ⇒ we get blindsided by cancellations.
# - **Accuracy / F1** — overall correctness; useful but threshold-dependent.
#
# Because operations can trade these off by moving the threshold (and the grade is AUC), we submit **probabilities, not hard labels**, and let the business pick the cut point that matches the cost of a missed cancellation vs a false alarm.
#

# %% [markdown]
# ## 8.3 Where is the model unsure?
#
# The distribution of predicted probabilities shows how _confident_ the model is and how many borderline cases it produces.
#

# %%
plt.figure(figsize=(9, 4.5))
sns.histplot(blend_t, bins=50, kde=True, color="teal")
plt.axvline(0.5, color="red", ls="--", label="decision boundary")
plt.axvspan(0.40, 0.60, color="orange", alpha=0.2, label="low-confidence zone")
plt.xlabel("predicted P(drop)")
plt.title("Prediction confidence — final blend (chrono holdout)")
plt.legend()
plt.tight_layout()
plt.show()

uncertain = ((blend_t > 0.40) & (blend_t < 0.60)).mean() * 100
print(f"share of holdout in the 0.40–0.60 low-confidence zone: {uncertain:.1f}%")

# %% [markdown]
# The scores are well spread toward the extremes (the model is decisive on most orders), with a minority of genuinely ambiguous cases in the 0.4–0.6 band. We dig into _why_ individual cases land there using SHAP next.
#

# %% [markdown]
# # 9. Interpretation with SHAP
#
# For interpretation we analyse **one** representative model — **LightGBM+time** — as the assignment requires. SHAP (SHapley Additive exPlanations) attributes each prediction to its features via a game-theoretic allocation, giving both global importance and per-observation explanations.
#

# %%
import shap
from lightgbm import LGBMClassifier

shap_model = LGBMClassifier(
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
shap_model.fit(
    Xtr_t, y_tr, categorical_feature=Xtr_t.select_dtypes("category").columns.tolist()
)
print(
    f"LightGBM+time chrono AUC: {roc_auc_score(y_va, shap_model.predict_proba(Xva_t)[:, 1]):.4f}"
)

X_shap = Xva_t.sample(min(2000, len(Xva_t)), random_state=SEED)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:  # some shap versions return (n, features, classes)
    shap_values = shap_values[:, :, 1]

# %% [markdown]
# ## 9.1 Global importance (beeswarm + bar)
#

# %%
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title("SHAP summary (beeswarm) — LightGBM+time")
plt.tight_layout()
plt.show()

importance = (
    pd
    .DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": np.abs(shap_values).mean(0),
    })
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
top = importance.head(20)
ax = top.sort_values("mean_abs_shap").plot.barh(
    x="feature", y="mean_abs_shap", legend=False, figsize=(8, 7), color="#4c72b0"
)
ax.set_xlabel("mean |SHAP value|")
ax.set_title("Top 20 features by SHAP importance")
plt.tight_layout()
plt.show()
display(top)

# %% [markdown]
# **Reading the SHAP importance.** The drivers line up with the EDA: `Payment_Terms`, the **time index** (`days_since_epoch`) and seasonality, the **frequency-encoded IDs** (`Agent_ID_freq`, `Company_ID_freq`), registration timing, and the engineered **history/ratio** features. The prominence of the time index confirms Section 7.3: the model genuinely leans on _when_ an order occurs to score the future correctly.
#
# ### The `Payment_Terms` leakage check
#
# EDA flagged prepaid-non-refundable as suspiciously strong. SHAP confirms it is influential but **not a lone dominator** — the model spreads its weight across many features. That, plus the fact that payment terms are set _at registration_ (before any cancellation), argues it is a genuine early risk signal (buyer's remorse / procurement friction) rather than a post-hoc label leak. We keep it but would confirm with the data owner before productionising.
#

# %% [markdown]
# ## 9.2 A dependence plot for the top signal
#

# %%
top_feat = (
    importance.loc[importance["feature"] != "Payment_Terms", "feature"].iloc[0]
    if importance["feature"].iloc[0] == "Payment_Terms"
    else importance["feature"].iloc[0]
)
try:
    shap.dependence_plot(
        top_feat, shap_values, X_shap, interaction_index=None, show=False
    )
    plt.title(f"SHAP dependence — {top_feat}")
    plt.tight_layout()
    plt.show()
except Exception as e:  # dependence_plot is finicky with category dtype
    print(f"(dependence plot skipped for {top_feat}: {e})")

# %% [markdown]
# ## 9.3 Explaining a single low-confidence order
#
# To answer "_how_ does the model handle uncertain observations?", we pick a borderline case (P(drop) ≈ 0.5) and decompose its prediction. The waterfall shows which features pushed the score up vs down — for ambiguous orders these forces roughly cancel.
#

# %%
sample_scores = shap_model.predict_proba(X_shap)[:, 1]
borderline = np.where((sample_scores > 0.45) & (sample_scores < 0.55))[0]
idx = int(borderline[0]) if len(borderline) else 0
base = explainer.expected_value
if isinstance(base, (list, np.ndarray)):
    base = np.asarray(base).ravel()[-1]

print(
    f"explaining order at sample position {idx} — model P(drop)={sample_scores[idx]:.3f}"
)
shap.plots._waterfall.waterfall_legacy(
    base,
    shap_values[idx],
    feature_names=list(X_shap.columns),
    max_display=14,
    show=False,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# For this borderline order the positive and negative contributions nearly balance — which is exactly why the model is unsure. Operationally these are the orders where a human follow-up adds the most value, since the model is honestly signalling "could go either way".
#

# %% [markdown]
# # 10. Final model & submission
#
# We refit all three boosters on **every** labelled row (train has no future to leak against once we are producing the final scores), rank-average their test predictions, and write the submission CSV. This is exactly `pipeline_v2.py`.
#

# %%
def build_submission(out_path="data/Group_27_Submission_v3.csv", write=False):
    fm = make_freq_maps(train_raw, test_raw)
    X_train_full = build_features(train_raw, fm)
    X_test = build_features(test_raw, fm)
    align_categories(X_train_full, X_test)
    y_full = train_raw[TARGET].values

    preds = []
    for name in ("lgbm", "xgb", "cat"):
        preds.append(fit_predict(name, X_train_full, y_full, X_test))
        print(f"fitted {name} on {len(X_train_full):,} rows")

    submission = pd.DataFrame({
        "Client_ID": test_raw["Client_ID"],
        "Drop_Probability": rank_avg(preds),
    })
    if write:
        submission.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({len(submission):,} rows)")
    return submission


# Set write=True to (re)generate the CSV. We write to a *v3* path so we never
# clobber the officially-scored data/Group_27_Submission.csv by accident.
submission = build_submission(write=False)
display(submission.head())
print(submission["Drop_Probability"].describe())

# %% [markdown]
# The output has the required schema (`Client_ID`, `Drop_Probability`), one row per test order, with probabilities spread across `[0, 1]`. It matches the officially-scored file (`data/Group_27_Submission.csv`) that placed **1st of 32** at **AUC 0.889314**.
#

# %% [markdown]
# # 11. Conclusions — executive summary
#
# **The problem.** Predict the probability a B2B course registration is cancelled, so Nova Academy can stop sinking cost into orders that will fall through.
#
# **The process.** A full CRISP-DM pass: EDA → cleaning → feature engineering → chronologically-validated modelling → evaluation → SHAP interpretation.
#
# **The central discovery.** _The test set is the future._ Train ends where test begins, and the drop rate drifts over time (confirmed by adversarial validation, AUC ≫ 0.5). This reframed the task as forecasting and made a
#
# **chronological holdout** — not a random split — the only trustworthy yardstick. It is the single biggest reason our score rose.
#
# **What worked.**
#
# 1. **Time-aware validation** — every decision judged on a future window.
# 2. **Thorough categorical cleaning** + **label-free frequency encoding**, so
#    dirty high-cardinality IDs became usable without a one-hot explosion (our
#    answer to the curse of dimensionality).
# 3. **Domain-driven features** — composition/ratio/history features and, counter
#    to intuition, a **linear time index** that lets trees score future rows in
#    the latest regime.
# 4. **A rank-average blend** of LightGBM + XGBoost + CatBoost, which beat every
#    single model on the future holdout.
#
# **Key findings for the business.** The strongest, actionable drivers are payment
# terms (prepaid-non-refundable is high-risk — worth a process review), the registration channel and enrolment type, agent/company identity and the presence of a company id, how early the group registered, and pre-course engagement (support tickets ⇒ commitment).
#
# **Result.** Leaderboard **AUC 0.889314 — 1st of 32 groups**, far above the 0.70
# bar.
#
# **Ways to push further (not yet implemented).**
#
# - Rolling multi-fold time-series CV for tuning (vs the single chrono holdout).
# - Careful leave-one-out / target encoding of `Agent_ID` / `Company_ID` (leakage
#   risk — must be fit inside CV folds).
# - Confirm the `Payment_Terms` signal with the data owner before leaning on it in
#   production.
# - Probability **calibration** (isotonic/Platt) if the business needs the scores
#   to read as true probabilities rather than just a good ranking.
#
