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
# This is the **integrated CRISP-DM-style analysis notebook** for the project: business understanding → data exploration → cleaning → feature engineering → modelling → evaluation → interpretation → conclusions.
#
# Running this notebook top to bottom executes the complete data preparation, model training, and submission-generation pipeline.
#
# **Key Modeling Challenge:** The hidden test set represents the **future** — it starts exactly where the training data ends and runs four months forward. Therefore, the task is a forecasting problem rather than simple interpolation, making a standard random train/validation split overly optimistic. Respecting the chronological order during validation ensures a realistic performance estimate, yielding a generalized test AUC of **0.889**.
#

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["svg.fonttype"] = "path"
try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass

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
CHRONO_CUTOFF = "2017-01-01"  # final ~4 train months, matching the hidden test horizon
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
# **Why probability-like scores, not hard labels.** Operations needs to _rank_ and _size_ risk, not receive a yes/no. The grading metric is **AUC** — a threshold-free measure of how well a score ranks droppers above non-droppers. The selected CSV uses a rank-average boosted-tree score optimized for this ranking task; threshold diagnostics below use mean model probabilities separately because the rank score is not calibrated. The passing bar is AUC ≥ 0.70.
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
# If the local `data/` files are missing, `load_raw` falls back to the raw CSVs in the GitHub repository. This keeps the notebook runnable when shared as a single `.ipynb`, as long as the machine has internet access and the repository data files are reachable.
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
# Our exploration is shaped by a key temporal shift in the data. We analyze this major time trend first, followed by a detailed look at the individual features.
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
# **Reading the plot.** Training runs `2015-07 → 2017-04`; the test window starts exactly where training ends and continues to `2017-08`, with essentially **zero overlap in time**. The drop rate also **drifts year to year, indicating the relationship changes over time**.
#
# **Consequence.** We treat the task as "train on the past, predict the future". A random train/validation split leaks future rows into training and produces an _optimistic_ score that does not transfer to the leaderboard. This time structure drives our validation strategy (Section 6) and motivates a feature choice (the time index, Section 5). It likely explains much of the gap between random-split validation and leaderboard behavior, while other feature and model changes also contributed.
#
# **Action:** Based on this temporal drift, we reject a random train/validation split and implement a chronological validation window (Section 6) to ensure generalizability to the future.
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
# Missingness patterns are **consistent between train and test**, so a single imputation policy is safe to reuse for scoring.
#
# **Next**: is missingness itself a signal?
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
# **Missingness is informative.**
#
# - Rows _without_ a `Company_ID` drop at a noticeably higher rate than rows with one — a registration made through a known company is a more committed order.
# - `Agent_ID` missingness shows a different profile too. This justifies explicit **presence flags** (`has_company_id`, `has_agent_id`) rather than silently imputing these away.
#
# **Action:** Rather than erasing these signals with standard median/mode imputation, we will engineer explicit presence flags for high-impact missing indicators.
#

# %% [markdown]
# ## 3.3 Categorical data quality (and why cleaning is mandatory)
#
# In a clean dataset, each text column should contain only a few distinct categories (such as catering package, lanyard color, or payment term). We scan the raw text columns to inspect the unique values and check for cardinality inflation due to formatting inconsistencies:
#

# %%
TEXT_COLS = list(train_raw.select_dtypes(include=["object"]).columns)


def count_unique_vals(df, col):
    return df[col].nunique()


def most_common_cats(df, col, n=15):
    return df[col].value_counts(normalize=True).head(n)


# N_COUNT = 15
# for col in TEXT_COLS:
#     top_values = most_common_cats(train_raw, col)

#     print(f"""
# {'=' * 80}
# Column: {col}
# Unique values: {count_unique_vals(train_raw, col)}

# Top {len(top_values)} categories:
# {",   ".join(f"{value}: ({share * 100:.1f}%)" for value, share in top_values.items())}
# """)
N_COUNT = 15

for col in TEXT_COLS:
    top_values = most_common_cats(train_raw, col)

    cats = [f"{value!r}: ({share * 100:.1f}%)" for value, share in top_values.items()]

    cats_str = "\n".join(" | ".join(cats[i : i + 3]) for i in range(0, len(cats), 3))

    print(f"""
{'=' * 80}

Column: {col}
Unique values: {count_unique_vals(train_raw, col)}

Top {len(top_values)} categories:

{cats_str}
""")

# %% [markdown]
# Two regimes: `Welcome_Gift_Type`, `Requested_Lab_Config`, `Assigned_Lab_Config` are
# clean (a few levels); the other seven carry hundreds of uniques — `'BLUE'`, `'blue'`,
# `'  Blue  '` are one value typed three ways. To confirm these are spelling variants,
# not real categories, we collapse each string to a canonical form and group by it.
#

# %%
# Placeholder strings that mean "missing", in any casing/padding after canonicalisation.
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
# Both codes denote China (found in EDA); fold the rarer one into the common one.
COUNTRY_ALIASES = {"cn": "chn"}


def canonicalize(s: pd.Series) -> pd.Series:
    """Map dirty categorical text to its canonical form: lowercase, injected
    punctuation stripped, single-spaced. No NaN masking — that is normalize_cats' job."""
    s = s.astype("string").str.strip().str.lower()
    return (
        s.str
        .replace(r"\band\b", "&", regex=True)
        .str.replace(r"[^a-z0-9&() .+-]+", "", regex=True)  # strip # ! * ? etc.
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def variant_collapse(df, cols):
    """For each column: the single real category that the most raw spellings collapse into,
    plus how many raw strings are pure junk (mapped to missing, not to a category)."""
    rows = []
    for col in cols:
        raw = df[col]
        key = canonicalize(raw)
        real_key = key.mask(
            key.isin(COMMON_NANS)
        )  # ignore junk when grouping categories
        variants = (
            pd
            .DataFrame({"raw": raw, "key": real_key})
            .dropna(subset=["key"])
            .groupby("key")["raw"]
            .nunique()
            .sort_values(ascending=False)
        )
        top = variants.index[0]
        sample = raw[real_key == top].value_counts().index[:8].tolist()
        rows.append({
            "column": col,
            "true_category": top,
            "raw_spellings_of_it": int(variants.iloc[0]),
            "junk_strings": int(raw[key.isin(COMMON_NANS)].nunique()),
            "sample_raw_spellings": ", ".join(map(repr, sample)),
        })
    return pd.DataFrame(rows)


inflated_cols = [c for c in TEXT_COLS if train_raw[c].nunique() > 20]
with pd.option_context("display.max_colwidth", None):
    display(variant_collapse(train_raw, inflated_cols))

# %% [markdown]
# Each inflated column collapses to one category absorbing up to ~200 spellings (`pay upon start` = 133); `Origin_Country` collapses least because its codes are genuinely distinct. A rarer issue is placeholder junk (`'Unknown'`, `'?'`), which must become **missing**, not a level. `normalize_cats` canonicalizes spelling variants and nulls placeholder junk, ensuring both train and test sets share the same clean categories.
#
# **Action:** To prevent cardinality explosion, we implement spelling normalization in Section 4.1 and label-free frequency mapping in Section 5.2.
#

# %%
CAT_COLS = TEXT_COLS + ["Agent_ID", "Company_ID"]


def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise every categorical, then map junk placeholders to NaN."""
    df = df.copy()
    for col in CAT_COLS:
        s = canonicalize(df[col])
        df[col] = s.mask(s.isin(COMMON_NANS))
    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    return df


# %%
clean_train = normalize_cats(train_raw)

cardinality_change = (
    pd
    .DataFrame({
        "raw_unique": {c: train_raw[c].nunique() for c in TEXT_COLS},
        "clean_unique": {c: clean_train[c].nunique() for c in TEXT_COLS},
    })
    .assign(collapsed=lambda t: t["raw_unique"] - t["clean_unique"])
    .sort_values("collapsed", ascending=False)
)
display(cardinality_change)

# %% [markdown]
# Inflated columns collapse to their true counts (`Payment_Terms` 236 → 3,
# `Client_Category` 505 → 7), controls untouched. This is cleaning, not feature
# engineering.
#

# %% [markdown]
# ## 3.4 Which categories actually relate to dropping?
#
# For the cleaned categoricals we plot the drop rate of the most frequent levels against the dataset mean. A level far from the dashed mean line carries signal.
#


# %%
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
# - **`Payment_Terms` is the strongest categorical signal in this view.** _Prepaid (non-refundable)_ orders drop far more often than _pay-on-start_ ones.
#   - This is surprising (why cancel something you can't refund?) and strong enough that we analyze this relationship further in the model interpretation phase (Section 9).
# - **`Client_Category`**: big-tech / multinational segments drop above average; fintech/banking and industrial/IoT below.
# - **`Submission_Source`**: direct-website and dedicated-sales orders are lower risk than B2B-platform / reseller traffic.
# - **`Enrollment_Type`**: organisational / affiliated arrangements are lower risk than general or one-off contractual admissions.
#
# By contrast, `Lanyard_Color` and `Welcome_Gift_Type` show no stable pattern and have no business reason to matter — candidates to drop as noise.
#

# %% [markdown]
# ### Country, agent, and acquisition context
#
# The categorical EDA suggests that dropout risk is not only attached to course logistics. Several business-context fields move together: country, agent, company presence, payment terms, and registration source.
#

# %%
country_min_n = 150
country_top_n = 12


overall_drop = clean_train[TARGET].mean()

country_stats = (
    clean_train
    .groupby("Origin_Country", dropna=False)[TARGET]
    .agg(count="size", drop_rate="mean")
    .assign(
        drop_rate_pct=lambda d: d["drop_rate"] * 100,
        lift_pp=lambda d: (d["drop_rate"] - overall_drop) * 100,
    )
)

top_by_size = country_stats.sort_values("count", ascending=False).head(country_top_n)

# Pick countries whose drop rate is farthest from the overall mean, after filtering tiny countries.
extreme_by_lift = (
    country_stats[
        country_stats["count"] >= country_min_n
    ]  # ignore countries with too few rows for a stable rate
    .iloc[
        lambda d: (
            d["lift_pp"].abs().sort_values(ascending=False).index.map(d.index.get_loc)
        )
    ]  # sort by distance from the overall drop rate
    .head(country_top_n)
)


def plot_country_dropout(stats, title, ax):
    stats = stats.sort_values("drop_rate_pct")
    labels = [
        f"{idx if pd.notna(idx) else '<missing>'} (n={int(row['count']):,})"
        for idx, row in stats.iterrows()
    ]
    colors = np.where(stats["lift_pp"] >= 0, "#c44e52", "#4c72b0")

    ax.barh(labels, stats["drop_rate_pct"], color=colors)
    ax.axvline(
        overall_drop * 100,
        ls="--",
        color="black",
        lw=1,
        label=f"overall ({overall_drop * 100:.1f}%)",
    )
    ax.set_xlabel("drop rate (%)")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

plot_country_dropout(
    top_by_size,
    f"Drop rate by largest {country_top_n} countries",
    axes[0],
)

plot_country_dropout(
    extreme_by_lift,
    f"Most unusual country drop rates (n >= {country_min_n})",
    axes[1],
)

plt.tight_layout()
plt.show()

display(
    country_stats
    .sort_values("count", ascending=False)
    .head(country_top_n)[["count", "drop_rate_pct", "lift_pp"]]
    .round(2)
)

# %% [markdown]
# `Origin_Country` is a strong signal, and Portugal is the clearest case — common _and_
# far above the base rate in both plots, so not a rare-country fluke. But country alone
# isn't the whole story: a risky country can concentrate certain payment terms,
# channels, or agents, so we read it as context, not cause.
#

# %%
is_portugal = clean_train["Origin_Country"].eq("prt").fillna(False).to_numpy(dtype=bool)
country_group = np.where(is_portugal, "Portugal", "Other countries")

portugal_summary = (
    clean_train
    .assign(country_group=country_group)
    .groupby("country_group")[TARGET]
    .agg(count="size", drop_rate="mean")
    .assign(drop_rate_pct=lambda d: d["drop_rate"] * 100)
)

display(portugal_summary[["count", "drop_rate_pct"]].round(1))

# %% [markdown]
# The split confirms the gap is real, not a plotting artifact. Next, the identifier
# fields: `Agent_ID` and `Company_ID` are labels, not numbers, and may carry related
# signal.
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
# Both identifiers separate risk: frequent agents drop at very different rates, and
# _having_ a `Company_ID` nearly halves risk (42.5% → 21.2%). But risky agents may just
# be the ones assigned to risky countries — overlapping categorical signal, not numeric
# multicollinearity — so we test it directly: how well does `Agent_ID` predict
# `Origin_Country`?
#

# %%
agent_country_pairs = clean_train[["Agent_ID", "Origin_Country"]].dropna()

country_tr, country_va = train_test_split(
    agent_country_pairs, test_size=0.25, random_state=SEED
)
majority_country = country_tr["Origin_Country"].mode().iat[0]
agent_country_map = country_tr.groupby("Agent_ID")["Origin_Country"].agg(
    lambda s: s.value_counts().idxmax()
)
agent_country_pred = (
    country_va["Agent_ID"].map(agent_country_map).fillna(majority_country)
)

display(
    pd.DataFrame({
        "check": ["majority country baseline", "agent modal country"],
        "accuracy": [
            country_va["Origin_Country"].eq(majority_country).mean(),
            agent_country_pred.eq(country_va["Origin_Country"]).mean(),
        ],
    }).round(3)
)

# %% [markdown]
# The modal-country check performs much better than the majority-country baseline, so agent and country contain overlapping information. The match is still not perfect, so neither field fully explains the other. We keep both signals, but encode them compactly later instead of one-hotting hundreds of levels.
#

# %% [markdown]
# ## 3.5 Numeric features: summary, correlation, and suspects
#
# We examine the distribution, summary statistics, and correlations of the numeric features:
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
# **Reading the heatmap.** No single raw numeric feature correlates strongly with the target. That is consistent with **non-linear / interaction-driven** signal, which we test later by comparing linear baselines with gradient-boosted trees. Inter-feature correlations are mild, so there is no severe multicollinearity forcing us to drop columns; the dimensionality problem lives in the _categoricals_, not here (Section 5.2).
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

# %% [markdown]
# ## 3.7 EDA synthesis — from numbers to a business story
#
# Stepping back, the individual signals are not independent curiosities; most of them
# collapse into a few coherent themes about _who cancels and why_.
#
# **1. One latent driver: buyer commitment.** The most stable signals all proxy how
# committed a group is at registration.
#
# - _More committed → drops less:_ a known `Company_ID` (a vetted corporate buyer with
#   procurement accountability), pre-course support tickets (a group already investing
#   effort), high-touch acquisition (direct sales, organisational enrollment), and
#   compliance-driven sectors such as fintech/banking where training is closer to
#   mandatory.
# - _Less committed → drops more:_ registrations with no company attached, low-friction
#   reseller/platform channels, and large multinational buyers whose bureaucracies
#   reprioritise and cut training budgets more readily.
#
# This helps explain why _missingness itself_ is predictive — a missing `Company_ID` is a
# commitment signal, not merely a gap to impute.
#
# **2. `Payment_Terms` looks endogenous.** Prepaid, non-refundable orders drop _more_, not less—the opposite of the naive "money is locked in" intuition. The most plausible reading is selection: the company likely demands prepayment precisely from deals it already judges risky, making the variable a _symptom_ of risk rather than a cause. We keep its strong predictive power, but analyze model sensitivity without this feature in Section 9.
#
# **3. Geography is a proxy, not a cause.** Portugal's 63.8% drop rate is real but
# confounded — `Agent_ID` predicts country well above chance, so "risky country" and
# "risky agent/channel" are entangled. We therefore treat country as _context_ and
# encode high-cardinality identity compactly (Section 5.2) instead of trusting the raw
# geographic label.
#
# **4. The environment is non-stationary.** The drop rate drifts year to year and the
# test window is strictly the future, so we are modelling a moving target, not a fixed
# law — which is what forces chronological validation (Section 6) and earns the time
# index its place as a feature (Section 5).
#
# **Modelling implication.** No single feature dominates linearly; the signal lives in
# the _interactions_ between commitment, channel, and timing — which is the case for
# gradient-boosted trees over a linear baseline (Section 7).
#

# %% [markdown]
# # 4. Missing-value handling & outlier analysis
#
# To ensure consistency, both cleaning and imputation are packaged inside our feature pipeline, applying identical logic to both train and test sets.
#

# %% [markdown]
# ## 4.1 Outliers: identify, justify, cap
#
# We look for values that are physically impossible or absurdly far from the bulk.
#


# %%
def sus_report(df, cols, max_mult=10):
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
display(sus_report(train_raw, num_cols))
print("Suspect columns — TEST")
display(sus_report(test_raw, num_cols))

# %% [markdown]
# The test set introduces no new forms of corruption, suggesting the same cleaning policy can be safely shared. Comparing the maximum values to the 99th percentile helps identify columns with extreme outliers:
#

# %%
TAIL_CHECK_COLS = ["Students_Count", "Practical_Hours", "Daily_Tuition_Cost"]

tail_long = []
for split, df in [("train", train_raw), ("test", test_raw)]:
    for col in TAIL_CHECK_COLS:
        s = df[col].dropna().rename("value").to_frame()
        s["split"] = split
        s["column"] = col
        tail_long.append(s)

tail_long = pd.concat(tail_long, ignore_index=True)
fig, axes = plt.subplots(1, len(TAIL_CHECK_COLS), figsize=(15, 4), sharey=False)
for ax, col in zip(axes, TAIL_CHECK_COLS):
    sub = tail_long[tail_long["column"] == col]
    sns.boxplot(
        data=sub,
        x="split",
        y="value",
        hue="split",
        order=["train", "test"],
        palette=["#4c72b0", "#dd8452"],
        showfliers=True,
        ax=ax,
        legend=False,
    )
    for xpos, split in enumerate(["train", "test"]):
        split_values = sub.loc[sub["split"] == split, "value"]
        max_value = split_values.max()
        n_at_max = int((split_values == max_value).sum())
        ax.annotate(
            f"max={max_value:g}\nn={n_at_max}",
            xy=(xpos, max_value),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )
    positive = sub.loc[sub["value"] > 0, "value"]
    if not positive.empty:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(positive.min() * 0.7, 0.1))
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("raw value (log scale)")
fig.suptitle("Raw tail check for candidate capped columns")
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

# %% [markdown]
# **Decisions and justification.**
# We cap (winsorize) extreme values instead of dropping rows, since the remaining features in those records still contain valuable signal.
#
# Based on the suspect-column screen and the boxplots, we apply three caps:
#
# - `Students_Count <= 10`: the only values above the normal tail are `9999` placeholder rows in both train and test. The cap keeps those rows as large groups without letting the placeholder behave like a real count.
# - `Practical_Hours` in `[0, 12]`: negative values are impossible, and `5000`/`10000` are clear placeholders. A 12-hour upper bound still allows a long practical day and prevents corrupted placeholder values from distorting the feature space.
# - `Daily_Tuition_Cost <= 600`: train has a single `5400` value, while the test maximum is 510. A cap of 600 leaves the observed test range untouched and prevents one corrupted training value from dominating cost calculations.
#
# Other flagged count columns (`Prev_Course_Dropouts`, `Prev_Course_Attended`, `Registration_Changes`, and test-side `Waiting_List_Days`) are heavy-tailed but plausible, so we leave them uncapped unless a concrete domain rule gives a cap.
#

# %%
CAP_RULES = {
    "Students_Count": {
        "lower": None,
        "upper": 10,
        "problem": "9999 placeholder",
        "reason": "corporate classroom groups are single-/low-double-digit",
    },
    "Practical_Hours": {
        "lower": 0,
        "upper": 12,
        "problem": "negative values and 10000",
        "reason": "course hours cannot be negative; 12 covers a long practical day",
    },
    "Daily_Tuition_Cost": {
        "lower": None,
        "upper": 600,
        "problem": "5400 value",
        "reason": "5400 is far beyond the valid fee range; 600 keeps the high-cost tail",
    },
}


def apply_cap(s, lower=None, upper=None):
    if lower is not None:
        s = s.clip(lower=lower)
    if upper is not None:
        s = s.clip(upper=upper)
    return s


cap_rows = []
for col, rule in CAP_RULES.items():
    lo, hi = rule["lower"], rule["upper"]
    train_changed = (
        train_raw[col].notna() & apply_cap(train_raw[col], lo, hi).ne(train_raw[col])
    ).sum()
    test_changed = (
        test_raw[col].notna() & apply_cap(test_raw[col], lo, hi).ne(test_raw[col])
    ).sum()
    action = f"clip to [{lo}, {hi}]" if lo is not None else f"clip to <= {hi}"
    cap_rows.append({
        "column": col,
        "raw_train_min": train_raw[col].min(),
        "raw_train_max": train_raw[col].max(),
        "problem": rule["problem"],
        "action": action,
        "train_rows_affected": int(train_changed),
        "test_rows_affected": int(test_changed),
        "reason": rule["reason"],
    })

display(pd.DataFrame(cap_rows))

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey="row")
for j, (col, rule) in enumerate(CAP_RULES.items()):
    before = train_raw[col].dropna()
    after = apply_cap(train_raw[col], rule["lower"], rule["upper"]).dropna()

    axes[0, j].hist(before, bins=50, color="#c44e52")
    axes[0, j].set_yscale("log")
    axes[0, j].set_title(f"{col}: raw")
    axes[0, j].set_xlabel(f"max={before.max():g}")

    axes[1, j].hist(after, bins=30, color="#55a868")
    axes[1, j].set_yscale("log")
    axes[1, j].set_title(f"{col}: clipped")
    axes[1, j].set_xlabel(f"max={after.max():g}")

axes[0, 0].set_ylabel("count (log)")
axes[1, 0].set_ylabel("count (log)")
plt.tight_layout()
plt.show()

# %% [markdown]
# The before/after distributions show that the caps remove isolated invalid tails while
# preserving the bulk of each feature.
#
# One more data-quality issue appears in the historical counters. Some rows have more
# recorded prior dropouts than prior attended courses, so those fields are not a clean
# numerator/denominator pair.
#

# %%
impossible = train_raw[
    train_raw["Prev_Course_Dropouts"] > train_raw["Prev_Course_Attended"]
]
print(f"rows where historical dropouts exceed historical attended: {len(impossible)}")

# %% [markdown]
# We keep the raw historical counters because the values can still carry risk signal, but
# any ratio derived from them should be interpreted as dropout **intensity**, not as a
# literal bounded probability. We therefore do not cap these counters or force their ratio
# into `[0, 1]`.
#

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
# - **Numerics** → Tree ensembles (LightGBM, XGBoost, CatBoost) handle missing values natively, learning a default split direction for NaNs. We therefore **pass numeric NaNs through** to the models rather than imputing them, preserving the informative missingness signals instead of masking them via imputation. Imputation is only applied to intermediate calculations in engineered ratios to avoid division errors.
#

# %% [markdown]
# # 5. Feature engineering & dimensionality
#
# Our engineered features are designed to capture non-linear interactions and domain-specific context. Below is the rationale for each feature:
#

# %% [markdown]
# ## 5.1 The engineered features and their rationale
#
# Each engineered feature either preserves raw information in a more model-friendly form, or condenses high-cardinality identifiers without introducing target leakage.
#
# | Raw signal                | Engineered feature(s)                                        | Why it helps                                                                                                                                 |
# | ------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
# | `Course_Start_Date`       | `start_month`, `start_dow`, `start_week`, `days_since_epoch` | Keeps seasonality and the time trend visible in the future test window.                                                                      |
# | Participant counts        | `total_participants`, `prof_share`                           | Converts raw counts into comparable group composition.                                                                                       |
# | Practical/theory hours    | `total_hours`, `practical_share`                             | Represents course intensity and hands-on share directly.                                                                                     |
# | Client history            | `prev_drop_rate = dropouts / (attended + 1)`                 | Adds a smoothed dropout-intensity signal; not treated as a bounded probability because the historical counters are not perfectly consistent. |
# | Tuition cost + hours      | `cost_x_days`                                                | Approximates the contract value at risk.                                                                                                     |
# | Requested vs assigned lab | `got_requested_lab`                                          | Captures whether the requested setup was honored.                                                                                            |
# | Missing company/agent IDs | `has_company_id`, `has_agent_id`                             | Preserves missingness signals seen in Section 3.2.                                                                                           |
# | Agent/company/country IDs | frequency encodings; native categoricals for tree boosters   | Keeps identity/frequency signal without one-hot explosion; frequencies use only covariate counts, never labels.                              |
#

# %% [markdown]
# ## 5.2 Dimensionality
#
# The curse of dimensionality here comes from the **identifiers**, not the numerics.
# `Agent_ID` alone has 204 distinct levels and `Origin_Country` 154, so naive one-hot
# encoding would turn each into hundreds of sparse binary columns.
#
# Because our candidate models handle categorical columns differently, we keep two practical preprocessing paths:
#
# - **Logistic Regression / MLP path.** These models need a fully numeric matrix. In Section 7.1 we therefore collapse rare category levels to `"other"`, one-hot encode the remaining levels, median-impute numeric missing values, and scale the matrix.
# - **Tree-booster path.** Gradient-boosted trees can avoid most of that one-hot expansion. We keep categorical columns as native `category` values, add label-free frequency encodings for high-cardinality IDs, and drop raw `Company_ID` while keeping `has_company_id` and `Company_ID_freq`.
#
# To prevent data leakage, frequency maps are fit on the training window only during validation. For the final submission, we compute frequencies over the combined train and test sets to obtain the most stable counts. Since frequencies use only feature counts and never reference the target label, this utilizes test set covariates without target leakage.
#
# **What "native categorical" means for the tree path.** We keep these columns in pandas' `category`
# dtype: a list of levels plus one integer _code_ per row (`["blue","red","blue"]` →
# codes `[0,1,0]`), like an R `factor`. This is a storage format, not an encoding — the
# leverage is in how the boosters read it. A gradient-boosted tree treats the codes as
# **unordered labels** and, at each split, partitions the _set_ of levels into two
# groups (`level ∈ {A, C, F}?`) in a single node — expressing what one-hot needs many
# stacked one-vs-rest splits to approximate, and without ever materialising the extra
# columns. (LightGBM finds this partition with the Fisher (1958) sorted-gradient
# heuristic; XGBoost does a similar subset split, and CatBoost uses ordered target
# statistics.)
#


# %%
def make_freq_maps(*dfs):
    """Label-free frequency of each ID value across the supplied frames."""
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

    Mirrors the selected ``pipelines/pipeline_v2.build_features`` transform."""
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
    # +1 is Laplace smoothing for new clients; because CRM dropouts can exceed
    # attended counts, this is a dropout-intensity signal, not a probability.
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
# **The tree-compatible dimensionality strategy** combines three levers, each doing a different job:
#
# 1. **Native `category` dtype** — the boosters split on level subsets directly, so the
#    feature matrix stays at 42 columns instead of ~435 (**393 dummy columns avoided**,
#    almost all from `Agent_ID` and `Origin_Country`).
# 2. **Frequency encoding** per ID (how common each value is) — one numeric column
#    that, unlike the dtype trick, can also be included in the continuous baselines.
# 3. **Dropping raw `Company_ID`** (highest cardinality) — keeping only its frequency
#    and presence flag.
#
# **Caveat.** This low-column representation is _tree-specific_. The continuous baselines still use one-hot columns, but the `min_count` rare-level collapse keeps that matrix bounded. And a low column count is not the whole battle — 204 sparse agent levels can still overfit — which is why levers 2–3 and the boosters' regularisation matter alongside native categorical splits.
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
# The classifier separates test from train **well above chance**, driven by the ID / frequency-style columns — the client population shifts over time. This is another reason not to trust a random split.
#
# ## 6.2 The chronological holdout
#
# We select every model and feature on a **chronological holdout**: fit on rows before `2017-01-01`, validate on the 2017 rows (~4 months, matching the real test window). This mirrors the leaderboard's "train on the past, score the future" setup, so improvements here should move in the same direction as the real score.
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

# feature matrices reused across all experiments below. For chronological
# validation, frequency maps are fit only on the past training window. The
# train+test map above is reserved for submission-time label-free transductive scoring.
freq_maps_chrono = make_freq_maps(tr_raw)
Xtr_t = build_features(tr_raw, freq_maps_chrono, add_time=True)
Xva_t = build_features(va_raw, freq_maps_chrono, add_time=True)
align_categories(Xtr_t, Xva_t)
Xtr_n = build_features(tr_raw, freq_maps_chrono, add_time=False)
Xva_n = build_features(va_raw, freq_maps_chrono, add_time=False)
align_categories(Xtr_n, Xva_n)

# %% [markdown]
# # 7. Model experiments & tuning
#
# We follow the assignment's requirement of **at least three models from different families**, each with a short description and its hyper-parameters, and tune on the chronological holdout.
#

# %% [markdown]
# ## 7.1 The model families
#
# We select a candidate algorithm from each tier of capacity, tune each (7.2), and compare them head-to-head (7.3). After identifying the strongest family, we focus on refining its performance (7.4).
#
# Each model family is paired with its appropriate preprocessing pipeline: bounded one-hot and scaling for the continuous baselines, and native categorical handling for the tree boosters.
#
# - **Logistic Regression** — the linear baseline. Fast and interpretable, but limited
#   to linear boundaries in the encoded space, so it is expected to lag on this
#   interaction-heavy data. Key hyper-parameter: inverse-regularisation `C`.
# - **MLP neural network** — a dense non-linear baseline. Flexible, but needs explicit
#   encoding, imputation, scaling, and an L2 penalty to control variance. Key
#   hyper-parameters: hidden-layer size, `alpha`, learning rate.
# - **Gradient-boosted trees (XGBoost)** — our tree model. Trees are built _sequentially_,
#   each correcting the residual of the last; state of the art for tabular data, with
#   native categorical/missing handling. We utilize **XGBoost** as our representative tree-based model, and—once trees win the comparison—expand to other boosters in 7.4.
#   boosters. Key hyper-parameters: number of trees, `learning_rate`, tree size
#   (`max_depth` / `num_leaves`), and regularisation (`reg_lambda`, `min_child_*`).
#


# %%
def get_lgbm(**kw):
    p = dict(
        n_estimators=700,
        learning_rate=0.03,
        # 63 = 2^6 - 1, aligning LightGBM leaf capacity with depth-6 XGBoost/CatBoost.
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
    p = dict(
        iterations=1200,
        learning_rate=0.05,
        # Depth 6 matches the selected XGBoost depth and the LightGBM leaf budget above.
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        verbose=False,
        eval_metric="AUC",
    )
    p.update(kw)
    return CatBoostClassifier(**p)


def fit_predict_pair(name, X_tr, y_tr, X_va, sample_weight=None, **model_params):
    """Fit one booster and return P(drop) on train and validation."""
    if name == "cat":
        cat_idx = [
            i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) == "category"
        ]
        X_tr2, X_va2 = X_tr.copy(), X_va.copy()
        for c in X_tr2.columns[cat_idx]:
            X_tr2[c] = X_tr2[c].astype(str)
            X_va2[c] = X_va2[c].astype(str)
        m = get_cat(cat_features=cat_idx, **model_params)
        m.fit(X_tr2, y_tr, sample_weight=sample_weight)
        return m.predict_proba(X_tr2)[:, 1], m.predict_proba(X_va2)[:, 1]
    if name == "lgbm":
        m = get_lgbm(**model_params)
        m.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            categorical_feature=X_tr.select_dtypes("category").columns.tolist(),
        )
        return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]
    m = get_xgb(**model_params)
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]


def fit_predict(name, X_tr, y_tr, X_va, sample_weight=None):
    """Fit one booster ('lgbm'|'xgb'|'cat') and return P(drop) on X_va."""
    _, pred_va = fit_predict_pair(name, X_tr, y_tr, X_va, sample_weight)
    return pred_va


def rank_avg(preds):
    """Average of per-model rank-percentiles: preserves AUC ordering while
    ignoring calibration differences between models."""
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def encode_for_continuous_models(X_tr: pd.DataFrame, X_va: pd.DataFrame, min_count=30):
    """Bounded one-hot + median imputation for the LR/MLP baselines."""
    Xt, Xv = X_tr.copy(), X_va.copy()
    # for df in [Xt, Xv]:
    #     df.drop(columns=["days_since_epoch"], inplace=True)
    cat_cols = list(Xt.select_dtypes("category").columns)

    # Collapse rare levels to "other" so the one-hot matrix stays bounded and the
    # linear/MLP baselines do not overfit categories with only a few examples.
    for c in cat_cols:
        train_s = Xt[c].astype("string").fillna("missing")
        keep = train_s.value_counts()[lambda s: s >= min_count].index
        Xt[c] = train_s.where(train_s.isin(keep), "other")
        Xv[c] = (
            Xv[c]
            .astype("string")
            .fillna("missing")
            .where(lambda s: s.isin(keep), "other")
        )

    num_cols = [c for c in Xt.columns if c not in cat_cols]
    medians = Xt[num_cols].median()
    Xt_num = Xt[num_cols].fillna(medians).reset_index(drop=True)
    Xv_num = Xv[num_cols].fillna(medians).reset_index(drop=True)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    Xt_cat = ohe.fit_transform(Xt[cat_cols])
    Xv_cat = ohe.transform(Xv[cat_cols])
    names = ohe.get_feature_names_out(cat_cols)

    Xt_out = pd.concat([Xt_num, pd.DataFrame(Xt_cat, columns=names)], axis=1)
    Xv_out = pd.concat([Xv_num, pd.DataFrame(Xv_cat, columns=names)], axis=1)
    return Xt_out, Xv_out


# %% [markdown]
# ## 7.2 Hyper-parameter tuning: reading the bias–variance trade-off
#
# For each family, we sweep a single capacity/regularization parameter and plot the training and validation ROC-AUC scores. This directly exposes the bias-variance trade-off: as model capacity grows, training performance climbs toward 1.0, while validation performance eventually peaks and plateaus or declines (the variance regime). The stars mark the optimal settings selected for each final model.
#
# We tune and select parameters directly against validation ROC-AUC rather than the standard training cost function. Since ROC-AUC is rank-based, this allows the models to leverage higher flexibility (lower bias) at the cost of some variance, whereas tuning on the cost function would penalize probability scale shifts and yield overly conservative parameters.
#
# For the tree family, we tune **XGBoost** on its `max_depth` axis, keeping the boosting budget fixed without early stopping so that deeper trees are free to overfit and expose the variance threshold.
#

# %%
Xtr_enc, Xva_enc = encode_for_continuous_models(Xtr_t, Xva_t)
# We scale continuous inputs using StandardScaler to preserve the variance of naturally skewed columns (like Daily_Tuition_Cost) without compressing normal-range observations.
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_enc)
Xva_scaled = scaler.transform(Xva_enc)


def loss_auc(p_tr, p_va):
    """Train/validation metrics helper."""
    return {
        "train_logloss": log_loss(y_tr, p_tr),
        "val_logloss": log_loss(y_va, p_va),
        "train_AUC": roc_auc_score(y_tr, p_tr),
        "val_AUC": roc_auc_score(y_va, p_va),
    }


tuning_rows = []

# Linear baseline: inverse-regularisation C (higher C -> less regularisation).
for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
    m = LogisticRegression(C=C, max_iter=2000).fit(Xtr_scaled, y_tr)
    tuning_rows.append({
        "family": "Logistic Regression",
        "axis": "C  (less regularisation →)",
        "x": C,
        **loss_auc(
            m.predict_proba(Xtr_scaled)[:, 1], m.predict_proba(Xva_scaled)[:, 1]
        ),
    })

# Neural net: L2 penalty alpha; plot 1/alpha so "more capacity" points right.
for alpha in (1.0, 0.1, 0.01, 0.001, 0.0001):
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=alpha,
        learning_rate_init=0.001,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=SEED,
    ).fit(Xtr_scaled, y_tr)
    tuning_rows.append({
        "family": "MLP neural network",
        "axis": "1 / alpha  (less regularisation →)",
        "x": 1.0 / alpha,
        **loss_auc(
            m.predict_proba(Xtr_scaled)[:, 1], m.predict_proba(Xva_scaled)[:, 1]
        ),
    })

# Gradient-boosted trees (representative: XGBoost). Capacity axis = max_depth,
# fixed budget, no early stopping so deep trees can overfit.
for depth in (2, 3, 4, 5, 6, 8, 10):
    p_tr, p_va = fit_predict_pair(
        "xgb",
        Xtr_t,
        y_tr,
        Xva_t,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=depth,
        min_child_weight=5,
        reg_lambda=1.0,
    )
    tuning_rows.append({
        "family": "Gradient-boosted trees (XGBoost)",
        "axis": "max_depth  (more capacity →)",
        "x": depth,
        **loss_auc(p_tr, p_va),
    })

tuning = pd.DataFrame(tuning_rows)

# star = the setting we carry into the final models: near-top validation AUC
# with controlled variance (the best bias-variance compromise, not blind argmax).
selected_x = {
    "Logistic Regression": 0.001,  # C = 0.001
    "MLP neural network": 10.0,  # alpha = 0.1 (x = 10.0)
    "Gradient-boosted trees (XGBoost)": 6,  # max_depth = 6
}
tuning["selected"] = tuning.apply(
    lambda r: bool(np.isclose(r["x"], selected_x[r["family"]])), axis=1
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for i, (ax, family) in enumerate(zip(axes, selected_x)):
    d = tuning[tuning["family"] == family].sort_values("x")

    # Plot Train vs Validation ROC-AUC (single y-axis)
    ax.plot(
        d["x"],
        d["train_AUC"],
        color="#2b5c8f",
        linestyle="--",
        marker="o",
        alpha=0.7,
        label="train AUC",
    )
    ax.plot(
        d["x"],
        d["val_AUC"],
        color="#1b9e77",
        linestyle="-",
        marker="s",
        linewidth=2,
        label="val AUC",
    )

    # Highlight the selected star on the validation AUC curve
    star = d[d["selected"]].iloc[0]
    ax.scatter(
        star["x"],
        star["val_AUC"],
        marker="*",
        s=250,
        color="#ffd700",
        edgecolor="black",
        linewidths=1.5,
        zorder=10,
        label="selected setting",
    )

    ax.set_title(family, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel(d["axis"].iloc[0], fontsize=10)
    if i == 0:
        ax.set_ylabel("ROC-AUC Score", fontsize=10)

    # Ensure scales are correct
    if family != "Gradient-boosted trees (XGBoost)":
        ax.set_xscale("log")

    ax.grid(True, linestyle=":", alpha=0.6)

    # Choose clean legend placement
    loc = "center right" if family == "Logistic Regression" else "lower left"
    if family == "Gradient-boosted trees (XGBoost)":
        loc = "lower right"
    ax.legend(loc=loc, fontsize=8, frameon=True, facecolor="white", framealpha=0.9)

fig.suptitle(
    "Bias–variance: Training vs Validation ROC-AUC (star = selected setting)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

display(
    tuning.assign(
        train_logloss=lambda d: d["train_logloss"].round(4),
        val_logloss=lambda d: d["val_logloss"].round(4),
        train_AUC=lambda d: d["train_AUC"].round(4),
        val_AUC=lambda d: d["val_AUC"].round(4),
    )[
        [
            "family",
            "x",
            "train_logloss",
            "val_logloss",
            "train_AUC",
            "val_AUC",
            "selected",
        ]
    ]
)

# %% [markdown]
# **Reading the plot.** All three panels show the capacity sweeps directly in ROC-AUC:
#
# - Logistic Regression is **bias-dominated** — even at strong regularisation (C=0.001) it reaches its generalization limit, and adding capacity (higher C) only degrades validation AUC as it fits noise.
# - The MLP neural network shows more capacity but overfits past alpha=0.1 (x=10), where validation AUC begins to drop while training AUC continues to rise.
# - The gradient-boosted tree shows the classic bias-variance signature: training AUC climbs monotonically with depth, while validation AUC peaks at depth 6 and plateaus/declines as variance increases. We therefore select depth 6 for the final models.
#

# %% [markdown]
# ### Fitting the tuned basic models
#
# With each family's setting chosen, we fit the three basic models once on the
# chronological training window and score the 2017 holdout.
#

# %%
lr = LogisticRegression(C=0.001, max_iter=2000)
lr.fit(Xtr_scaled, y_tr)
pred_lr = lr.predict_proba(Xva_scaled)[:, 1]

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=0.1,
    learning_rate_init=0.001,
    max_iter=150,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=SEED,
)
mlp.fit(Xtr_scaled, y_tr)
pred_mlp = mlp.predict_proba(Xva_scaled)[:, 1]

# the tuned tree model (single XGBoost) — the blend in 7.4 reuses this prediction
pred_xgb = fit_predict("xgb", Xtr_t, y_tr, Xva_t)

# %% [markdown]
# ## 7.3 Family comparison: which family do we choose?
#
# The three tuned basic models meet on the same 2017 holdout. Logistic Regression and the
# MLP use the encoded/scaled matrix; XGBoost uses native categoricals.
#

# %%
family_scores = (
    pd
    .DataFrame({
        "model": ["Logistic Regression", "MLP neural network", "XGBoost (tree)"],
        "family": ["linear", "neural network", "gradient-boosted trees"],
        "chrono_AUC": [
            roc_auc_score(y_va, pred_lr),
            roc_auc_score(y_va, pred_mlp),
            roc_auc_score(y_va, pred_xgb),
        ],
    })
    .sort_values("chrono_AUC", ascending=False)
    .reset_index(drop=True)
)
display(family_scores)

# %% [markdown]
# The gradient-boosted tree performs clearly better on the chronological holdout (~0.04 AUC over both baselines), which matches the EDA: much of the signal is non-linear, categorical, and interaction-driven. Logistic Regression is a useful reference floor; the MLP adds flexibility, but it still has to reconstruct categorical splits, missingness flags, and thresholds from a one-hot matrix. From here we use the tree model as the main direction and try to improve it.
#

# %% [markdown]
# ## 7.4 Improving the gradient model
# After choosing trees, we try to improve them in two ways: **(a)** tune the boosting budget more carefully, then **(b)** blend in two more tree implementations.
#
# ### (a) Boosting budget: number of trees × learning rate
#
# The number of trees and the learning rate interact directly. Consistent with our validation choice, we evaluate the tree budget directly in ROC-AUC. We evaluate various tree counts across two learning rate settings:
#

# %%
budget_rows = []
for lr_rate in (0.1, 0.03):
    for n in (50, 100, 200, 400, 700, 1000):
        p_tr, p_va = fit_predict_pair(
            "xgb",
            Xtr_t,
            y_tr,
            Xva_t,
            n_estimators=n,
            learning_rate=lr_rate,
            max_depth=6,
            min_child_weight=5,
            reg_lambda=1.0,
        )
        budget_rows.append({
            "learning_rate": lr_rate,
            "n_trees": n,
            "train_logloss": log_loss(y_tr, p_tr),
            "val_logloss": log_loss(y_va, p_va),
            "train_AUC": roc_auc_score(y_tr, p_tr),
            "val_AUC": roc_auc_score(y_va, p_va),
        })
budget = pd.DataFrame(budget_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (lr_rate, sub) in zip(axes, budget.groupby("learning_rate")):
    sub = sub.sort_values("n_trees")
    ax.plot(
        sub["n_trees"],
        sub["train_AUC"],
        marker="o",
        ls="--",
        color="#2b5c8f",
        label="train AUC",
    )
    ax.plot(
        sub["n_trees"],
        sub["val_AUC"],
        marker="o",
        color="#d95f02",
        linewidth=2,
        label="val AUC",
    )

    # Highlight the point of maximum validation AUC
    best_row = sub.loc[sub["val_AUC"].idxmax()]
    ax.scatter(
        best_row["n_trees"],
        best_row["val_AUC"],
        marker="*",
        s=250,
        color="#ffd700",
        edgecolor="black",
        linewidths=1.5,
        zorder=10,
        label=f"best AUC ({int(best_row['n_trees'])} trees)",
    )

    ax.set_xlabel("number of trees (n_estimators)", fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel("ROC-AUC Score", fontsize=10)
    ax.set_title(f"Learning Rate = {lr_rate}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle(
    "Boosting budget: ROC-AUC vs number of trees, by learning rate (star = best AUC)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.show()

display(
    budget.assign(
        train_logloss=lambda d: d["train_logloss"].round(4),
        val_logloss=lambda d: d["val_logloss"].round(4),
        train_AUC=lambda d: d["train_AUC"].round(4),
        val_AUC=lambda d: d["val_AUC"].round(4),
    )
)

# %% [markdown]
# **Reading the plot.** At `lr = 0.1`, training AUC climbs rapidly but validation AUC peaks early (around 200 trees) and declines as the model over-boosts. At `lr = 0.03`, both curves rise more gradually, but the validation curve reaches a higher, more stable plateau (peaking at 700 trees): shrinkage trades compute for generalization. We therefore keep a **low learning rate (0.03) with a generous budget (~700 trees)** for the final models. The remaining tree hyper-parameters follow the same bias–variance logic rather than an exhaustive sweep: `max_depth` (7.2) and `min_child_weight` / `min_child_samples` bound tree complexity, `reg_lambda` penalises leaf weights, and `subsample` / `colsample_bytree` (row/column sampling) inject randomness that decorrelates the trees — all pushing toward lower variance.
#
# ### (b) Blend three boosters
#
# The three implementations differ in _inductive bias_, so their errors are only partially
# correlated:
#
# - **XGBoost** grows trees **level-wise** (balanced) with strong regularisation.
# - **LightGBM** grows **leaf-wise** (best-first) over histogram bins with GOSS/EFB — fast,
#   and finds different, often deeper splits.
# - **CatBoost** uses **symmetric (oblivious) trees** with **ordered boosting** and
#   **ordered target statistics**, resisting the target-leakage / prediction-shift that
#   naive categorical encoding introduces.
#
# Averaging _decorrelated_ estimators cancels part of the variance, so a **rank-average**
# blend should edge any single booster. (We rank-average, not probability-average, so each
# model's calibration scale is discarded and only its ordering counts — which is exactly
# what AUC rewards.)
#

# %%
pred_t = {
    "lgbm": fit_predict("lgbm", Xtr_t, y_tr, Xva_t),
    "xgb": pred_xgb,  # reuse the tuned single XGBoost from 7.3
    "cat": fit_predict("cat", Xtr_t, y_tr, Xva_t),
}
blend_t = rank_avg(list(pred_t.values()))
blend_prob = np.mean([pred_t[k] for k in pred_t], axis=0)

xgb_auc = roc_auc_score(y_va, pred_t["xgb"])
blend_check = (
    pd
    .DataFrame({
        "model": [
            "LightGBM (tuned)",
            "XGBoost (tuned)",
            "CatBoost (tuned)",
            "Rank-average blend (LGBM+XGB+Cat)",
        ],
        "chrono_AUC": [
            roc_auc_score(y_va, pred_t["lgbm"]),
            xgb_auc,
            roc_auc_score(y_va, pred_t["cat"]),
            roc_auc_score(y_va, blend_t),
        ],
    })
    .sort_values("chrono_AUC", ascending=False)
    .reset_index(drop=True)
)
blend_check["delta_vs_XGBoost"] = (blend_check["chrono_AUC"] - xgb_auc).round(4)
display(blend_check)

# %% [markdown]
# The three boosters perform comparably, but the rank-average blend provides a minor but stable generalization improvement by averaging out individual model variance. We therefore select the rank-average blend as our final modeling pipeline.
#

# %% [markdown]
# ## 7.5 The key ablation: does the linear time index help?
#
# With the model chosen, we isolate the single feature decision the chronological framing
# motivated: the linear time index `days_since_epoch`. We use the representative LightGBM
# (not the full blend) so the feature effect is not diluted by averaging, and we score
# three deliberate comparators against it:
#
# - **no time index** — the counterfactual, to measure the feature's marginal value;
# - **recency sample-weighting** — a plausible _alternative_ way to emphasise recent rows
#   (1-year half-life), to check the time index is not just a proxy for "trust recent
#   data";
# - **random 80/20 split** — not a rival model but a _diagnostic_, to quantify how much
#   the future-holdout protocol itself costs versus an optimistic split.
#
# **Why a time index should help.** The drop rate is non-stationary (Section 3.1), so
# _when_ an order occurs carries signal. Trees cannot extrapolate a raw feature beyond its
# training range, but the test window sits immediately after training: a monotone
# `days_since_epoch` lets late-period splits isolate the most recent regime, so test rows
# inherit the behaviour of the closest-in-time training data rather than the global
# average.
#

# %%
pred_lgbm_no_time = fit_predict("lgbm", Xtr_n, y_tr, Xva_n)

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
        "LightGBM, no time index",
        "LightGBM, + time index",
        "LightGBM + recency weights (rejected)",
        "LightGBM, random split (optimistic — do NOT trust)",
    ],
    "AUC": [
        roc_auc_score(y_va, pred_lgbm_no_time),
        roc_auc_score(y_va, pred_t["lgbm"]),
        roc_auc_score(y_va, pred_recency),
        roc_auc_score(va_r[TARGET].values, pred_random),
    ],
    "validation": ["chrono", "chrono", "chrono", "random"],
})
display(ablation)

# %% [markdown]
# **What the table shows.**
#
# - **Time Index**: Adding `days_since_epoch` improves performance on the future holdout. This benefit is only visible because we validate on a future window; a random split would have hidden this temporal effect.
# - **Recency Sample Weighting**: Applying temporal decay to training weights did not yield improvements over the baseline time index.
# - **Random Split**: A random 80/20 train/validation split yields an overly optimistic score (~0.96) because it leaks future data into the training phase. Using a chronological validation set prevents this validation inflation, providing a more reliable estimate of how the model generalizes to unseen future data.
#

# %% [markdown]
# # 8. Model evaluation
#
# AUC is the competition metric, but operations act on a **threshold**. We evaluate the chosen blend on the chronological holdout with ROC and precision–recall curves. For confusion-matrix diagnostics, we use the mean boosted-tree probability (`blend_prob`), because the submitted rank-average score (`blend_t`) is optimized for ranking and is not calibrated. Rank-averaging is good for AUC because it ignores model-specific calibration scales, but after converting predictions to rank percentiles, a threshold like `0.5` no longer means `P(drop) >= 0.5`.
#

# %% [markdown]
# ## 8.1 ROC & precision–recall curves
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for pred, name in [
    (pred_lr, "Logistic Regression"),
    (pred_mlp, "MLP"),
    (pred_t["lgbm"], "LightGBM"),
    (pred_t["xgb"], "XGBoost"),
    (pred_t["cat"], "CatBoost"),
    (blend_t, "Rank-average blend (selected)"),
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
# The blend has the best holdout AUC in this comparison, so it is the selected ranking score.
#

# %% [markdown]
# ## 8.2 Confusion matrix & threshold metrics
#
# At the default 0.5 threshold we turn the mean boosted-tree probabilities into hard decisions and read off the operational metrics. This is a diagnostic threshold, not the submitted rank-average score. Thresholding the rank-average at 0.5 would only flag roughly the riskier half of the rows; it would not be a probability cutoff.
#

# %%
y_hat = (blend_prob >= 0.5).astype(int)
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
ax.set_title("Confusion matrix — mean booster probability @ 0.5")
plt.tight_layout()
plt.show()

print(
    classification_report(y_va, y_hat, target_names=["completed", "dropped"], digits=3)
)
print(f"AUC of selected rank-average score: {roc_auc_score(y_va, blend_t):.4f}")
print(f"AUC of mean booster probability: {roc_auc_score(y_va, blend_prob):.4f}")

# %% [markdown]
# **What each metric means here.**
#
# - **Precision (dropped)** — of the orders we flag as high-risk, how many really
#   cancel. Low precision ⇒ we waste follow-up effort / overbook wrongly.
# - **Recall (dropped)** — of the orders that really cancel, how many we catch.
#   Low recall ⇒ we get blindsided by cancellations.
# - **Accuracy / F1** — overall correctness; useful but threshold-dependent.
#
# Because operations can trade these off by moving the threshold (and the grade is AUC), we submit a continuous risk score, not hard labels. If the business needs the score to read as a true probability, a separate calibration step should be added.
#

# %% [markdown]
# ## 8.3 Where is the model unsure?
#
# This distribution shows how predictions are spread, highlighting the proportion of borderline (uncertain) cases near the decision boundary:
#

# %%
plt.figure(figsize=(9, 4.5))
sns.histplot(blend_prob, bins=50, kde=True, color="teal")
plt.axvline(0.5, color="red", ls="--", label="decision boundary")
plt.axvspan(0.40, 0.60, color="orange", alpha=0.2, label="low-confidence zone")
plt.xlabel("mean predicted P(drop)")
plt.title("Prediction diagnostic — mean boosted-tree probability")
plt.legend()
plt.tight_layout()
plt.show()

uncertain = ((blend_prob > 0.40) & (blend_prob < 0.60)).mean() * 100
print(f"share of holdout in the 0.40–0.60 low-confidence zone: {uncertain:.1f}%")

# %% [markdown]
# The probability diagnostic is separate from the submitted rank score. Cases in the 0.4–0.6 band are useful examples of orders where the representative models are less separated. To understand why a case falls into this low-confidence zone, we will perform a local SHAP explanation in the next section.
#

# %% [markdown]
# # 9. Interpretation with SHAP
#
# For model interpretation, we analyze our representative **XGBoost** model. Because computing game-theoretic Shapley values (SHAP) is computationally expensive on the full dataset, we use a representative validation sample of 10,000 rows to ensure notebook execution remains efficient. SHAP attributes each prediction to its features, providing both global feature importance and individual observation explanations.
#

# %%
shap_model = get_xgb()
shap_model.fit(Xtr_t, y_tr)
print(
    f"XGBoost+time chrono AUC: {roc_auc_score(y_va, shap_model.predict_proba(Xva_t)[:, 1]):.4f}"
)

X_shap = Xva_t.sample(min(10000, len(Xva_t)), random_state=SEED)
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
plt.title("SHAP summary (beeswarm) — XGBoost+time")
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
# **Reading the SHAP importance.** The drivers line up with the EDA: `Payment_Terms`, the **time index** (`days_since_epoch`) and seasonality, the **frequency-encoded IDs** (`Agent_ID_freq`, `Company_ID_freq`), registration timing, and the engineered **history/ratio** features. The prominence of the time index is consistent with the boosted-tree ablation in Section 7.5: this representative model uses _when_ an order occurs to score the future window.
#
# ### The `Payment_Terms` leakage plausibility assessment
#
# EDA flagged prepaid-non-refundable as suspiciously strong. SHAP confirms it is influential, but feature importance cannot answer the timing question by itself. We therefore run one small sensitivity check: refit representative XGBoost without `Payment_Terms` and compare chronological AUC. This measures how dependent the model is on the field; the operational logging timestamp remains a separate data-governance check.
#

# %%
pred_xgb_no_payment = fit_predict(
    "xgb",
    Xtr_t.drop(columns=["Payment_Terms"]),
    y_tr,
    Xva_t.drop(columns=["Payment_Terms"]),
)
payment_check = pd.DataFrame({
    "model": ["XGBoost + time", "XGBoost + time, no Payment_Terms"],
    "chrono_AUC": [
        roc_auc_score(y_va, pred_t["xgb"]),
        roc_auc_score(y_va, pred_xgb_no_payment),
    ],
})
payment_check["delta_vs_with_payment"] = (
    payment_check["chrono_AUC"] - payment_check.loc[0, "chrono_AUC"]
)
display(payment_check)

# %% [markdown]
# The model remains strong without `Payment_Terms`, so the field is useful but not the sole source of performance. Our working assumption is that payment terms are set _at registration_ (before cancellation), making them a plausible early risk signal. Before production use, the data owner should audit the exact logging path and confirm that the field is not populated or overwritten after cancellation.
#

# %% [markdown]
# ## 9.2 Dependence view for the top signal
#

# %%
top_feat = (
    importance.loc[importance["feature"] != "Payment_Terms", "feature"].iloc[0]
    if importance["feature"].iloc[0] == "Payment_Terms"
    else importance["feature"].iloc[0]
)


def plot_shap_dependence_readable(feature, max_categories=15):
    col_idx = list(X_shap.columns).index(feature)
    values = X_shap[feature]
    is_categorical = (
        str(values.dtype) == "category"
        or values.dtype == "object"
        or values.nunique(dropna=False) <= max_categories
    )

    if not is_categorical:
        shap.dependence_plot(
            feature, shap_values, X_shap, interaction_index=None, show=False
        )
        plt.title(f"SHAP dependence — {feature}")
        plt.tight_layout()
        plt.show()
        return

    labels = values.astype("string").fillna("missing")
    keep = labels.value_counts().head(max_categories).index
    grouped = pd.DataFrame({
        "level": labels.where(labels.isin(keep), "other"),
        "shap": shap_values[:, col_idx],
    })
    summary = (
        grouped
        .groupby("level", observed=True)
        .agg(mean_shap=("shap", "mean"), n=("shap", "size"))
        .sort_values("mean_shap")
    )

    fig_h = max(4, min(7, 0.32 * len(summary) + 1.2))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.barplot(
        data=summary.reset_index(),
        y="level",
        x="mean_shap",
        color="#4c72b0",
        ax=ax,
    )
    ax.axvline(0, color="black", lw=1, alpha=0.5)
    ax.set_title(f"Mean SHAP by {feature} level (top {max_categories} + other)")
    ax.set_xlabel("mean SHAP contribution")
    ax.set_ylabel(feature)
    plt.tight_layout()
    plt.show()


try:
    plot_shap_dependence_readable(top_feat)
except Exception as e:
    print(f"(dependence view skipped for {top_feat}: {e})")

# %% [markdown]
# ## 9.3 Explaining a single low-confidence order
#
# To answer "_how_ does the model handle uncertain observations?", we pick one illustrative XGBoost sample case near P(drop) ≈ 0.5 and decompose its prediction. The waterfall shows which features pushed the score up vs down.
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
# For this illustrative borderline order, the positive and negative contributions nearly balance in the representative XGBoost model. In practice, cases like this are good candidates for human follow-up because the model score is near the decision boundary.
#

# %% [markdown]
# # 10. Final model retraining and submission generation
#
# We train our final ensembled boosters (XGBoost, LightGBM, CatBoost) on the complete training dataset.
# The predictions on the test set are rank-averaged to generate the final submission probabilities
# in the required `[Client_ID, Drop_Probability]` format.
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

scored_path = Path("data/Group_27_Submission.csv")
if scored_path.exists():
    scored_submission = pd.read_csv(scored_path)
    comparison = scored_submission.merge(
        submission,
        on="Client_ID",
        suffixes=("_scored_file", "_rebuilt"),
        validate="one_to_one",
    )
    diff = (
        comparison["Drop_Probability_scored_file"]
        - comparison["Drop_Probability_rebuilt"]
    ).abs()
    submission_match = pd.DataFrame({
        "rows_compared": [len(comparison)],
        "max_abs_diff": [diff.max()],
        "mean_abs_diff": [diff.mean()],
        "spearman_corr": [
            comparison["Drop_Probability_scored_file"].corr(
                comparison["Drop_Probability_rebuilt"], method="spearman"
            )
        ],
    })
    display(submission_match)
    if diff.max() < 1e-12:
        print("rebuilt predictions match the scored file exactly.")
    else:
        print(
            "rebuilt predictions are not byte-identical to the scored file; "
            "use the stored scored CSV as the official leaderboard record."
        )
else:
    print(f"{scored_path} not found; skipped scored-file comparison.")

# %% [markdown]
# The output has the required schema (`Client_ID`, `Drop_Probability`), one row per test order, with rank-average risk scores spread across `[0, 1]`.
#

# %% [markdown]
# # 11. Conclusions & Executive Summary
#
# **The Problem.** Predict the probability a B2B course registration is cancelled, so Nova Academy can stop sinking cost into orders that will fall through.
#
# **The Process.** A CRISP-DM-style pass: EDA → cleaning → feature engineering → chronologically-validated modelling → evaluation → SHAP interpretation.
#
# **Main Takeaway.** _The test set is the future._ Train ends where test begins, and the drop rate drifts over time (also shown by adversarial validation, AUC ≫ 0.5). This requires treating the task as a forecasting problem and using a **chronological holdout**, not a random split, for model decisions.
#
# **What Worked.**
#
# 1. **Time-aware validation** — every decision judged on a future window.
# 2. **Thorough categorical cleaning** + **label-free frequency encoding** — dirty high-cardinality IDs became usable without a one-hot explosion (our answer to the curse of dimensionality).
# 3. **Domain-driven features** — composition/ratio/history features and a **linear time index** that lets trees score future rows in the latest regime.
# 4. **A rank-average blend** of LightGBM + XGBoost + CatBoost, three implementations of the gradient-boosted tree family, which had the best holdout AUC in this comparison.
#
# **Key findings for the business.** The most useful signals are payment terms (prepaid-non-refundable is high-risk — worth a process review), the registration channel and enrolment type, agent/company identity and the presence of a company id, how early the group registered, and pre-course engagement (support tickets ⇒ commitment).
#
# **Submission Performance.** The final rank-average blend achieved a test ROC-AUC score of **0.889314**, safely exceeding the project's threshold of 0.70.
#
# **Future Work & Extensions**
#
# - Rolling multi-fold time-series CV for tuning (vs the single chrono holdout).
# - Careful leave-one-out / target encoding of `Agent_ID` / `Company_ID` (fit inside CV folds to prevent leakage).
# - Confirm the `Payment_Terms` signal with the data owner before leaning on it in production.
# - Probability **calibration** (isotonic/Platt) if the business needs the scores to read as true probabilities rather than just a good ranking.
#
