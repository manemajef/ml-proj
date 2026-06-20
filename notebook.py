# ---
# jupyter:
#   jupytext:
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
# # ML Project Notebook
#
# Rotem David Semah
#

# %%
# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

TARGET_NAME = "Dropped_Course"

data = pd.read_csv("data/Train_Data.csv")
official_test_data = pd.read_csv("data/Test_Data_No_Target.csv")


# %% [markdown]
# ### Helper functions
#


# %%
def plot_dropout_by_category(
    df: pd.DataFrame,
    col: str,
    target: str = TARGET_NAME,
    min_count: int = 50,
    top_n: int = 10,
    figsize=(9, 5),
):
    stats = df.groupby(col, dropna=False)[target].agg(dropout_rate="mean", count="size")

    stats = (
        stats[stats["count"] >= min_count]
        .sort_values("count", ascending=False)
        .head(top_n)
        .sort_values("dropout_rate", ascending=True)
    )

    labels = [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(labels, stats["dropout_rate"] * 100)

    overall = df[target].mean() * 100
    ax.axvline(
        overall,
        linestyle="--",
        linewidth=1.5,
        label=f"Dataset mean ({overall:.1f}%)",
    )

    ax.set_xlabel("Dropout rate (%)")
    ax.set_title(f"Dropout rate by {col}")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return stats


def plot_dropout_by_numeric_bins(
    df: pd.DataFrame,
    col: str,
    target: str = TARGET_NAME,
    bins=8,
    figsize=(9, 4),
):
    tmp = df[[col, target]].dropna().copy()

    tmp["bin"] = pd.qcut(
        tmp[col],
        q=bins,
        duplicates="drop",
    )

    stats = tmp.groupby("bin", observed=True)[target].agg(
        dropout_rate="mean", count="size"
    )

    fig, ax = plt.subplots(figsize=figsize)

    stats["dropout_rate"].mul(100).plot.bar(ax=ax)

    overall = df[target].mean() * 100
    ax.axhline(
        overall,
        linestyle="--",
        linewidth=1.5,
        label=f"Dataset mean ({overall:.1f}%)",
    )

    ax.set_ylabel("Dropout rate (%)")
    ax.set_title(f"Dropout rate by {col} bins")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return stats


def plot_dropout_over_time(
    df: pd.DataFrame,
    date_col: str = "Course_Start_Date",
    target: str = TARGET_NAME,
    freq: str = "M",
    figsize=(12, 4),
):
    tmp = df[[date_col, target]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    stats = (
        tmp
        .groupby(tmp[date_col].dt.to_period(freq))[target]
        .agg(dropout_rate="mean", count="size")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    stats["dropout_rate"].mul(100).plot(
        ax=ax,
        marker="o",
    )

    overall = df[target].mean() * 100
    ax.axhline(
        overall,
        linestyle="--",
        linewidth=1.5,
        label=f"Dataset mean ({overall:.1f}%)",
    )

    ax.set_ylabel("Dropout rate (%)")
    ax.set_title("Dropout rate over time")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return stats


def br():
    print()


# %% [markdown]
# ## 1. EDA
#

# %%
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
print("data shape: ", data.shape)
data.head()

# %% [markdown]
# Split the labeled file into a development training set and a validation set. The separate `Test_Data_No_Target.csv` file is kept as the final prediction set.
#

# %%
train_data, valid_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data[TARGET_NAME],
)

print(f"train_data shape: {train_data.shape}")
print(f"valid_data shape: {valid_data.shape}")
train_data.info()


# %% [markdown]
# I use `df` as shorthand for the development training split during EDA. Target-based analysis is based on this labeled split. The official test file has no `Dropped_Course` label, so it is used only for schema and missingness checks.
#

# %%
df = train_data

# %% [markdown]
# ### Dataset Scope And Target Balance
#
# Before looking at individual features, I check the target distribution and compare missing-value patterns between the labeled data and the official test file.
#

# %%
target_balance = df[TARGET_NAME].value_counts().sort_index().to_frame("count")
target_balance["rate_%"] = (
    df[TARGET_NAME].value_counts(normalize=True).sort_index().mul(100).round(1)
)
display(target_balance)

missing_compare = pd.DataFrame({
    "train_missing_%": df.isna().mean().mul(100).round(2),
    "official_test_missing_%": official_test_data.isna().mean().mul(100).round(2),
    "train_missing_count": df.isna().sum(),
    "official_test_missing_count": official_test_data.isna().sum(),
})

missing_compare = missing_compare.query(
    "train_missing_count > 0 or official_test_missing_count > 0"
).sort_values("train_missing_%", ascending=False)

display(missing_compare)


# %% [markdown]
# The target is not severely imbalanced: roughly 59% of the training examples are not dropped and 41% are dropped. This supports using AUC as the main evaluation metric.
#
# The official test file has similar missingness patterns to the labeled data. The missing-value strategy therefore needs to transform all rows consistently in both validation and final prediction data.
#

# %% [markdown]
# #### Missingness Versus Target
#
# For columns with meaningful missingness, I also check whether the fact that a value is missing is itself related to dropout. This helps decide whether to add missingness indicators.
#

# %%
missingness_target_cols = [
    "Agent_ID",
    "Company_ID",
    "Registration_Days_Before",
    "Physical_Course_Kits",
    "Daily_Tuition_Cost",
    "Requested_Lab_Config",
    "Payment_Terms",
]

missingness_target_rows = []
for col in missingness_target_cols:
    stats = (
        df
        .assign(is_missing=df[col].isna())
        .groupby("is_missing")[TARGET_NAME]
        .agg(count="size", dropout_rate="mean")
        .reset_index()
    )
    stats["column"] = col
    missingness_target_rows.append(stats)

missingness_target_summary = pd.concat(missingness_target_rows, ignore_index=True)
missingness_target_summary["dropout_rate_%"] = (
    missingness_target_summary["dropout_rate"] * 100
).round(1)

missingness_target_summary[
    [
        "column",
        "is_missing",
        "count",
        "dropout_rate_%",
    ]
]


# %% [markdown]
# Missingness is informative for some columns. Missing `Company_ID` is associated with a much higher dropout rate, which supports adding a company-presence indicator. Missing `Agent_ID` also has a different dropout profile. For `Registration_Days_Before` and `Physical_Course_Kits`, missingness itself is weaker, but missingness indicators are still useful because they distinguish observed values from imputed values.
#

# %% [markdown]
# ### Checking Cat cols
#

# %% [markdown]
# First, inspect the non-numeric columns: missing values, cardinality, and the most frequent raw categories.
#

# %%
cat_cols = df.select_dtypes(include=["object"]).columns
print("Number of none numeric cols: ", len(cat_cols))
print("The % of none numeric cols: ", 100 * len(cat_cols) / (df.shape[1] - 1))


def get_cat_smr(df: pd.DataFrame, cat_cols: pd.DataFrame.columns):
    cat_rows = []
    for col in cat_cols:
        vc = (df[col].value_counts(dropna=False, normalize=True)).mul(100).round(1)
        top_8 = [f"{idx} ({pct:.1f}%)" for idx, pct in vc.head(8).items()]
        cat_rows.append({
            "column": col,
            "missing_%": round(df[col].isna().mean() * 100, 1),
            "missing_count": df[col].isna().sum(),
            "unique_count": df[col].nunique(dropna=True),
            "top_8_cat": top_8,
        })
    return pd.DataFrame(cat_rows).sort_values("missing_count", ascending=False)


get_cat_smr(df, cat_cols)

# %% [markdown]
# #### Categorical Data Quality
#
# The raw categorical values contain artificial noise: casing differences, punctuation inside labels, spacing issues, and placeholder strings such as `unknown` or `?`. For example, a value like `blu#e` should map back to the same category as `blue`. I normalize these values before deeper categorical analysis.
#

# %%
import re

COMMON_NANS = {
    "nan",
    "none",
    "na",
    "n/a",
    "null",
    "",
    "unknown",
    "unknonwn",
    "?",
    "-",
    "--",
    ".",
}


def normalize_col(df: pd.DataFrame, col):
    s = df[col].astype("string")
    s = (
        s.str
        .strip()
        .str.lower()
        .str.replace(r"\band\b", "&", regex=True)
        .str.replace(r"[^a-z0-9&() ]+", "", regex=True)  # removes # ! * ? etc
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    s = s.mask(s.isin(COMMON_NANS), np.nan)
    df[col] = s


def normalize_cat_cols(df: pd.DataFrame, cat_cols):

    df = df.copy()

    for col in cat_cols:
        s = df[col].astype("string")

        s = (
            s.str
            .strip()
            .str.lower()
            .str.replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() ]+", "", regex=True)  # removes # ! * ? etc
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        s = s.mask(s.isin(COMMON_NANS), np.nan)

        df[col] = s

    return df


cat_normed_data = normalize_cat_cols(df, cat_cols)
get_cat_smr(cat_normed_data, cat_cols)


# %% [markdown]
# After normalization, duplicate category variants collapse into cleaner groups.
#

# %%
import math


normed_catted_df = normalize_cat_cols(df, cat_cols)

plot_cols = [c for c in cat_cols if c != "Course_Start_Date"]

ncols = 2
nrows = math.ceil(len(plot_cols) / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(12, 4 * nrows),
)

axes = axes.flatten()


def get_stats_for_top_freq_cats(
    df: pd.DataFrame,
    col: str,
    target: str = TARGET_NAME,
    min_count: int = 20,
    head: int = 8,
    sort_by: str = "dropout_rate",
    ascending: bool = False,
):
    """
    Select the top `head` categories by frequency, then sort that selected set for display.
    """
    stats = df.groupby(col, dropna=False)[target].agg(
        dropout_rate="mean",
        count="size",
    )

    stats = stats[stats["count"] >= min_count]
    stats = stats.sort_values("count", ascending=False).head(head)
    stats = stats.sort_values(sort_by, ascending=ascending)
    return stats


overall_dropout = normed_catted_df[TARGET_NAME].mean() * 100

for ax, col in zip(axes, plot_cols):
    stats = get_stats_for_top_freq_cats(
        normed_catted_df,
        col,
        ascending=True,
    )  # reverse sort for the plot
    frequency = stats["count"] / len(normed_catted_df) * 100
    dropout = stats["dropout_rate"] * 100

    y = np.arange(len(stats))
    h = 0.4

    ax.barh(
        y + h / 2,
        frequency,
        height=h,
        label="Frequency %",
    )

    ax.barh(
        y - h / 2,
        dropout,
        height=h,
        label="Dropout %",
    )

    ax.axvline(
        overall_dropout,
        linestyle="--",
        linewidth=1.5,
        color="tab:orange",
        label=f"Dataset dropout mean ({overall_dropout:.1f}%)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(stats.index, fontsize=6)
    ax.set_title(col)

    if ax is axes[0]:
        ax.legend()


# %% [markdown]
#

# %% [markdown]
# Now I check whether the cleaned categorical values are actually related to dropout.
#


# %%
def show_cat_rate_table():
    cat_rate_rows = []

    for col in [
        col
        for col in cat_cols
        if col not in ["Lanyard_Color", "Welcome_Gift_Type", "Course_Start_Date"]
    ]:
        stats = get_stats_for_top_freq_cats(
            normed_catted_df,
            col,
            head=5,
            sort_by="count",
            ascending=False,
        )

        row = {"column": col}

        for i, (cat_value, cat_stats) in enumerate(stats.iterrows(), start=1):
            dropout_rate = cat_stats["dropout_rate"] * 100
            count = int(cat_stats["count"])

            row[f"top_{i}"] = f"{cat_value}: drop%={dropout_rate:.1f}, count={count}"

        cat_rate_rows.append(row)

    return pd.DataFrame(cat_rate_rows)


show_cat_rate_table()


# %% [markdown]
# The compact table above is a screening view: for each categorical feature it keeps the frequent categories and reports their dropout rate. This helps identify which categorical variables deserve cleaner final plots. `Lanyard_Color` and `Welcome_Gift_Type` show weak business patterns, while `Payment_Terms`, `Client_Category`, `Submission_Source`, `Enrollment_Type`, and `Agent_ID` show meaningful differences.
#

# %% [markdown]
# **Payment terms signal**
#
# `Payment_Terms` is a very strong categorical signal: prepaid nonrefundable registrations have an unusually high dropout rate compared with pay-upon-start registrations. This feature should be retained and checked again during model evaluation because it may dominate the model.
#

# %%
normed_catted_df.groupby("Payment_Terms")["Dropped_Course"].agg(
    count="size",
    mean="mean",
)

# %% [markdown]
# **Course start date pattern**
#

# %%
plot_dropout_over_time(df)


# %% [markdown]
# #### Categorical Conclusions
#
# The categorical EDA reveals both data-quality issues and predictive patterns. Category normalization is therefore part of data cleaning, and the same normalization should be reused later for validation and final prediction data.
#
# Key conclusions:
#
# - `Payment_Terms` is the strongest categorical signal.
# - `Client_Category`, `Submission_Source`, and `Enrollment_Type` show meaningful differences in dropout rates and should be encoded.
# - `Lanyard_Color` and `Welcome_Gift_Type` are candidates for removal because they do not show a useful relationship with dropout.
# - `Course_Start_Date` shows a clear time pattern. It should be represented through date-derived features or careful categorical handling.
# - Categorical missing values should be encoded explicitly as a `missing` category.
#

# %% [markdown]
# ### Numeric cols
#

# %% [markdown]
# First, we have to separate numeric features into three subsets:
#
# - the target
# - true numeric features
# - identifier numbers, such as client ID or agent ID, etc., which shouldn't really be treated as a number.
#

# %%
all_num_cols = df.select_dtypes(include=["int64", "float64"]).columns
IDE_COLNAMES = ["Agent_ID", "Company_ID", "Client_ID"]
TARGET = "Dropped_Course"
ide_cols = [col for col in all_num_cols if col in IDE_COLNAMES]

num_cols = [col for col in all_num_cols if col not in IDE_COLNAMES and col != TARGET]

# %% [markdown]
# #### Identifier columns
#

# %% [markdown]
# Analyze `Agent_ID` and `Company_ID` as categorical identifiers.
#

# %%
get_cat_smr(df, ide_cols)

# %%
df["Client_ID"].nunique() == len(df)

# %% [markdown]
# `Client_ID` is unique per row, so it is not useful as a predictive feature. `Agent_ID` and `Company_ID` are numeric identifiers, but their values represent categories rather than quantities, so they are analyzed as categorical variables.
#
# `Company_ID` is missing for most rows. That missingness is meaningful because it separates registrations made through a company from registrations without a company identifier.
#

# %%
ide_cols = [col for col in ide_cols if col != "Client_ID"]


# %%
def plot_id_target_rate(
    df: pd.DataFrame,
    col: str,
    target: str = TARGET_NAME,
    min_count: int = 100,
    top_n: int = 15,
    figsize=(10, 8),
):
    return plot_dropout_by_category(
        df,
        col,
        target=target,
        min_count=min_count,
        top_n=top_n,
        figsize=figsize,
    )


# %%
plot_id_target_rate(df, "Agent_ID")

# %% [markdown]
# **Agent ID signal**
#
# Dropout rates differ substantially between frequent agents. This indicates that `Agent_ID` contains useful categorical information.
#

# %%
plot_id_target_rate(
    df,
    "Company_ID",
    min_count=50,
    top_n=10,
)

# %% [markdown]
# Registrations with a company identifier have substantially lower dropout than registrations without one. This supports creating a `has_company_id` feature and treating `Company_ID` as a categorical feature only when enough observations are available.
#

# %%
df.groupby(df.Company_ID.notna()).Dropped_Course.agg(count="size", drop_rate="mean")

# %% [markdown]
# The company-presence comparison confirms that the registration path is meaningful: rows with a company identifier have lower dropout than rows without one.
#

# %% [markdown]
# #### Numeric Columns
#

# %%
CALC_PLOTS = False


def plot_numeric_dist(df, col, target=TARGET_NAME):
    if col == target:
        return

    s = df[col]

    print(f"{col}")
    print(f"missing: {s.isna().sum()} ({s.isna().mean() * 100:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df, x=target, y=col, ax=axes[0])
    axes[0].set_title("By target")

    sns.histplot(data=df, x=col, hue=target, bins=50, ax=axes[1])
    axes[1].set_title("Distribution")

    plt.tight_layout()
    plt.show()


if CALC_PLOTS:
    for col in num_cols:
        plot_numeric_dist(df, col)


# %%
def get_num_smr(
    df: pd.DataFrame,
    cols,
    target="Dropped_Course",
):
    rows = []

    for col in cols:
        s = df[col]

        rows.append({
            "column": col,
            "missing_%": round(s.isna().mean() * 100, 1),
            "missing_count": s.isna().sum(),
            "corr_target": round(
                s.corr(df[target]),
                3,
            ),
            "mean": round(s.mean(), 2),
            "median": round(s.median(), 2),
            "std": round(s.std(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
            "q25": round(s.quantile(0.25), 2),
            "q75": round(s.quantile(0.75), 2),
            "skew": round(s.skew(), 2),
        })

    return pd.DataFrame(rows).sort_values(
        "corr_target",
        key=abs,
        ascending=False,
    )


get_num_smr(df, num_cols)

# %% [markdown]
# **Numeric missing values**
#
# The numeric summary table gives the first pass: missingness, correlation with the target, central tendency, spread, and extreme values.
#
# Missing-value decisions:
#
# - Most numeric columns: use median imputation fitted on the development training split. This is stable and easy to justify.
# - Important missing columns: add missingness indicators, because missingness itself is informative for several features.
# - `Daily_Tuition_Cost`: use median imputation and keep a missingness indicator. The missing rate is very low, so this is simpler and easier to reuse safely than a separate model-based imputer.
# - Categorical columns: fill missing values with an explicit `missing` category after normalization.
#
# The CRISP-DM lecture emphasizes sklearn preprocessing tools. I therefore fit the imputers on the development training split and reuse the fitted objects for validation and final prediction data.
#

# %%
sus_cols = [
    "Waiting_List_Days",
    "Prev_Course_Attended",
    "Registration_Days_Before",
    "Practical_Hours",
    "Students_Count",
    "Daily_Tuition_Cost",
]
for col in sus_cols:
    print(col)
    print(", ".join(map(str, df[col].nlargest(20).unique())))
    br()


# %% [markdown]
# `Prev_Course_Attended` has high values, but they form a smooth tail and remain plausible for recurring organizations. `Daily_Tuition_Cost = 5400` is a suspicious high value and should be revisited in the outlier-analysis section. `Students_Count = 9999` and extreme `Practical_Hours` values look like placeholder or entry errors.
#

# %%
sus_cols = [
    col for col in sus_cols if col not in ["Prev_Course_Attended", "Daily_Tuition_Cost"]
]
for col in sus_cols:
    print(f"\n{col}")
    print(df[col].quantile([0.9999, 0.999, 0.995, 0.99, 0.95, 0.9]))
    br()


# %% [markdown]
# **Outlier candidates**
#
# The clearest invalid or suspicious values are:
#
# - `Students_Count`: values above the 99.9th percentile jump from normal small counts to `9999`.
# - `Practical_Hours`: negative values and very large values such as `5000`/`10000` are not plausible course-hour values.
#
# `Waiting_List_Days` and `Registration_Days_Before` have long tails, but those values can represent real early registrations or long waiting periods, so they are retained at this stage.
#


# %%
def make_capped_copy(df: pd.DataFrame, cols: list[str], q: float = 0.999):
    capped_df = df.copy()
    cap_rows = []

    for col in cols:
        cap = capped_df[col].quantile(q)
        n_capped = (capped_df[col] > cap).sum()
        cap_rows.append({"column": col, "cap": cap, "values_set_missing": n_capped})
        capped_df[col] = capped_df[col].where(
            capped_df[col] <= cap,
            pd.NA,
        )

    return capped_df, pd.DataFrame(cap_rows)


df_capped, cap_summary = make_capped_copy(df, ["Students_Count", "Practical_Hours"])
display(cap_summary)


# %% [markdown]
# **Checking the Corr with target after capping**:
#

# %%
target_corr = (
    df_capped[num_cols]
    .corrwith(df_capped["Dropped_Course"])
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

display(target_corr)

# %%
corr_matrix = df_capped[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)


# %%
CORR_THRESH = 0.2

corr_pairs = (
    corr_matrix
    .where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1,
        ).astype(bool)
    )
    .stack()
    .reset_index()
)

corr_pairs.columns = [
    "feature_1",
    "feature_2",
    "corr",
]

hi_corr_pairs = corr_pairs[corr_pairs["corr"].abs() >= CORR_THRESH].sort_values(
    "corr",
    key=abs,
    ascending=False,
)
hi_corr_pairs

# %% [markdown]
# ### Final EDA Plots
#
# These are the curated EDA plots kept for the report. Earlier tables are used for screening; these plots communicate the main conclusions more clearly.
#

# %%
payment_terms_stats = plot_dropout_by_category(
    normed_catted_df, "Payment_Terms", min_count=20, top_n=5
)
client_category_stats = plot_dropout_by_category(
    normed_catted_df, "Client_Category", min_count=100, top_n=8
)
submission_source_stats = plot_dropout_by_category(
    normed_catted_df, "Submission_Source", min_count=100, top_n=6
)
enrollment_type_stats = plot_dropout_by_category(
    normed_catted_df, "Enrollment_Type", min_count=100, top_n=6
)
agent_stats = plot_dropout_by_category(df, "Agent_ID", min_count=100, top_n=15)
registration_days_stats = plot_dropout_by_numeric_bins(
    df_capped, "Registration_Days_Before", bins=8
)
support_tickets_stats = plot_dropout_by_numeric_bins(
    df_capped, "Pre_Course_Supports_Tickets", bins=6
)
course_start_stats = plot_dropout_over_time(df)


# %% [markdown]
# #### Final Plot Interpretation
#
# - `Payment_Terms`: prepaid nonrefundable registrations have a much higher dropout rate than pay-upon-start registrations, making this one of the strongest categorical signals.
# - `Client_Category`: dropout differs strongly by segment. Big tech and multinational clients are above the dataset mean, while nonprofit/edtech, fintech/banking, and industrial tech/IoT are lower.
# - `Submission_Source`: direct website and dedicated-sales registrations are lower risk than B2B platform/reseller traffic.
# - `Enrollment_Type`: organizational arrangements and affiliated admissions are lower risk than general admission and contractual agreement.
# - `Agent_ID`: frequent agents have very different dropout rates, so the agent identifier carries useful categorical information.
# - `Registration_Days_Before`: dropout rises as registration happens further before the course, especially in the highest bin.
# - `Pre_Course_Supports_Tickets`: rows with more support tickets have lower dropout in this data, suggesting that pre-course engagement may be protective.
# - `Course_Start_Date`: dropout changes over time, suggesting a period or seasonality effect.
#
# Overall EDA conclusion: the strongest visible signals are payment terms, registration timing, agent/company registration path, client segment, source/enrollment channel, and support-ticket engagement.
#

# %% [markdown]
# ### Missing Value Completion
#
# The missing-value completion step follows the EDA conclusions. Numeric columns are filled with train-fitted medians, categorical columns receive an explicit `missing` category, and important missing columns receive missingness indicators. I use a simple median rule for `Daily_Tuition_Cost` as well: the missing rate is very low, and the median rule keeps the preprocessing easy to explain and reuse safely for validation and final prediction data.
#

# %%
ID_CATEGORICAL_COLS = ["Agent_ID", "Company_ID"]
IMPORTANT_MISSING_FLAGS = [
    "Agent_ID",
    "Registration_Days_Before",
    "Physical_Course_Kits",
    "Daily_Tuition_Cost",
    "Requested_Lab_Config",
    "Payment_Terms",
]


def complete_missing_values(
    data: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = data.copy()
    if train_df is None:
        train_df = data

    num_cols = [
        col
        for col in train_df.select_dtypes(include=["int64", "float64"]).columns
        if col not in {TARGET_NAME, "Client_ID", *ID_CATEGORICAL_COLS}
    ]

    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    num_medians = train_df[num_cols].median()

    if "Client_ID" in df.columns:
        df = df.drop(columns=["Client_ID"])

    if "Company_ID" in df.columns:
        df["has_company_id"] = df["Company_ID"].notna().astype(int)

    for col in IMPORTANT_MISSING_FLAGS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    present_numeric_cols = [col for col in num_cols if col in df.columns]
    df[present_numeric_cols] = df[present_numeric_cols].fillna(
        num_medians[present_numeric_cols]
    )

    present_categorical_cols = [col for col in cat_cols if col in df.columns]

    df = normalize_cat_cols(df, present_categorical_cols)

    df[present_categorical_cols] = (
        df[present_categorical_cols].fillna("missing").astype("object")
    )

    for col in ID_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .fillna("missing")
                .astype("object")
            )

    return df


# %% [markdown]
# ## 2. Outlier Analysis & Feature Engineering
#

# %% [markdown]
# ### Baseline Model Benchmark
#
# The benchmark separates feature engineering from generic preprocessing. Each
# `prepare_features_*` function creates a concrete modeling table. The benchmark then
# handles only generic steps: missing-value imputation, categorical encoding, numeric
# scaling, and model fitting. This makes the experiment readable: every added feature
# group can be compared with the same benchmark.
#


# %% [markdown]
# Utils:
#

# %%
DF = pd.DataFrame  # a shorter alias for typechecking

train_df = complete_missing_values(train_data, train_data)
val_df = complete_missing_values(valid_data, train_data)

test_df = complete_missing_values(
    pd.read_csv("data/Test_Data_No_Target.csv"), train_data
)


def split_xy(df: DF, target=TARGET_NAME):
    X, y = df.drop(columns=[target]), df[target]
    return X, y


def get_train_val_xy(train_df: DF, val_df: DF):
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    return X_train, y_train, X_val, y_val


def encode_cats(X_train, X_val):
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        feature_name_combiner=lambda col, val: f"ohe__{col}__{val}",
    ).set_output(transform="pandas")

    return (
        pd.concat(
            [
                X_train.drop(columns=cat_cols),
                enc.fit_transform(X_train[cat_cols]),
            ],
            axis=1,
        ),
        pd.concat(
            [
                X_val.drop(columns=cat_cols),
                enc.transform(X_val[cat_cols]),
            ],
            axis=1,
        ),
    )


# %% [markdown]
# **Scorring for 3 models**
#

# %%
from sklearn.preprocessing import StandardScaler


def get_lg_model():
    return LogisticRegression(
        max_iter=2000,
    )


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )


def get_xgb_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def get_score(train_df, val_df, model, scale=False):
    X_train, y_train, X_val, y_val = get_train_val_xy(train_df, val_df)
    X_train, X_val = encode_cats(X_train, X_val)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y_val, preds)


MODELS = {
    "Logistic Regression": {
        "model_getter": get_lg_model,
        "scale": True,
    },
    "Random Forest": {
        "model_getter": get_rf_model,
        "scale": False,
    },
    "XGBoost": {
        "model_getter": get_xgb_model,
        "scale": False,
    },
}


def benchmark_models(train_df, val_df):
    rows = []

    for name, cfg in MODELS.items():
        score = get_score(
            train_df,
            val_df,
            model=cfg["model_getter"](),
            scale=cfg["scale"],
        )

        rows.append({
            "model": name,
            "auc": score,
        })

    return pd.DataFrame(rows).sort_values("auc", ascending=False)


scores_0 = benchmark_models(train_df, val_df)
display(scores_0)


# %% [markdown]
# #### Conclusions
#
# All 3 models are doing surprinsgly well. I want to see what happens if i drop the start_date
#

# %% [markdown]
# **Concerns about `Course_Start_Date`**
#
# Course start date has a few problems. One is simply that we have too many categories, which bloats the dimensions. The second, and more concerning, issue is that the test data includes dates that are not present in the training set. The model might find correlations between specific dates and drop rates, but those correlations could relate to events outside the data—such as an economic crisis. Many factors could impact the results over time, so we shouldn’t expect the old dates to predict the new ones. Let’s see how much it matters and whether it’s safe to drop it.
#

# %%
train_no_date = train_df.drop(columns=["Course_Start_Date"])
val_no_date = val_df.drop(columns=["Course_Start_Date"])

scores_1 = benchmark_models(train_no_date, val_no_date)

score_compare = scores_0.merge(
    scores_1,
    on="model",
    suffixes=("_with_date", "_no_date"),
)

score_compare["diff"] = score_compare["auc_with_date"] - score_compare["auc_no_date"]
score_compare = score_compare.sort_values("diff", ascending=False)

display(score_compare)


# %% [markdown]
# **Im dropping it.**
#


# %%
train_df0, val_df0 = train_df, val_df
train_df, val_df = (
    train_df.drop(columns=["Course_Start_Date"]),
    val_df.drop(columns=["Course_Start_Date"]),
)


# %%
def get_encoded_xy(train_df, val_df):
    X_train, y_train, X_val, y_val = get_train_val_xy(train_df, val_df)
    X_train, X_val = encode_cats(X_train, X_val)
    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = get_encoded_xy(train_df, val_df)

rf = get_rf_model()

rf.fit(X_train, y_train)

preds = rf.predict_proba(X_val)[:, 1]
print("AUC:", roc_auc_score(y_val, preds))
rf_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)


rf_importance.head(30)
rf_importance.head(25).sort_values("importance").plot.barh(
    x="feature",
    y="importance",
    figsize=(10, 8),
    legend=False,
)


plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
display(rf_importance.head(70))

# %% [markdown]
# The results are concerning because the strongest feature here is clearly the suspicious one: non‑refundable payment. It seems either to be an error or something they flipped, but it just doesn’t make sense that there would be an almost 100 % correlation between that and the dropout rate. It just doesn’t make sense, and that is concerning. What happens if we drop that from the model? How badly will it perform afterward? The question is whether we can trust the same corruption or weird coincidence to also apply in the test data. I’m not sure. So it’s a big question what we do about it.
#

# %% [markdown]
# ## Dimension Reduction
#

# %%
X_train.shape

# %% [markdown]
# We have almost 600 features after `one-hot-encoding`. This is definitely unacceptable. So we’re going to fix it by treating:
#
# - company ID
# - agent ID
# - country.
#
# ### Main stradegy
#
# For agent ID and country ID, I’ll use the top n by coverage or samples.
# For `Company_ID`, I’ll drop the raw high-cardinality ID and keep only the
# existing `has_company_id` flag.
#

# %%
for label, col in [
    ("Agent_ID", train_df["Agent_ID"]),
    ("Origin_Country", train_df["Origin_Country"]),
]:
    counts = col.value_counts()
    print(f"\n=== {label} ===")
    print(counts.head(10))
    for thr in (150, 300, 500):
        kept = (counts >= thr).sum()
        cov = counts[counts >= thr].sum() / counts.sum()
        print(f"  >={thr:5d}: {kept:3d} kept, {cov:.1%} coverage")


# %% [markdown]
# It seems like the right one for `Agent_ID` is 300. As for `Origin_Country`, I see `cn` and `chn` probably both mean China, so first fix that and then re-evaluate.
#


# %%
def keep_by_min_count(col: pd.Series, min_count: int) -> set:
    counts = col.value_counts()
    return set(counts[counts >= min_count].index)


def describe_keep(col: pd.Series, keep: set, label: str):
    coverage = col.isin(keep).mean()  # share of non-missing rows kept
    print(
        f"{label}: kept {len(keep)} of {col.nunique()} "
        f"covering {coverage:.1%} of non-missing rows"
    )


AGENT_MIN_COUNT = 300  # clear cliff: 15 agents, 85% coverage
COUNTRY_MIN_COUNT = (
    300  # fat middle: 18 countries, 92% coverage — country signal is more distributed
)

agents_before = train_df["Agent_ID"]
agents_keep = keep_by_min_count(agents_before, AGENT_MIN_COUNT)
describe_keep(agents_before, agents_keep, "Agent_ID")

countries_before = train_df["Origin_Country"]
countries_keep = keep_by_min_count(countries_before, COUNTRY_MIN_COUNT)
describe_keep(countries_before, countries_keep, "Origin_Country")


# %%
company_counts = train_df["Company_ID"].value_counts()
print(company_counts.head(10))

# %% [markdown]
# So let's drop raw `Company_ID` completely. The lower-cardinality
# `has_company_id` flag was already created earlier and keeps the stable part of
# the signal.
#

# %%
COUNTRY_ALIASES = {"cn": "chn"}


def apply_ide_reduction(df: DF, train_df: DF) -> DF:
    df = df.copy()
    train_df = train_df.copy()

    df["Origin_Country"] = df["Origin_Country"].replace(COUNTRY_ALIASES)
    train_df["Origin_Country"] = train_df["Origin_Country"].replace(COUNTRY_ALIASES)

    countries_to_keep = keep_by_min_count(
        train_df["Origin_Country"], COUNTRY_MIN_COUNT
    ) | {"missing"}
    agents_to_keep = keep_by_min_count(train_df["Agent_ID"], AGENT_MIN_COUNT) | {
        "missing"
    }

    # collapse non-kept to "other"
    df["Agent_ID"] = df["Agent_ID"].where(df["Agent_ID"].isin(agents_to_keep), "other")
    df["Origin_Country"] = df["Origin_Country"].where(
        df["Origin_Country"].isin(countries_to_keep), "other"
    )

    df = df.drop(columns=["Company_ID"], errors="ignore")
    df = df.drop(columns=["Agent_ID_missing"], errors="ignore")

    return df


# %%
initial_train = train_df.copy()
train_df1 = train_df.copy()
val_df1 = val_df.copy()

train_reduced_df = apply_ide_reduction(train_df, initial_train)
val_reduced_df = apply_ide_reduction(val_df, initial_train)

X_train, y_train = split_xy(train_reduced_df, TARGET_NAME)
X_val, y_val = split_xy(val_reduced_df, TARGET_NAME)

X_train, X_val = encode_cats(X_train, X_val)
train_df, val_df = train_reduced_df, val_reduced_df
X_train.shape

# %% [markdown]
# Great so $\approx 400$ dimensitons eliminated.
#

# %%
benchmark_models(train_df, val_df)


# %% [markdown]
# #### Conclusions
#
# Logistic regression actualy used those dummies, while the tree based models are happy without them.
#
# The dim reduction helped very slightly to XBboost, but anyway ... its required
#

# %% [markdown]
# ### Feature Engeenring & Noise Reduction
#
# **Main Ideas**:
#
# - We're going to switch the assigned requested lab config with the received requested lab config Boolean. Since the preferences are what matters, and whether they have their preferences also matters, that's the only detail the models need to know.
#
# - We're going to drop the returning client dummy since this information already exists in previous course attendance. So, yeah, it's useless.
#
# - As we've seen in the EDA, the lanyard color and welcome‑gift type seem to have no correlation, and they have no business justification for actually mattering. So I'll drop them as noise reduction.
#


# %%
def apply_feature_eng(df: DF):
    df = df.copy()
    df["recived_requested_lab"] = (
        df['Requested_Lab_Config'] == df['Assigned_Lab_Config']
    ).astype(int)
    to_drop = [
        'Assigned_Lab_Config',
        'Lanyard_Color',
        'Requested_Lab_Config',
        'Returning_Client',
        'Welcome_Gift_Type',
    ]
    df = df.drop(columns=to_drop, errors="ignore")
    return df


# %% [markdown]
# ## Outliers
#
# we've already seen in the EDA that there are some outliers concerns. for now, I would use the simple capping suggested in EDA and test if it helps.
#

# %%
# def find_sus_columns(df, num_cols):
#     rows = []

#     for c in num_cols:
#         s = df[c].dropna()

#         if s.empty or s.nunique() <= 2:
#             continue

#         q99 = s.quantile(0.99)
#         q999 = s.quantile(0.999)
#         max_val = s.max()
#         min_val = s.min()

#         max_over_q99 = max_val / (q99 + 1e-9)
#         gap_999_99 = q999 - q99

#         reasons = []

#         if min_val < 0:
#             reasons.append("negative_values")

#         if q99 > 0 and max_over_q99 > 10:
#             reasons.append("huge_max_vs_q99")

#         if gap_999_99 > q99:
#             reasons.append("big_tail_jump")

#         if not reasons:
#             continue

#         rows.append({
#             "col": c,
#             "min": min_val,
#             "max": max_val,
#             "q99": q99,
#             "q999": q999,
#             "max_over_q99": max_over_q99,
#             "gap_999_99": gap_999_99,
#             "reason": ", ".join(reasons),
#         })


#     return pd.DataFrame(rows).sort_values("max_over_q99", ascending=False)
def find_sus_columns(df, num_cols, max_mult=10):
    sus = []
    for c in num_cols:
        s = df[c].dropna()
        q99 = s.quantile(0.99)
        iqr = s.quantile(0.75) - s.quantile(0.25)
        scale = max(q99, iqr, 1.0)

        reasons = []
        if s.min() < 0:
            reasons.append("negative")
        if s.max() > max_mult * scale:  # max sits absurdly far past the bulk
            reasons.append(f"max={s.max():g} vs q99={q99:g}")

        if reasons:
            sus.append({
                "col": c,
                "min": s.min(),
                "max": s.max(),
                "q99": q99,
                "why": ", ".join(reasons),
            })
    return pd.DataFrame(sus)


print("sus cols on train data")
display(find_sus_columns(train_df, num_cols))
br()
print("sus cols on test data")
display(find_sus_columns(test_df, num_cols))

# %%
print("train: ")
print(find_sus_columns(train_df, num_cols))
br()
print("test: ")
print(find_sus_columns(test_df, num_cols))

# %% [markdown]
# so we see that the test data dosnt have outliers that train has missed. lets plot the instresting ones.
#
# - Student count , Practical hours: Already spotted in EDA. Obvsius typos / intentional sabotage
# - Daily tuition cost : also spotted in EDA. has a single rediculusly high value in train data for the top 1.
#
# **Low risk ones**
#
# - Waiting List days, Prev Course attended: Dosnt "scream" fake, but need carfull inspection.
#   - I will test to see if `prev_dropout` is > then `prev_attended` in some rows, which is obiusly impossible.
#

# %%
impossible_rows_train = train_df[
    train_df["Prev_Course_Dropouts"] > train_df["Prev_Course_Attended"]
]
print("train condtrdictions: ", len(impossible_rows_train))

impossible_rows_test = test_df[
    test_df["Prev_Course_Dropouts"] > test_df["Prev_Course_Attended"]
]
print("test contredictions: ", len(impossible_rows_test))


# %%
def apply_capping(df: DF, show_plots: bool = False) -> DF:
    df = df.copy()
    caps = {
        "Students_Count": (None, 3),  # bulk maxes at 3; 9999 is an entry error
        "Practical_Hours": (0, 8),  # negative impossible; bulk ≤ ~8
        "Daily_Tuition_Cost": (None, 500),  # ~train q99.9 (268); 5400 is implausible
    }
    for col, (lo, hi) in caps.items():
        if col not in df.columns:
            continue

        before = df[col].dropna()
        df[col] = df[col].clip(lower=lo, upper=hi)

        if show_plots:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].hist(before, bins=50)
            ax[0].set_yscale("log")
            ax[0].set_title(f"{col} before (max={before.max():g})")
            ax[1].hist(df[col].dropna(), bins=50)
            ax[1].set_yscale("log")
            ax[1].set_title(f"{col} after (max={df[col].max():g})")
            plt.tight_layout()
            plt.show()

    return df


capped_train = apply_capping(train_df, show_plots=True)


# %%
# capped_tst = apply_capping(test_df, show_plots=True)

# %% [markdown]
# ### Final bench mark for part 2
#

# %%
PREPROCESSING_STEPS = [
    "split labeled data before fitting preprocessing statistics",
    "drop Client_ID from model features but keep it separately for submission",
    "add has_company_id before dropping raw Company_ID",
    "add missingness flags for selected informative missing columns",
    "fill numeric missing values with train-fitted medians",
    "normalize categorical strings and convert common NA tokens to missing",
    "strip .0 from Agent_ID and Company_ID and treat them as categories",
    "drop Course_Start_Date",
    "merge Origin_Country cn into chn",
    "collapse rare Agent_ID and Origin_Country values into other",
    "drop raw Company_ID and keep only has_company_id",
    "replace requested/assigned lab configs with recived_requested_lab",
    "drop noisy or redundant columns: Lanyard_Color, Returning_Client, Welcome_Gift_Type",
    "cap suspicious outliers in Students_Count, Practical_Hours, and Daily_Tuition_Cost",
    "one-hot encode categorical columns after preprocessing",
]


def apply_preprocessing(df: DF, train_reference: DF) -> DF:
    """
    Apply the notebook's selected preprocessing to a raw train/validation/test frame.

    `train_reference` must be the raw training data used to fit medians and
    category keep-lists. For validation, pass `train_data`; for final test
    predictions, pass the full labeled training file.
    """
    train_completed = complete_missing_values(train_reference, train_reference)
    df_completed = complete_missing_values(df, train_reference)

    train_completed = train_completed.drop(
        columns=["Course_Start_Date"], errors="ignore"
    )
    df_completed = df_completed.drop(columns=["Course_Start_Date"], errors="ignore")

    df_processed = apply_ide_reduction(df_completed, train_completed)
    df_processed = apply_feature_eng(df_processed)
    df_processed = apply_capping(df_processed)

    return df_processed


aplly_preprocessing = apply_preprocessing


def fit_predict_proba(
    train_processed: DF,
    predict_processed: DF,
    model,
    scale: bool = False,
):
    X_train, y_train = split_xy(train_processed, TARGET_NAME)
    X_predict = predict_processed.drop(columns=[TARGET_NAME], errors="ignore")

    X_train, X_predict = encode_cats(X_train, X_predict)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_predict = scaler.transform(X_predict)

    model.fit(X_train, y_train)
    return model.predict_proba(X_predict)[:, 1]


def predict_test(
    test_df: DF,
    train_raw: DF = data,
    output_path: str = "data/Group_XX_Submission.csv",
) -> pd.DataFrame:
    """
        hen finaly --- write the predictions for test data into `./data/Group_XX_Submission.csv` with according to source rules.

        choose the best model, or if the val show that a weighted avg of the 3 versions, use that.

        source requires the test pred to be somthing like this:
        1. קובץ פלט: קובץ זה נדרש להיות בפורמט CSV ולהכיל שתי עמודות, כאשר העמודה הראשונה תיקרא Client_ID והעמודה השנייה תיקרא Drop_Probability. מטרת הקובץ הינה להציג את תחזיות המודל עבור כל תצפית. כל שורה תציג Client_ID אשר נמצא בקובץ Test_Data_No_Target.csv ואת ההסתברות שהמודל נותן (ולא חיזוי “קשה” של 0 או 1) לביטול ההשתתפות של הקבוצה. שם הקובץ צריך להיות Group_XX_Submission.csv, כאשר XX מייצג את מספר הקבוצה. להלן דוגמה לקובץ זה:
    Client_ID	Drop_Probability
    62246	0.45774832
    43031	0.10221539
    26571	0.99805343
    77694	0.72571185
    22185	0.20942985
    54569	0.35561964
    64162	0.78605343

    """
    train_split, val_split = train_test_split(
        train_raw,
        test_size=0.2,
        random_state=42,
        stratify=train_raw[TARGET_NAME],
    )

    processed_train = apply_preprocessing(train_split, train_split)
    processed_val = apply_preprocessing(val_split, train_split)

    scores = []
    for model_name, cfg in MODELS.items():
        preds = fit_predict_proba(
            processed_train,
            processed_val,
            model=cfg["model_getter"](),
            scale=cfg["scale"],
        )
        _, y_val = split_xy(processed_val, TARGET_NAME)
        scores.append({
            "model": model_name,
            "auc": roc_auc_score(y_val, preds),
        })

    score_df = pd.DataFrame(scores).sort_values("auc", ascending=False)
    try:
        display(score_df)
    except NameError:
        print(score_df)

    best = score_df.iloc[0]
    best_cfg = MODELS[best["model"]]

    processed_full_train = apply_preprocessing(train_raw, train_raw)
    processed_test = apply_preprocessing(test_df, train_raw)

    test_preds = fit_predict_proba(
        processed_full_train,
        processed_test,
        model=best_cfg["model_getter"](),
        scale=best_cfg["scale"],
    )

    submission = pd.DataFrame({
        "Client_ID": test_df["Client_ID"],
        "Drop_Probability": test_preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"wrote {output_path} using {best['model']} (val AUC={best['auc']:.4f})")

    return submission


# %% [markdown]
#
