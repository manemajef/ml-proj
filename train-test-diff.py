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
# # Train/test distribution check
#
# This is adversarial validation: combine the labeled train rows and official
# test rows, add `IS_TEST`, remove fields that trivially reveal the source, then
# train tree models to classify whether each row came from the official test set.
#
# If AUC is near 0.5, train and test look similar after removing those fields.
# If AUC is high, the feature-importance table points to covariate shift.
#

# %%
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def show(obj):
    try:
        display(obj)
    except NameError:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)


# %%
TARGET_NAME = "Dropped_Course"
DATE_COL = "Course_Start_Date"
SOURCE_TARGET = "IS_TEST"
TRAIN_PATH = "data/Train_Data.csv"
TEST_PATH = "data/Test_Data_No_Target.csv"
RANDOM_STATE = 42

DROP_SOURCE_LEAK_COLS = [
    TARGET_NAME,
    DATE_COL,
    "Client_ID",
    "Unnamed: 0",
]


# %% [markdown]
# ## 01 Load and combine data
#

# %%
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train[SOURCE_TARGET] = False
test[SOURCE_TARGET] = True

combined = pd.concat([train, test], ignore_index=True, sort=False)

show(
    pd.DataFrame([
        {"data": "train", "rows": len(train), "columns": train.shape[1]},
        {"data": "test", "rows": len(test), "columns": test.shape[1]},
        {"data": "combined", "rows": len(combined), "columns": combined.shape[1]},
    ])
)


# %% [markdown]
# ## 02 Prepare features
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


def normalize_cat_cols(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    df = df.copy()

    for col in cat_cols:
        s = df[col].astype("string")
        s = (
            s.str
            .strip()
            .str.lower()
            .str.replace(r"\band\b", "&", regex=True)
            .str.replace(r"[^a-z0-9&() ]+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df[col] = s.mask(s.isin(COMMON_NANS), np.nan)

    return df


def prepare_source_features(df: pd.DataFrame):
    y = df[SOURCE_TARGET].astype(int)
    X = df.drop(columns=[SOURCE_TARGET, *DROP_SOURCE_LEAK_COLS], errors="ignore")

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    X = normalize_cat_cols(X, cat_cols)
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("missing").astype("object")

    return X, y, cat_cols


def encode_train_val(X_train: pd.DataFrame, X_val: pd.DataFrame):
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        feature_name_combiner=lambda col, val: f"ohe__{col}__{val}",
    ).set_output(transform="pandas")

    X_train_num = X_train.drop(columns=cat_cols).reset_index(drop=True)
    X_val_num = X_val.drop(columns=cat_cols).reset_index(drop=True)

    if not cat_cols:
        return X_train_num, X_val_num

    X_train_cat = enc.fit_transform(X_train[cat_cols]).reset_index(drop=True)
    X_val_cat = enc.transform(X_val[cat_cols]).reset_index(drop=True)

    return (
        pd.concat([X_train_num, X_train_cat], axis=1),
        pd.concat([X_val_num, X_val_cat], axis=1),
    )


X, y, cat_cols = prepare_source_features(combined)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)
X_train_enc, X_val_enc = encode_train_val(X_train, X_val)

print("encoded shape:", X_train_enc.shape, X_val_enc.shape)
print("test share:", y.mean().round(4))


# %% [markdown]
# ## 03 Train source classifiers
#

# %%
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def fit_source_models():
    rows = []
    fitted = {}

    for name, model in MODELS.items():
        model.fit(X_train_enc, y_train)
        preds = model.predict_proba(X_val_enc)[:, 1]
        rows.append({"model": name, "source_auc": roc_auc_score(y_val, preds)})
        fitted[name] = model

    return pd.DataFrame(rows).sort_values("source_auc", ascending=False), fitted


source_scores, fitted_models = fit_source_models()
show(source_scores)


# %% [markdown]
# ## 04 Feature importance
#

# %%
def original_feature_name(encoded_feature: str) -> str:
    if encoded_feature.startswith("ohe__"):
        return encoded_feature.split("__", 2)[1]
    return encoded_feature


def aggregate_importance(model, model_name: str, top_n: int = 25) -> pd.DataFrame:
    importances = pd.Series(model.feature_importances_, index=X_train_enc.columns)
    by_feature = (
        importances
        .groupby(original_feature_name)
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    by_feature.columns = ["feature", "importance"]
    by_feature.insert(0, "model", model_name)
    return by_feature


importance_tables = [
    aggregate_importance(model, name) for name, model in fitted_models.items()
]
importance = pd.concat(importance_tables, ignore_index=True)
show(importance)


# %% [markdown]
# ## 05 Value-level drift for top features
#
# This table is descriptive. It is meant to explain what changed, not to prove
# causality.
#

# %%
def describe_feature_shift(feature: str) -> pd.DataFrame:
    source = combined[[feature, SOURCE_TARGET]].copy()

    if pd.api.types.is_numeric_dtype(source[feature]):
        train_values = source.loc[~source[SOURCE_TARGET], feature]
        test_values = source.loc[source[SOURCE_TARGET], feature]
        return pd.DataFrame([
            {
                "feature": feature,
                "type": "numeric",
                "train_missing_pct": train_values.isna().mean(),
                "test_missing_pct": test_values.isna().mean(),
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "train_p50": train_values.median(),
                "test_p50": test_values.median(),
                "train_p90": train_values.quantile(0.9),
                "test_p90": test_values.quantile(0.9),
            }
        ])

    source = normalize_cat_cols(source, [feature])
    source[feature] = source[feature].fillna("missing")

    rows = []
    for value, grp in source.groupby(feature, dropna=False):
        rows.append({
            "feature": feature,
            "type": "categorical",
            "value": value,
            "train_pct": ((grp[SOURCE_TARGET] == False).sum())
            / (source[SOURCE_TARGET] == False).sum(),
            "test_pct": (grp[SOURCE_TARGET].sum()) / source[SOURCE_TARGET].sum(),
            "diff_pct": (grp[SOURCE_TARGET].sum()) / source[SOURCE_TARGET].sum()
            - ((grp[SOURCE_TARGET] == False).sum())
            / (source[SOURCE_TARGET] == False).sum(),
        })

    return (
        pd
        .DataFrame(rows)
        .assign(abs_diff=lambda df: df["diff_pct"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(8)
        .drop(columns=["abs_diff"])
    )


top_features = (
    importance
    .groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .head(8)
    .index
)

shift_summary = pd.concat(
    [describe_feature_shift(feature) for feature in top_features],
    ignore_index=True,
    sort=False,
)
show(shift_summary)

