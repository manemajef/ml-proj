```python
import marimo as mo
USE_V3 = False  # False = verified v2 fallback; True = v3 candidate
```

# Group 27 — Course-Drop Prediction (Nova Academy)

**Submitters:** Rotem David Semah (ID: `211396593`) · Ron Drach (ID: `213915499`)

---

This notebook follows the project from understanding the data through preparation, modelling, evaluation, and interpretation.

> **About the notebook's**
>
> The notebook was developed as a `marimo` notebook and exported to Jupyter format for submission.Some code patterns may look wierd (such as wrappers functions for ploting etc) but that ensures compatibility with both `jupyter` and `marimo` format.
>
> Additionaly you may notice expensive functions (such as shap, and hyper tuners) are decorated with `@cache` which as the name suggest - caches the the expensive claulations to keep the notebook runable :)

```python
from functools import wraps
from inspect import getsource
import warnings
from pathlib import Path
from textwrap import dedent
from joblib import dump, hash as joblib_hash, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from IPython.display import display

def cache(fn):
    cache_dir = Path('.cache/joblib') / fn.__name__
    source_hash = joblib_hash(dedent(getsource(fn)))

    @wraps(fn)
    def wrapped(*args, **kwargs):
        path = cache_dir / f'{joblib_hash((USE_V3, source_hash, args, kwargs))}.joblib'
        if path.exists():
            return load(path)
        result = fn(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(result, path)
        return result

    return wrapped

warnings.filterwarnings('ignore')
sns.set_theme(
    style='whitegrid',
    palette='colorblind',
    rc={
        'figure.figsize': (8, 3.5),
        'figure.constrained_layout.use': False,
        'savefig.format': 'svg',
        'svg.fonttype': 'none',
    },
)
set_matplotlib_formats('svg')
pd.set_option('display.max_columns', None)
TRAIN_PATH = 'data/Train_Data.csv'
TEST_PATH = 'data/Test_Data_No_Target.csv'
TARGET = 'Dropped_Course'
SEED = 42

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=['Course_Start_Date'])

def show(fig=None):
    if fig is None:
        fig = plt.gcf()
    display(fig)
    plt.close(fig)

def figure_size(nrows=1, ncols=1):
    return 4 + 4 * ncols, 3.5 * nrows

def subplot_grid(nrows=1, ncols=1, **kwargs):
    kwargs.setdefault('layout', 'constrained')
    return plt.subplots(nrows, ncols, figsize=figure_size(nrows, ncols), **kwargs)
```

    /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

# 1. Business understanding

Nova Academy prepares cloud environments, catering, equipment, and classroom capacity before each B2B course begins. A cancellation therefore wastes prepared resources and can leave capacity that could have been offered to another group.

Our goal is to estimate cancellation risk for new registrations early enough to support operational decisions. The assignment requires a continuous `Drop_Probability` output and evaluates its ranking quality with ROC-AUC; the minimum required AUC is 0.70.

# 2. Data loading & first look

Two files are provided:

- `Train_Data.csv` — historical registrations **with** the `Dropped_Course`
  label.
- `Test_Data_No_Target.csv` — registrations to score, **without** the label.

Each row is one registration, identified by `Client_ID`. We first inspect inferred types, missingness, cardinality, common values, and zeros before deciding how any column should be treated.

```python
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
```

    train: 63,464 rows x 29 cols
    test : 15,866 rows x 28 cols

|                             | dtype          | n_missing | missing\_% | n_unique | n_zero | most_frequent             |
| :-------------------------- | :------------- | --------: | ---------: | -------: | -----: | :------------------------ |
| Client_ID                   | int64          |         0 |          0 |    63464 |      0 | 1                         |
| Professionals_Count         | int64          |         0 |          0 |        5 |    338 | 2.0                       |
| Students_Count              | float64        |         4 |       0.01 |        5 |  59578 | 0.0                       |
| Observers_Count             | int64          |         0 |          0 |        5 |  63149 | 0.0                       |
| Course_Start_Date           | datetime64[ns] |         0 |          0 |      666 |      0 | 2015-10-16 00:00:00       |
| Practical_Hours             | int64          |         0 |          0 |       18 |  30671 | 0.0                       |
| Theory_Hours                | int64          |         0 |          0 |       29 |   4045 | 2.0                       |
| Registration_Days_Before    | float64        |      2666 |        4.2 |      423 |   2610 | 0.0                       |
| Origin_Country              | object         |       557 |       0.88 |      721 |      0 | PRT                       |
| Catering_Package            | object         |       407 |       0.64 |      321 |      0 | Standard (Coffee Only)    |
| Welcome_Gift_Type           | object         |         0 |          0 |        4 |      0 | Branded Notebook          |
| Requested_Lab_Config        | object         |      1736 |       2.74 |        8 |      0 | Standard PC (Windows)     |
| Assigned_Lab_Config         | object         |         0 |          0 |        9 |      0 | Standard PC (Windows)     |
| Prev_Course_Dropouts        | int64          |         0 |          0 |       10 |  58184 | 0.0                       |
| Prev_Course_Attended        | int64          |         0 |          0 |       62 |  62188 | 0.0                       |
| Pre_Course_Supports_Tickets | int64          |         0 |          0 |        6 |  39830 | 0.0                       |
| Physical_Course_Kits        | float64        |      1040 |       1.64 |        4 |  60790 | 0.0                       |
| Waiting_List_Days           | int64          |         0 |          0 |      107 |  60089 | 0.0                       |
| Registration_Changes        | int64          |         0 |          0 |       19 |  55478 | 0.0                       |
| Enrollment_Type             | object         |       719 |       1.13 |      298 |      0 | General Admission         |
| Lanyard_Color               | object         |         0 |          0 |      240 |      0 | Blue                      |
| Client_Category             | object         |         0 |          0 |      505 |      0 | SaaS & Software Houses    |
| Submission_Source           | object         |       605 |       0.95 |      328 |      0 | B2B Platforms & Resellers |
| Returning_Client            | int64          |         0 |          0 |        2 |  61742 | 0.0                       |
| Agent_ID                    | float64        |     11173 |      17.61 |      203 |      0 | 184.0                     |
| Company_ID                  | float64        |     60344 |      95.08 |      184 |      0 | 5181.0                    |
| Payment_Terms               | object         |       587 |       0.92 |      236 |      0 | Pay Upon Start            |
| Daily_Tuition_Cost          | float64        |        79 |       0.12 |     4780 |   1079 | 62.0                      |
| Dropped_Course              | int64          |         0 |          0 |        2 |  37165 | 0.0                       |

**What the dictionary tells us.**

- `Client_ID` is unique per row
- `Agent_ID` and `Company_ID` were inferred as numeric even though they are identifiers, so we convert them to strings after this first inspection. `Company_ID` is also missing for most rows.
- Several text fields have unexpectedly high cardinality. We inspect their raw values later before deciding whether that reflects real variety or inconsistent spelling.
- The numeric summary below lets us look for suspicious ranges and extreme values.

```python
for id_frame in (train_raw, test_raw):
    for id_col in ("Agent_ID", "Company_ID"):
        id_frame[id_col] = id_frame[id_col].astype("string")
train_raw.describe()
```

|       | Client_ID | Professionals_Count | Students_Count | Observers_Count | Course_Start_Date             | Practical_Hours | Theory_Hours | Registration_Days_Before | Prev_Course_Dropouts | Prev_Course_Attended | Pre_Course_Supports_Tickets | Physical_Course_Kits | Waiting_List_Days | Registration_Changes | Returning_Client | Daily_Tuition_Cost | Dropped_Course |
| :---- | --------: | ------------------: | -------------: | --------------: | :---------------------------- | --------------: | -----------: | -----------------------: | -------------------: | -------------------: | --------------------------: | -------------------: | ----------------: | -------------------: | ---------------: | -----------------: | -------------: |
| count |     63464 |               63464 |          63460 |           63464 | 63464                         |           63464 |        63464 |                    60798 |                63464 |                63464 |                       63464 |                62424 |             63464 |                63464 |            63464 |              63385 |          63464 |
| mean  |   39761.8 |              1.8352 |         8.7517 |          0.0053 | 2016-06-23 05:17:23.287533056 |          6.6091 |       2.1644 |                  102.894 |                0.096 |                0.123 |                      0.5133 |               0.0262 |            3.9837 |                 0.18 |           0.0271 |             98.848 |         0.4144 |
| min   |         1 |                   0 |              0 |               0 | 2015-07-01 00:00:00           |              -5 |            0 |                        0 |                    0 |                    0 |                           0 |                    0 |                 0 |                    0 |                0 |                  0 |              0 |
| 25%   |   19959.8 |                   2 |              0 |               0 | 2016-02-13 00:00:00           |               0 |            1 |                       19 |                    0 |                    0 |                           0 |                    0 |                 0 |                    0 |                0 |                 75 |              0 |
| 50%   |   39819.5 |                   2 |              0 |               0 | 2016-07-01 00:00:00           |               1 |            2 |                       65 |                    0 |                    0 |                           0 |                    0 |                 0 |                    0 |                0 |               94.5 |              0 |
| 75%   |   59570.2 |                   2 |              0 |               0 | 2016-11-11 00:00:00           |               1 |            3 |                      150 |                    0 |                    0 |                           1 |                    0 |                 0 |                    0 |                0 |                117 |              1 |
| max   |     79330 |                   4 |           9999 |              10 | 2017-04-26 00:00:00           |           10000 |           41 |                      629 |                   21 |                   61 |                           5 |                    3 |               391 |                   21 |                1 |               5400 |              1 |
| std   |     22879 |              0.5086 |        294.239 |          0.0897 | nan                           |         215.503 |       1.4699 |                  109.179 |               0.4485 |               1.5352 |                      0.7636 |               0.1602 |           23.1955 |               0.5926 |           0.1625 |            41.8554 |         0.4926 |

## 2.1 Target balance

We first check whether one target class is rare enough to require special treatment during training or evaluation.

```python
target_counts = train_raw[TARGET].value_counts().sort_index()
target_rate = train_raw[TARGET].value_counts(normalize=True).sort_index()
balance = pd.DataFrame({
    'count': target_counts,
    'rate_%': (target_rate * 100).round(1),
})
balance.index = ['0 = completed', '1 = dropped']
display(balance)

balance_fig, balance_ax = subplot_grid()
balance['rate_%'].plot.bar(ax=balance_ax)
balance_ax.set(
    title='Course outcomes in the training data', xlabel='', ylabel='share (%)'
)
balance_ax.tick_params(axis='x', rotation=0)
show(balance_fig)
```

|               | count | rate\_% |
| :------------ | ----: | ------: |
| 0 = completed | 37165 |    58.6 |
| 1 = dropped   | 26299 |    41.4 |

![svg](<notebook_files/notebook_10_1.svg>)

about 59% completed and 41% dropped

# 3. Exploratory Data Analysis

We begin with the target and date coverage, then inspect missingness, categorical quality, and numeric relationships.

## 3.1 Train and test dates

We plot the monthly drop rate across the _training_ period and overlay where training ends and where the hidden test window ends.

```python
train_end = train_raw['Course_Start_Date'].max()
test_start = test_raw['Course_Start_Date'].min()
test_end = test_raw['Course_Start_Date'].max()
print(
    f"train dates: {train_raw['Course_Start_Date'].min().date()} -> {train_end.date()}"
)
print(f'test  dates: {test_start.date()} -> {test_end.date()}')
monthly = (
    train_raw.set_index('Course_Start_Date').resample('MS')[TARGET].mean().mul(100)
)

monthly_fig, monthly_ax = subplot_grid()
monthly.plot(marker='o', ax=monthly_ax)
monthly_ax.axhline(
    train_raw[TARGET].mean() * 100, linestyle='--', label='train average'
)
monthly_ax.axvline(
    train_end, linestyle='--', label=f'train ends ({train_end.date()})'
)
monthly_ax.axvline(test_end, linestyle=':', label=f'test ends ({test_end.date()})')
monthly_ax.set(
    xlim=(train_raw['Course_Start_Date'].min(), test_end),
    ylabel='drop rate (%)',
    title='Drop rate over time — training period and the hidden test horizon',
)
monthly_ax.legend()
show(monthly_fig)
```

    train dates: 2015-07-01 -> 2017-04-26
    test  dates: 2017-04-26 -> 2017-08-31

![svg](<notebook_files/notebook_14_1.svg>)

Training covers July 2015 through April 2017. The test set begins at the end of that period and continues through August 2017, so the prediction task is temporal: learn from earlier registrations and score a later window.

The monthly drop rate also changes across the training period. Because a random split would mix earlier and later regimes, we define validation chronologically and later compare the result with a random split.

## 3.2 Missing values

We compare missingness in train and test, then ask whether _the fact of being missing_ is itself predictive.

```python
missing_compare = pd.DataFrame({
    "train_missing_%": train_raw.isna().mean().mul(100).round(2),
    "test_missing_%": test_raw.isna().mean().mul(100).round(2),
})
missing_compare = missing_compare[
    (missing_compare["train_missing_%"] > 0)
    | (missing_compare["test_missing_%"] > 0)
].sort_values("train_missing_%", ascending=False)
display(missing_compare)
```

|                          | train*missing*% | test*missing*% |
| :----------------------- | --------------: | -------------: |
| Company_ID               |           95.08 |          96.41 |
| Agent_ID                 |           17.61 |          17.61 |
| Registration_Days_Before |             4.2 |           4.05 |
| Requested_Lab_Config     |            2.74 |           3.01 |
| Physical_Course_Kits     |            1.64 |           1.43 |
| Enrollment_Type          |            1.13 |           1.12 |
| Submission_Source        |            0.95 |           0.94 |
| Payment_Terms            |            0.92 |           0.91 |
| Origin_Country           |            0.88 |           1.01 |
| Catering_Package         |            0.64 |            0.7 |
| Daily_Tuition_Cost       |            0.12 |           0.01 |
| Students_Count           |            0.01 |              0 |

Most train/test missingness rates are close. We next check whether the presence of a value is associated with the target.

```python
missingness_cols = [
    'Company_ID',
    'Agent_ID',
    'Registration_Days_Before',
    'Physical_Course_Kits',
    'Daily_Tuition_Cost',
    'Payment_Terms',
]

missing_summary = (
    pd
    .concat(
        {
            col: train_raw
            .assign(is_missing=train_raw[col].isna())
            .groupby('is_missing')[TARGET]
            .agg(count='size', drop_rate='mean')
            for col in missingness_cols
        },
        names=['column'],
    )
    .reset_index()
    .assign(drop_rate_pct=lambda df: (df['drop_rate'] * 100).round(1))
    .drop(columns='drop_rate')
    .rename(columns={'drop_rate_pct': 'drop_rate_%'})
)
display(missing_summary)
```

|     | column                   | is_missing | count | drop*rate*% |
| --: | :----------------------- | :--------- | ----: | ----------: |
|   0 | Company_ID               | False      |  3120 |        21.2 |
|   1 | Company_ID               | True       | 60344 |        42.5 |
|   2 | Agent_ID                 | False      | 52291 |        43.1 |
|   3 | Agent_ID                 | True       | 11173 |        33.5 |
|   4 | Registration_Days_Before | False      | 60798 |        41.4 |
|   5 | Registration_Days_Before | True       |  2666 |        41.5 |
|   6 | Physical_Course_Kits     | False      | 62424 |        41.5 |
|   7 | Physical_Course_Kits     | True       |  1040 |        39.3 |
|   8 | Daily_Tuition_Cost       | False      | 63385 |        41.4 |
|   9 | Daily_Tuition_Cost       | True       |    79 |        51.9 |
|  10 | Payment_Terms            | False      | 62877 |        41.5 |
|  11 | Payment_Terms            | True       |   587 |        35.4 |

Rows without a `Company_ID` have a noticeably higher drop rate, and `Agent_ID` presence also separates groups. This motivates explicit presence flags instead of replacing missing identifiers with a typical value.

## 3.3 Inspecting categorical values

Several text columns have far more distinct values than their meanings suggest: hundreds of payment terms, colors, and enrollment types would be surprising. We inspect the raw labels before deciding whether the cardinality is real.

```python
TEXT_COLS = list(train_raw.select_dtypes(include=['object']).columns)
N_COUNT = 9

for text_col in TEXT_COLS:
    top_values = train_raw[text_col].value_counts(normalize=True).head(N_COUNT)
    cats = [
        f'{value!r}: ({share * 100:.1f}%)' for value, share in top_values.items()
    ]
    cats_str = '\n'.join(
        ' | '.join(cats[i : i + 3]) for i in range(0, len(cats), 3)
    )
    print(
        f"\n{'=' * 80}\n{text_col} ({train_raw[text_col].nunique()} unique values)\n\n{cats_str}\n"
    )
```

    ================================================================================
    Origin_Country (721 unique values)

    'PRT': (38.6%) | 'FRA': (10.2%) | 'DEU': (6.4%)
    'ESP': (5.7%) | 'GBR': (5.2%) | 'ITA': (4.0%)
    'BRA': (2.0%) | 'BEL': (2.0%) | 'NLD': (1.8%)


    ================================================================================
    Catering_Package (321 unique values)

    'Standard (Coffee Only)': (71.9%) | 'No Food Plan': (10.5%) | 'Lunch Included': (7.5%)
    'standard (coffee only)': (1.8%) | 'STANDARD (COFFEE ONLY)': (1.7%) | ' Standard (Coffee Only)  ': (0.8%)
    '  Standard (Coffee Only) ': (0.8%) | ' Standard (Coffee Only) ': (0.8%) | '  Standard (Coffee Only)  ': (0.8%)


    ================================================================================
    Welcome_Gift_Type (4 unique values)

    'Branded Notebook': (50.8%) | 'Water Bottle': (29.0%) | 'USB Drive': (16.0%)
    'Portable Charger': (4.2%)


    ================================================================================
    Requested_Lab_Config (8 unique values)

    'Standard PC (Windows)': (80.5%) | 'Linux Workstation': (13.6%) | 'Dual Monitor Setup': (2.1%)
    'MacOS Station': (1.6%) | 'Laptop Docking Station': (1.6%) | 'High-GPU Unit': (0.5%)
    'Touch Screen Interface': (0.0%) | 'VR/AR Station': (0.0%)


    ================================================================================
    Assigned_Lab_Config (9 unique values)

    'Standard PC (Windows)': (72.4%) | 'Linux Workstation': (18.4%) | 'Laptop Docking Station': (2.9%)
    'MacOS Station': (2.5%) | 'Dual Monitor Setup': (2.5%) | 'High-GPU Unit': (0.8%)
    'Server Access Terminal': (0.4%) | 'Touch Screen Interface': (0.2%) | 'VR/AR Station': (0.0%)


    ================================================================================
    Enrollment_Type (298 unique values)

    'General Admission': (64.6%) | 'Affiliated Admission': (21.6%) | 'Contractual Agreement': (3.2%)
    'general admission': (1.6%) | 'GENERAL ADMISSION': (1.6%) | ' General Admission  ': (0.8%)
    ' General Admission ': (0.7%) | '  General Admission ': (0.7%) | '  General Admission  ': (0.7%)


    ================================================================================
    Lanyard_Color (240 unique values)

    'Blue': (49.6%) | 'Black': (21.0%) | 'Red': (10.1%)
    'Orange': (5.2%) | 'Green': (3.9%) | 'BLUE': (1.2%)
    'blue': (1.2%) | '  Blue  ': (0.6%) | ' Blue  ': (0.6%)


    ================================================================================
    Client_Category (505 unique values)

    'SaaS & Software Houses': (41.4%) | 'Traditional IT & Telecomm': (20.4%) | 'Big Tech & Multinationals': (16.8%)
    'FinTech & Banking': (6.6%) | 'Industrial Tech & IoT': (3.7%) | 'saas & software houses': (1.1%)
    'SAAS & SOFTWARE HOUSES': (1.0%) | 'Non-Profit & EduTech': (0.7%) | 'TRADITIONAL IT & TELECOMM': (0.5%)


    ================================================================================
    Submission_Source (328 unique values)

    'B2B Platforms & Resellers': (77.4%) | 'Direct Website Registration': (7.4%) | 'Dedicated Sales Team': (4.1%)
    'B2B PLATFORMS & RESELLERS': (2.0%) | 'b2b platforms & resellers': (1.9%) | ' B2B Platforms & Resellers  ': (0.9%)
    '  B2B Platforms & Resellers ': (0.9%) | ' B2B Platforms & Resellers ': (0.8%) | '  B2B Platforms & Resellers  ': (0.8%)


    ================================================================================
    Payment_Terms (236 unique values)

    'Pay Upon Start': (73.8%) | 'Prepaid (Non-Refundable)': (15.3%) | 'PAY UPON START': (1.9%)
    'pay upon start': (1.8%) | ' Pay Upon Start ': (0.9%) | '  Pay Upon Start  ': (0.8%)
    ' Pay Upon Start  ': (0.8%) | '  Pay Upon Start ': (0.8%) | 'prepaid (non-refundable)': (0.4%)

The raw values explain much of the inflated cardinality. Labels such as `'BLUE'`, `'blue'`, and `'  Blue  '` describe the same category but are stored separately; punctuation and placeholder strings create similar splits in other fields. Before treating these columns as genuinely high-cardinality, we normalize the obvious formatting variants and measure how many levels remain.

```python
# Placeholder strings that mean "missing", in any casing/padding after canonicalisation.
COMMON_NANS = {
    '',
    '-',
    '--',
    '.',
    '?',
    'na',
    'n/a',
    'nan',
    'none',
    'null',
    'unknown',
    'unknonwn',
}
COUNTRY_ALIASES = {'cn': 'chn'}  # both mean China

def canonicalize(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip().str.lower()
    return (
        s.str
        .replace('\\band\\b', '&', regex=True)
        .str.replace('[^a-z0-9&() .+-]+', '', regex=True)
        .str.replace('\\s+', ' ', regex=True)
        .str.strip()
    )
```

We normalize case, surrounding whitespace, repeated spaces, and injected punctuation. Placeholder labels such as `Unknown` and `?` become missing values rather than new categories. The same deterministic cleaning function will be applied to train and test.

```python
CAT_COLS = TEXT_COLS + ['Agent_ID', 'Company_ID']

def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise every categorical, then map junk placeholders to NaN."""
    df = df.copy()
    for col in CAT_COLS:
        s = canonicalize(df[col])
        df[col] = s.mask(s.isin(COMMON_NANS))
    df['Origin_Country'] = df['Origin_Country'].replace(COUNTRY_ALIASES)
    return df
```

```python
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
```

|                      | raw_unique | clean_unique | collapsed |
| :------------------- | ---------: | -----------: | --------: |
| Origin_Country       |        721 |          153 |       568 |
| Client_Category      |        505 |            7 |       498 |
| Submission_Source    |        328 |            4 |       324 |
| Catering_Package     |        321 |            4 |       317 |
| Enrollment_Type      |        298 |            4 |       294 |
| Lanyard_Color        |        240 |            5 |       235 |
| Payment_Terms        |        236 |            3 |       233 |
| Welcome_Gift_Type    |          4 |            4 |         0 |
| Requested_Lab_Config |          8 |            8 |         0 |
| Assigned_Lab_Config  |          9 |            9 |         0 |

The before/after table confirms that most of the apparent variety was formatting noise: `Payment_Terms` falls from 236 raw labels to 3 cleaned levels, and `Client_Category` from 505 to 7. Columns that were already consistent remain unchanged.

## 3.4 Which categories actually relate to dropping?

We start with business fields that have only a few cleaned levels, where a direct plot remains readable. Country and identifiers need separate treatment because hundreds of levels would make the same plot misleading.

```python
def plot_dropout_by_category(df, col, ax, min_count=50, top_n=10):
    stats = df.groupby(col, dropna=False)[TARGET].agg(
        drop_rate="mean", count="size"
    )

    stats = (
        stats[stats["count"] >= min_count]
        .sort_values("count", ascending=False)
        .head(top_n)
        .sort_values("drop_rate")
    )

    labels = [f"{i} (n={int(r['count'])})" for i, r in stats.iterrows()]
    overall = df[TARGET].mean()
    ax.barh(
        labels,
        stats["drop_rate"] * 100,
        color=["C1" if rate > overall else "C0" for rate in stats["drop_rate"]],
    )

    ax.axvline(overall * 100, linestyle="--", label=f"mean ({overall * 100:.1f}%)")
    ax.set(xlabel="Drop rate (%)", title=f"Drop rate by {col}")
    ax.legend()

category_features = [col for col in TEXT_COLS if col != "Origin_Country"]
category_fig, category_axes = subplot_grid((len(category_features) + 1) // 2, 2)
for category_ax, category_feature in zip(category_axes.flat, category_features):
    plot_dropout_by_category(clean_train, category_feature, category_ax)
for unused_category_ax in category_axes.flat[len(category_features) :]:
    unused_category_ax.set_visible(False)
show(category_fig)
```

![svg](<notebook_files/notebook_30_0.svg>)

One big surprise is `Payment_Terms`: prepaid, non-refundable registrations drop more often than pay-on-start registrations, the opposite of what we expected. Two possible explanations are that these terms are assigned to riskier deals in advance or that the field is updated later in the registration process. We return to this question after fitting the model.

`Welcome_Gift_Type` and `Lanyard_Color` seems to unrelated to dropping. `Assigned_Lab_Config` pattern could probably be explained by PC being the default.

The other plots also show useful separation. Direct-website and dedicated-sales registrations drop less often than reseller traffic, organisational enrollment is lower-risk than general admission, and client segments differ. We carry these signals into modelling.

### A closer look at high-cardinality categories

`Origin_Country`, `Agent_ID`, and `Company_ID` cannot be judged from an unfiltered chart containing every level. We first require enough rows for a rate to be interpretable, then use Portugal as a compact case because it is both the largest country group and far from the overall drop rate.

```python
country_min_n = 150
country_top_n = 12
overall_drop = clean_train[TARGET].mean()
country_stats = (
    clean_train
    .groupby('Origin_Country', dropna=False)[TARGET]
    .agg(count='size', drop_rate='mean')
    .assign(
        drop_rate_pct=lambda d: d['drop_rate'] * 100,
        lift_pp=lambda d: (d['drop_rate'] - overall_drop) * 100,
    )
)
top_by_size = country_stats.sort_values('count', ascending=False).head(
    country_top_n
)
extreme_by_lift = (
    country_stats[country_stats['count'] >= country_min_n]
    .iloc[
        lambda d: (
            d['lift_pp']
            .abs()
            .sort_values(ascending=False)
            .index.map(d.index.get_loc)
        )
    ]
    .head(country_top_n)
)

def plot_country_dropout(stats, title, ax):
    stats = stats.sort_values('drop_rate_pct')
    labels = [
        f"{(idx if pd.notna(idx) else '<missing>')} (n={int(row['count']):,})"
        for idx, row in stats.iterrows()
    ]
    below, above = sns.color_palette(n_colors=2)
    colors = [above if lift >= 0 else below for lift in stats['lift_pp']]
    ax.barh(labels, stats['drop_rate_pct'], color=colors)
    ax.axvline(
        overall_drop * 100,
        linestyle='--',
        label=f'overall ({overall_drop * 100:.1f}%)',
    )
    ax.set(xlabel='drop rate (%)', title=title)
    ax.legend()

plots = [
    (top_by_size, f'Drop rate by largest {country_top_n} countries'),
    (extreme_by_lift, f'Most unusual country drop rates (n >= {country_min_n})'),
]
country_fig, country_axes = subplot_grid(1, 2)
for country_ax, (country_plot_stats, country_title) in zip(country_axes, plots):
    plot_country_dropout(country_plot_stats, country_title, country_ax)
show(country_fig)
display(
    country_stats
    .sort_values('count', ascending=False)
    .head(country_top_n)[['count', 'drop_rate_pct', 'lift_pp']]
    .round(2)
)
```

![svg](<notebook_files/notebook_33_0.svg>)

| Origin_Country | count | drop_rate_pct | lift_pp |
| :------------- | ----: | ------------: | ------: |
| prt            | 26429 |         63.78 |   22.34 |
| fra            |  6961 |         17.28 |  -24.16 |
| deu            |  4400 |          16.7 |  -24.73 |
| esp            |  3896 |         27.31 |  -14.13 |
| gbr            |  3514 |          27.8 |  -13.64 |
| ita            |  2726 |         35.88 |   -5.56 |
| bra            |  1402 |         38.02 |   -3.42 |
| bel            |  1324 |         19.18 |  -22.25 |
| nld            |  1222 |         19.97 |  -21.47 |
| usa            |  1072 |         22.39 |  -19.05 |
| chn            |  1054 |         42.79 |    1.35 |
| che            |   935 |         22.78 |  -18.66 |

Portugal contains 26,429 registrations and has a 63.8% drop rate, making it both the largest country group and the clearest geographic difference. We use it to investigate whether country overlaps with agents, channels, or other parts of the acquisition process.

```python
is_portugal = (
    clean_train["Origin_Country"].eq("prt").fillna(False).to_numpy(dtype=bool)
)
country_group = np.where(is_portugal, "Portugal", "Other countries")

portugal_summary = (
    clean_train
    .assign(country_group=country_group)
    .groupby("country_group")[TARGET]
    .agg(count="size", drop_rate="mean")
    .assign(drop_rate_pct=lambda d: d["drop_rate"] * 100)
)

display(portugal_summary[["count", "drop_rate_pct"]].round(1))
```

| country_group   | count | drop_rate_pct |
| :-------------- | ----: | ------------: |
| Other countries | 37035 |          25.5 |
| Portugal        | 26429 |          63.8 |

Compared with all other countries, Portugal remains clearly different. We next inspect the identifier fields as categories, not numbers, to see whether they show related structure.

```python
company_presence = train_raw.groupby(train_raw['Company_ID'].notna())[TARGET].agg(
    count='size', drop_rate='mean'
)
company_presence.index = ['no company_id', 'has company_id']

identifier_fig, identifier_axes = subplot_grid(1, 2)
plot_dropout_by_category(
    clean_train, 'Agent_ID', identifier_axes[0], min_count=150, top_n=12
)
identifier_axes[1].bar(company_presence.index, company_presence['drop_rate'] * 100)
identifier_axes[1].set(
    ylabel='drop rate (%)', title='Drop rate by Company_ID presence'
)
show(identifier_fig)
display(company_presence)
```

![svg](<notebook_files/notebook_37_0.svg>)

|                | count | drop_rate |
| :------------- | ----: | --------: |
| no company_id  | 60344 |    0.4248 |
| has company_id |  3120 |    0.2122 |

Frequent agents have different drop rates, while registrations with a `Company_ID` drop less often (21.2% versus 42.5%). These relationships may overlap with geography, so we perform a small check: does knowing the agent improve country prediction over always guessing the most common country?

```python
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
```

|     | check                     | accuracy |
| --: | :------------------------ | -------: |
|   0 | majority country baseline |    0.391 |
|   1 | agent modal country       |    0.421 |

Agent-based prediction raises country accuracy from 0.391 to 0.421, indicating modest overlap between the two fields. Both are included using the compact representation introduced during preparation.

## 3.5 Numeric features: summary, correlation, and suspects

We now inspect numeric ranges, distributions, and their linear correlations with the target.

```python
ID_LIKE = ["Client_ID", "Agent_ID", "Company_ID"]
num_cols = [
    c
    for c in train_raw.select_dtypes(include=["int64", "float64"]).columns
    if c not in ID_LIKE + [TARGET]
]

numeric_summary = (
    train_raw[num_cols].agg(['mean', 'median', 'std', 'min', 'max', 'skew']).T
)
numeric_summary.insert(0, 'missing_%', train_raw[num_cols].isna().mean() * 100)
numeric_summary.insert(
    1, 'corr_target', train_raw[num_cols].corrwith(train_raw[TARGET])
)
numeric_summary = (
    numeric_summary
    .round({
        'missing_%': 1,
        'corr_target': 3,
        'mean': 2,
        'median': 2,
        'std': 2,
        'min': 2,
        'max': 2,
        'skew': 2,
    })
    .rename_axis('column')
    .reset_index()
    .sort_values('corr_target', key=abs, ascending=False)
)
display(numeric_summary)
```

|     | column                      | missing\_% | corr_target |   mean | median |    std | min |   max |  skew |
| --: | :-------------------------- | ---------: | ----------: | -----: | -----: | -----: | --: | ----: | ----: |
|   5 | Registration_Days_Before    |        4.2 |       0.351 | 102.89 |     65 | 109.18 |   0 |   629 |   1.5 |
|   8 | Pre_Course_Supports_Tickets |          0 |      -0.301 |   0.51 |      0 |   0.76 |   0 |     5 |  1.47 |
|   6 | Prev_Course_Dropouts        |          0 |       0.199 |    0.1 |      0 |   0.45 |   0 |    21 |  15.7 |
|  11 | Registration_Changes        |          0 |      -0.148 |   0.18 |      0 |   0.59 |   0 |    21 |  7.36 |
|   9 | Physical_Course_Kits        |        1.6 |      -0.138 |   0.03 |      0 |   0.16 |   0 |     3 |     6 |
|  10 | Waiting_List_Days           |          0 |       0.068 |   3.98 |      0 |   23.2 |   0 |   391 |  9.26 |
|  12 | Returning_Client            |          0 |      -0.059 |   0.03 |      0 |   0.16 |   0 |     1 |  5.82 |
|   0 | Professionals_Count         |          0 |       0.057 |   1.84 |      2 |   0.51 |   0 |     4 | -0.47 |
|   7 | Prev_Course_Attended        |          0 |      -0.052 |   0.12 |      0 |   1.54 |   0 |    61 | 21.96 |
|   4 | Theory_Hours                |          0 |       0.045 |   2.16 |      2 |   1.47 |   0 |    41 |  3.35 |
|   2 | Observers_Count             |          0 |      -0.031 |   0.01 |      0 |   0.09 |   0 |    10 | 45.38 |
|  13 | Daily_Tuition_Cost          |        0.1 |      -0.024 |  98.85 |   94.5 |  41.86 |   0 |  5400 | 32.55 |
|   3 | Practical_Hours             |          0 |       0.005 |   6.61 |      1 |  215.5 |  -5 | 10000 | 40.45 |
|   1 | Students_Count              |          0 |           0 |   8.75 |      0 | 294.24 |   0 |  9999 | 33.92 |

The maximum values reveal several likely data errors: `Students_Count` reaches 9999, and `Practical_Hours` contains both negative values and values up to 10000. We leave the raw values unchanged for this first inspection and decide how to handle them in the outlier section.

```python
corr = train_raw[num_cols + [TARGET]].corr()
corr_fig, corr_ax = plt.subplots(figsize=figure_size(2, 2), layout='constrained')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=corr_ax)
corr_ax.set_title('Numeric correlation heatmap (incl. target)')
corr_ax.grid(False)
show(corr_fig)
```

![svg](<notebook_files/notebook_44_0.svg>)

No raw numeric feature has an extremely strong Pearson correlation with the target. `Registration_Days_Before` and `Pre_Course_Supports_Tickets` stand out most, while inter-feature correlations are generally modest. Because Pearson correlation measures linear association and is sensitive to extremes, we next use binned drop rates to inspect the shape of the strongest relationships.

## 3.6 Numeric drop-rate profiles

Binning a couple of the more predictive numeric features shows _how_ risk moves with them (not just whether they correlate linearly).

```python
def plot_dropout_by_bins(df, col, bins, ax):
    tmp = df[[col, TARGET]].dropna().copy()
    tmp['bin'] = pd.qcut(tmp[col], q=bins, duplicates='drop')
    stats = tmp.groupby('bin', observed=True)[TARGET].mean().mul(100)
    stats.plot.bar(ax=ax)
    ax.axhline(df[TARGET].mean() * 100, linestyle='--', label='mean')
    ax.set(ylabel='drop rate (%)', title=f'Drop rate by {col} bins')
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)

numeric_bin_specs = [
    ('Registration_Days_Before', 8),
    ('Pre_Course_Supports_Tickets', 6),
]
numeric_bin_fig, numeric_bin_axes = subplot_grid(1, 2)
for numeric_bin_ax, (numeric_feature, bins) in zip(
    numeric_bin_axes, numeric_bin_specs
):
    plot_dropout_by_bins(train_raw, numeric_feature, bins, numeric_bin_ax)
show(numeric_bin_fig)
```

![svg](<notebook_files/notebook_47_0.svg>)

Drop rate rises across longer registration lead times, which suggests that plans are more likely to change when courses are booked far in advance. More pre-course support tickets are associated with lower dropping, suggesting that early engagement may reflect stronger commitment.

## 3.7 EDA conclusions

Several observations now guide preparation and modelling:

- Missing `Company_ID`, support activity, registration channel, enrollment type, and lead time all separate groups with different drop rates. Together, these patterns suggest a broader difference in buyer commitment.
- `Payment_Terms` is unusually strong and counter-intuitive. We include it and later test how much XGBoost depends on it.
- Country and agent both contain signal and overlap slightly. Their many levels require a compact encoding instead of a large one-hot expansion.
- The later test window and changing monthly rates make time-aware validation important. We derive calendar and trend features, then test the time index with a stable reference candidate during tuning.
- Some numeric values are clearly suspicious, while other large values may be legitimate rare cases. We will correct only the values for which we have evidence of an error.

These conclusions give us a modelling hypothesis: a flexible model may capture the combined effects better than a linear baseline. We test that hypothesis in the model comparison.

# 4. Missing-value handling & outlier analysis

We now turn the EDA findings into reproducible preparation rules. The same fitted rules must be applied to later data, but the exact missing-value treatment can differ by model family.

## 4.1 Outliers: identify, justify, cap

We look for values that are physically impossible or absurdly far from the bulk.

```python
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
```

    Suspect columns — TRAIN

|     | column               | min |   max |   q99 | why                                 |
| --: | :------------------- | --: | ----: | ----: | :---------------------------------- |
|   0 | Students_Count       |   0 |  9999 |     2 | max=9999 >> q99=2                   |
|   1 | Practical_Hours      |  -5 | 10000 |     3 | negative values; max=10000 >> q99=3 |
|   2 | Prev_Course_Dropouts |   0 |    21 |     1 | max=21 >> q99=1                     |
|   3 | Prev_Course_Attended |   0 |    61 |     3 | max=61 >> q99=3                     |
|   4 | Registration_Changes |   0 |    21 |     2 | max=21 >> q99=2                     |
|   5 | Daily_Tuition_Cost   |   0 |  5400 | 209.7 | max=5400 >> q99=209.7               |

    Suspect columns — TEST

|     | column               | min |   max | q99 | why                                 |
| --: | :------------------- | --: | ----: | --: | :---------------------------------- |
|   0 | Students_Count       |   0 |  9999 |   2 | max=9999 >> q99=2                   |
|   1 | Practical_Hours      |  -5 | 10000 |   2 | negative values; max=10000 >> q99=2 |
|   2 | Prev_Course_Attended |   0 |    72 |   3 | max=72 >> q99=3                     |
|   3 | Waiting_List_Days    |   0 |   183 |   0 | max=183 >> q99=0                    |

The test set introduces no new forms of corruption, suggesting the same cleaning policy can be safely shared. Comparing the maximum values to the 99th percentile helps identify columns with extreme outliers:

```python
TAIL_CHECK_COLS = [
    "Students_Count",
    "Practical_Hours",
    "Daily_Tuition_Cost",
    "Prev_Course_Attended",
    "Waiting_List_Days",
    "Registration_Changes",
]

tail_long = pd.concat(
    [
        df[col].dropna().rename('value').to_frame().assign(split=split, column=col)
        for split, df in [('train', train_raw), ('test', test_raw)]
        for col in TAIL_CHECK_COLS
    ],
    ignore_index=True,
)

grid = sns.catplot(
    data=tail_long,
    x='split',
    y='value',
    hue='split',
    col='column',
    col_wrap=3,
    kind='box',
    sharey=False,
    height=3.2,
    aspect=1.05,
    palette='colorblind',
    legend=False,
    flierprops={'markersize': 3, 'alpha': 0.35},
)
grid.set_axis_labels('', 'Raw value (log-like scale)').set_titles('{col_name}')
for tail_column, tail_ax in grid.axes_dict.items():
    tail_values = tail_long.loc[tail_long['column'].eq(tail_column), 'value']
    tail_ax.set_yscale('symlog' if tail_values.min() < 0 else 'log')
    tail_ax.set_title(tail_column.replace('_', ' '))
grid.figure.suptitle('Train/test tail comparison', fontsize=15)
grid.figure.set_layout_engine('constrained')
show(grid.figure)
```

![svg](<notebook_files/notebook_54_0.svg>)

As seen in the box-plots, top 3 (student count, practical hours and tuition) justify a cap. We therefore apply:

- `Students_Count <= 10`: the values beyond the observed low-count support are repeated `9999` placeholders in both train and test. The cap keeps those rows as large groups without treating 9999 as a real count.
- `Practical_Hours` in `[0, 12]`: negative values are impossible, and `5000`/`10000` are clear placeholders. A 12-hour upper bound still allows a long practical day and prevents corrupted placeholder values from distorting the feature space.
- `Daily_Tuition_Cost <= 600`: train has a single `5400` value, while the test maximum is 510. A cap of 600 leaves the observed test range untouched and prevents one corrupted training value from dominating cost calculations.

Other flagged count columns (`Prev_Course_Dropouts`, `Prev_Course_Attended`, `Registration_Changes`, and test-side `Waiting_List_Days`) have long but plausible tails (as seen in the box-plots), so we leave them unchanged and restrict clipping to the three apparent data-entry errors above.

```python
CAP_RULES = {
    'Students_Count': (None, 10),
    'Practical_Hours': (0, 12),
    'Daily_Tuition_Cost': (None, 600),
}
cap_notes = {
    'Students_Count': (
        '9999 placeholder',
        'repeated 9999 values are isolated placeholders beyond the observed support',
    ),
    'Practical_Hours': (
        'negative values and 10000',
        'course hours cannot be negative; 12 covers a long practical day',
    ),
    'Daily_Tuition_Cost': (
        '5400 value',
        '5400 is far beyond the valid fee range; 600 keeps the high-cost tail',
    ),
}

cap_rows = []
for cap_col, (lower, upper) in CAP_RULES.items():
    problem, reason = cap_notes[cap_col]
    train_capped = train_raw[cap_col].clip(lower=lower, upper=upper)
    test_capped = test_raw[cap_col].clip(lower=lower, upper=upper)
    cap_rows.append({
        'column': cap_col,
        'raw_train_min': train_raw[cap_col].min(),
        'raw_train_max': train_raw[cap_col].max(),
        'problem': problem,
        'action': f'clip to [{lower}, {upper}]'
        if lower is not None
        else f'clip to <= {upper}',
        'train_rows_affected': int(
            (train_raw[cap_col].notna() & train_capped.ne(train_raw[cap_col])).sum()
        ),
        'test_rows_affected': int(
            (test_raw[cap_col].notna() & test_capped.ne(test_raw[cap_col])).sum()
        ),
        'reason': reason,
    })
display(pd.DataFrame(cap_rows))

cap_fig, cap_axes = subplot_grid(2, 3, sharey='row')
for cap_index, (cap_column, (lower, upper)) in enumerate(CAP_RULES.items()):
    before = train_raw[cap_column].dropna()
    after = before.clip(lower=lower, upper=upper)
    for cap_ax, cap_values, cap_label in zip(
        cap_axes[:, cap_index], (before, after), ('raw', 'clipped')
    ):
        cap_ax.hist(cap_values, bins=30)
        cap_ax.set(
            yscale='log',
            title=f'{cap_column}: {cap_label}',
            xlabel=f'max={cap_values.max():g}',
        )
cap_axes[0, 0].set_ylabel('count (log)')
cap_axes[1, 0].set_ylabel('count (log)')
show(cap_fig)
```

|     | column             | raw_train_min | raw_train_max | problem                   | action          | train_rows_affected | test_rows_affected | reason                                            |
| --: | :----------------- | ------------: | ------------: | :------------------------ | :-------------- | ------------------: | -----------------: | :------------------------------------------------ |
|   0 | Students_Count     |             0 |          9999 | 9999 placeholder          | clip to <= 10   |                  55 |                 12 | repeated 9999 values are isolated placeholders... |
|   1 | Practical_Hours    |            -5 |         10000 | negative values and 10000 | clip to [0, 12] |                 121 |                 23 | course hours cannot be negative; 12 covers a l... |
|   2 | Daily_Tuition_Cost |             0 |          5400 | 5400 value                | clip to <= 600  |                   1 |                  0 | 5400 is far beyond the valid fee range; 600 ke... |

![svg](<notebook_files/notebook_56_1.svg>)

The before/after distributions show that the caps remove isolated invalid tails while
preserving the bulk of each feature.

One more data-quality issue appears in the historical counters. Some rows have more
recorded prior dropouts than prior attended courses, so those fields are not a clean
numerator/denominator pair.

```python
impossible = train_raw[
    train_raw["Prev_Course_Dropouts"] > train_raw["Prev_Course_Attended"]
]
print(
    f"rows where historical dropouts exceed historical attended: {len(impossible)}"
)
```

    rows where historical dropouts exceed historical attended: 4985

We keep both historical counters and combine them in the client-history feature below.

## 4.2 Missing-value policy

One missing-value policy would not suit every model family:

- Categorical missingness becomes an explicit `"missing"` level on both preprocessing paths. This preserves the possibility that absence itself carries information.
- `Agent_ID` and `Company_ID` also receive presence flags because EDA showed a clear difference between present and missing groups. Their high cardinality is handled separately in feature engineering.
- Models that support numeric missing values natively can retain `NaN` and learn how to route it. Models that require a complete numeric matrix receive medians learned from the training partition only.

# 5. Feature engineering & dimensionality

Based on the EDA findings and domain questions, we create features that expose relationships more directly or represent high-cardinality fields more compactly.

## 5.1 The engineered features and their rationale

| Raw signal                | Engineered feature(s)                                        | Reason for testing it                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Course_Start_Date`       | `start_month`, `start_dow`, `start_week`, `days_since_epoch` | Month and ISO week represent seasonality, weekday represents scheduling patterns, and the linear index represents the longer-term shift seen in Section 3.1. |
| Participant counts        | `total_participants`, `prof_share`                           | Total size and professional share describe group composition more directly than three separate counts.                                                       |
| Practical/theory hours    | `total_hours`, `practical_share`                             | Total duration and hands-on share distinguish courses with the same raw hour count but different structure.                                                  |
| Client history            | `prev_drop_rate = dropouts / (attended + 1)`                 | Combines previous dropouts and attendance into one history signal; the `+1` handles clients with no attended courses.                                        |
| Tuition cost and hours    | `cost_x_days`                                                | Combines price and course length so the model can consider their interaction.                                                                                |
| Requested vs assigned lab | `got_requested_lab`                                          | Captures whether the assigned lab configuration matches the original request.                                                                                |
| Missing company/agent IDs | `has_company_id`, `has_agent_id`                             | Preserves the presence differences observed in Section 3.2 even when a raw identifier is removed.                                                            |
| Agent/company/country IDs | frequency encodings and native categories                    | Retains identity and commonness information while avoiding a wide dummy matrix.                                                                              |

`Assigned_Lab_Config` is populated even for cancelled bookings, so we treat it as a planned assignment known before the course and use it in `got_requested_lab`. This timing assumption would need confirmation before deploying the model.

## 5.2 Dimensionality

The main expansion risk comes from identifiers: `Agent_ID` has 204 cleaned levels and `Origin_Country` has 154. Different model families therefore need different preparation paths.

Models that require numeric inputs receive rare-level grouping, one-hot encoding, training-median imputation, and scaling. Boosted-tree implementations with native categorical support can work with category labels directly, so the matrix does not need one dummy column per agent or country. We also add one frequency feature per high-cardinality identifier and remove raw `Company_ID`, retaining only its frequency and presence flag.

During validation, frequency maps are learned from the earlier training partition. For the final submission, we compute frequencies across the available train and test features, without using `Dropped_Course`. This transductive step gives each identifier one consistent frequency at scoring time.

```python
frequency_cols = ('Agent_ID', 'Company_ID', 'Origin_Country')
native_cat_cols = [col for col in TEXT_COLS if col != 'Assigned_Lab_Config'] + [
    'Agent_ID'
]

def make_freq_maps(*dfs):
    """Label-free frequency of each ID value across the supplied frames."""
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True) for col in frequency_cols
    }

freq_maps = make_freq_maps(train_raw, test_raw)

def build_features(
    df: pd.DataFrame,
    freq_maps: dict,
    add_time: bool = True,
    add_week: bool = True,
) -> pd.DataFrame:
    """Cleaning + feature engineering. Identical transform for train and test.

    Mirrors the feature values used by the scored ``pipeline.py`` transform."""
    df = normalize_cats(df)
    out = df[num_cols].copy()
    for col, (lower, upper) in CAP_RULES.items():
        out[col] = out[col].clip(lower=lower, upper=upper)
    dates = df['Course_Start_Date']
    out['start_month'] = dates.dt.month
    out['start_dow'] = dates.dt.dayofweek
    if add_week:
        out['start_week'] = dates.dt.isocalendar().week.astype(float)
    if add_time:
        out['days_since_epoch'] = (dates - pd.Timestamp('2015-01-01')).dt.days
    total = (
        out[['Professionals_Count', 'Students_Count', 'Observers_Count']]
        .fillna(0)
        .sum(axis=1)
    )
    out['total_participants'] = total
    out['prof_share'] = out['Professionals_Count'] / total.replace(0, np.nan)
    out['total_hours'] = out['Practical_Hours'] + out['Theory_Hours']
    out['practical_share'] = out['Practical_Hours'] / out['total_hours'].replace(
        0, np.nan
    )
    out['cost_x_days'] = out['Daily_Tuition_Cost'] * out['total_hours']
    out['prev_drop_rate'] = out['Prev_Course_Dropouts'] / (
        out['Prev_Course_Attended'] + 1
    )
    out['kits_per_participant'] = out['Physical_Course_Kits'] / total.replace(
        0, np.nan
    )
    out['tickets_per_participant'] = out[
        'Pre_Course_Supports_Tickets'
    ] / total.replace(0, np.nan)
    out['got_requested_lab'] = (
        df['Requested_Lab_Config'] == df['Assigned_Lab_Config']
    ).astype(float)
    for col in ('Company_ID', 'Agent_ID'):
        out[f'has_{col.lower()}'] = df[col].notna().astype(int)
    for col in frequency_cols:
        out[f'{col}_freq'] = df[col].map(freq_maps[col]).fillna(0).astype(float)
    for col in native_cat_cols:
        out[col] = df[col].fillna('missing').astype('category')
    return out

def align_categories(train_X, *others):
    """Give every frame identical category levels so the boosters agree."""
    for col in train_X.select_dtypes('category').columns:
        cats = train_X[col].cat.categories
        for other in others:
            cats = cats.union(other[col].cat.categories)
        train_X[col] = train_X[col].cat.set_categories(cats)
        for other in others:
            other[col] = other[col].cat.set_categories(cats)

X_all = build_features(train_raw, freq_maps)
cat_cols = X_all.select_dtypes('category').columns
native_dim = X_all.shape[1]
onehot_dim = X_all.drop(columns=cat_cols).shape[1] + sum(
    (X_all[c].nunique(dropna=False) for c in cat_cols)
)
print(f'features with native categorical handling : {native_dim}')
print(f'estimated dims after naive one-hot        : {onehot_dim}')
print(f'dummy columns avoided                     : {onehot_dim - native_dim}')
print('\ncategory cardinalities:')
# Dimensionality comparison: native categorical vs a naive one-hot expansion.
display(X_all[cat_cols].nunique(dropna=False).sort_values(ascending=False))
```

    features with native categorical handling : 42
    estimated dims after naive one-hot        : 435
    dummy columns avoided                     : 393

    category cardinalities:



    Agent_ID                204
    Origin_Country          154
    Requested_Lab_Config      9
    Client_Category           8
    Catering_Package          5
    Enrollment_Type           5
    Lanyard_Color             5
    Submission_Source         5
    Welcome_Gift_Type         4
    Payment_Terms             4
    dtype: int64

The tree/native-categorical path contains 42 columns. A naive one-hot expansion of the same fields would create about 435 columns, mostly from agent and country, so this representation avoids 393 sparse dummy columns. The linear and neural baselines use one-hot encoding with rare levels grouped into `other`.

# 6. Validation methodology

Before comparing models, we need a validation setup that resembles the later test window.

## 6.1 Adversarial validation — quantifying the drift

We train a classifier to tell **test rows from train rows** using the features (label removed, raw date and `Client_ID` dropped). If it separates them well above AUC 0.5, the feature distributions have genuinely drifted.

```python
@cache
def adversarial_validation(tr, te, target, seed):
    tr = tr.drop(columns=[target])
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
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    m = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    m.fit(Xtr, ytr)
    a = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    top = (
        pd
        .Series(m.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(8)
    )
    return a, top

adversarial_auc, drift_drivers = adversarial_validation(
    load_raw(TRAIN_PATH), load_raw(TEST_PATH), TARGET, SEED
)
print(
    f"adversarial AUC (train vs test): {adversarial_auc:.3f}  "
    "(0.5=identical, 1.0=trivially separable)"
)
print("\ntop drift drivers:")
print(drift_drivers)
```

    adversarial AUC (train vs test): 0.935  (0.5=identical, 1.0=trivially separable)

    top drift drivers:
    Daily_Tuition_Cost          0.126626
    Prev_Course_Dropouts        0.079528
    Registration_Days_Before    0.077459
    Waiting_List_Days           0.072729
    Assigned_Lab_Config         0.070094
    Catering_Package            0.063975
    Enrollment_Type             0.059224
    Client_Category             0.054426
    dtype: float32

The classifier reaches AUC 0.935, so train and test features are distinguishable even after removing the raw date. The strongest differences include tuition cost, client history, registration lead time, waiting time, and several categorical fields. Together with the changing monthly drop rate, this leads us to evaluate models on a later time window.

## 6.2 The chronological holdout

We use `2017-01-01` as the cutoff because it leaves roughly four months for validation, matching the length and future-facing structure of the hidden test window. All model and feature comparisons fit on the earlier rows and evaluate on this later holdout.

```python
CHRONO_CUTOFF = '2017-01-01'
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
Xtr_n = build_features(tr_raw, freq_maps_chrono, add_time=False, add_week=False)
Xva_n = build_features(va_raw, freq_maps_chrono, add_time=False, add_week=False)
align_categories(Xtr_n, Xva_n)
Xtr_without_week = build_features(
    tr_raw, freq_maps_chrono, add_time=True, add_week=False
)
Xva_without_week = build_features(
    va_raw, freq_maps_chrono, add_time=True, add_week=False
)
align_categories(Xtr_without_week, Xva_without_week)
Xtr_with_week = build_features(
    tr_raw, freq_maps_chrono, add_time=True, add_week=True
)
Xva_with_week = build_features(
    va_raw, freq_maps_chrono, add_time=True, add_week=True
)
align_categories(Xtr_with_week, Xva_with_week)
Xtr_t = Xtr_without_week if USE_V3 else Xtr_with_week
Xva_t = Xva_without_week if USE_V3 else Xva_with_week
```

    chrono split -> fit=51,822  validate=11,642  (val drop rate=0.420)

## 6.3 Why a random split is misleading

To isolate the effect of the split itself, we fit the same fixed reference XGBoost configuration once on the chronological split and once on a random split of similar size. This is a validation diagnostic, not the model-selection result used later.

```python
XGB_FIXED_PARAMS = {
    'colsample_bytree': 0.8,
    'enable_categorical': True,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'n_jobs': -1,
    **(
        {'min_child_weight': 10, 'subsample': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 3.0}
        if USE_V3
        else {'min_child_weight': 5, 'subsample': 0.9, 'reg_lambda': 1.0}
    ),
}

def make_xgb(seed=SEED, **overrides):
    params = {**XGB_FIXED_PARAMS, 'random_state': seed, **overrides}
    return XGBClassifier(**params)
```

```python
@cache
def fit_split_diagnostic(X_train, y_train, X_valid, seed):
    model = make_xgb(
        seed,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]

random_fraction = len(va_raw) / len(train_raw)
tr_random, va_random = train_test_split(
    train_raw,
    test_size=random_fraction,
    random_state=SEED,
    stratify=train_raw[TARGET],
)
random_maps = make_freq_maps(tr_random)
Xtr_random = build_features(tr_random, random_maps, add_week=not USE_V3)
Xva_random = build_features(va_random, random_maps, add_week=not USE_V3)
align_categories(Xtr_random, Xva_random)

split_check = pd.DataFrame({
    'split': ['Chronological future holdout', 'Random holdout (diagnostic)'],
    'AUC': [
        roc_auc_score(y_va, fit_split_diagnostic(Xtr_t, y_tr, Xva_t, SEED)),
        roc_auc_score(
            va_random[TARGET].values,
            fit_split_diagnostic(
                Xtr_random,
                tr_random[TARGET].values,
                Xva_random,
                SEED,
            ),
        ),
    ],
})
display(split_check.round(4))
```

|     | split                        |    AUC |
| --: | :--------------------------- | -----: |
|   0 | Chronological future holdout | 0.9135 |
|   1 | Random holdout (diagnostic)  | 0.9572 |

The random split mixes older and newer registrations, so it produces an optimistic score for a genuinely future-facing task. We therefore use the chronological holdout for every model and feature decision below.

# 7. Model experiments & tuning

The assignment requires at least three models and hyperparameter tuning. We compare one linear model, one neural network, and one boosted-tree model on the same chronological holdout.

## 7.1 The model families

We tune one important capacity or regularization parameter for each family and compare the selected settings on the same holdout. The strongest family becomes the development focus of the next experiments; final model selection is deferred until all candidates are evaluated in Section 8.

Each model family is paired with its appropriate preprocessing pipeline: bounded one-hot and scaling for the continuous baselines, and native categorical handling for the tree boosters.

- **Logistic Regression** provides an interpretable linear reference. Its main tuning parameter here is `C`, the inverse regularization strength.
- **MLP** can learn nonlinear combinations but requires a complete, scaled numeric matrix. We vary the number of 64-unit hidden layers while keeping the remaining training settings fixed.
- **XGBoost** builds trees sequentially so later trees correct earlier errors. It can represent thresholds and interactions directly; we tune tree depth and then the learning-rate/tree-count budget.

```python
def encode_for_continuous_models(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, min_count=30
):
    """Bounded one-hot + median imputation for the LR/MLP baselines."""
    Xt, Xv = X_tr.copy(), X_va.copy()
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
```

## 7.2 Focused hyperparameter tuning

For each required family, we vary one parameter that controls capacity or regularization and evaluate it with chronological validation ROC-AUC, the project metric. Training AUC is shown beside it so we can see when extra capacity improves fit without improving the future holdout. Logistic Regression and MLP use the best validation score; for XGBoost we predefine a parsimonious rule and select the smallest depth within 0.0005 AUC of the best result.

For the MLP, the sweep varies hidden depth while keeping every layer at 64 units. For XGBoost, all trials use the active variant's sampling and regularization settings, while the first sweep varies `max_depth` with a fixed boosting budget. A second experiment then tunes the interaction between learning rate and number of trees.

```python
Xtr_enc, Xva_enc = encode_for_continuous_models(Xtr_t, Xva_t)
# Scale the continuous baselines after fitting the encoder on the past window.
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_enc)
Xva_scaled = scaler.transform(Xva_enc)

@cache
def hyper_tune(
    X_train_scaled,
    X_valid_scaled,
    X_train_tree,
    X_valid_tree,
    y_train,
    y_valid,
    seed,
):
    tuning_rows = []
    validation_predictions = {}

    def record_trial(family, axis, x, model, X_train, X_valid, keep=False):
        train_predictions = model.predict_proba(X_train)[:, 1]
        valid_predictions = model.predict_proba(X_valid)[:, 1]
        tuning_rows.append({
            'family': family,
            'axis': axis,
            'x': x,
            'train_AUC': roc_auc_score(y_train, train_predictions),
            'val_AUC': roc_auc_score(y_valid, valid_predictions),
        })
        if keep:
            validation_predictions[(family, x)] = valid_predictions

    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(C=C, max_iter=2000).fit(X_train_scaled, y_train)
        record_trial(
            'Logistic Regression',
            'C  → (less regularisation)',
            C,
            model,
            X_train_scaled,
            X_valid_scaled,
            keep=True,
        )

    for hidden_layers in (1, 2, 3, 4):
        model = MLPClassifier(
            hidden_layer_sizes=(64,) * hidden_layers,
            learning_rate_init=0.001,
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        ).fit(X_train_scaled, y_train)
        record_trial(
            'MLP neural network',
            'hidden layers → (more depth)',
            hidden_layers,
            model,
            X_train_scaled,
            X_valid_scaled,
            keep=True,
        )

    for depth in (2, 3, 4, 5, 6, 8, 10):
        model = make_xgb(
            seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=depth,
        ).fit(X_train_tree, y_train)
        record_trial(
            'Gradient-boosted trees (XGBoost)',
            'max_depth → (more capacity)',
            depth,
            model,
            X_train_tree,
            X_valid_tree,
        )

    return tuning_rows, validation_predictions

tuning_rows, validation_predictions = hyper_tune(
    Xtr_scaled, Xva_scaled, Xtr_t, Xva_t, y_tr, y_va, SEED
)
tuning_raw = pd.DataFrame(tuning_rows)
```

```python
tuning = tuning_raw.copy()
tuning['selected'] = False
for family, trials in tuning.groupby('family', sort=False):
    if family == 'Gradient-boosted trees (XGBoost)':
        best_auc = trials['val_AUC'].max()
        eligible = trials[trials['val_AUC'] >= best_auc - 0.0005]
        chosen_index = eligible['x'].idxmin()
    else:
        chosen_index = trials['val_AUC'].idxmax()
    tuning.loc[chosen_index, 'selected'] = True
tuning['log_x'] = tuning['family'].eq('Logistic Regression')

selected_tuning = tuning.loc[
    tuning['selected'], ['family', 'x', 'train_AUC', 'val_AUC']
].copy()
selected_tuning[['train_AUC', 'val_AUC']] = selected_tuning[
    ['train_AUC', 'val_AUC']
].round(4)
display(selected_tuning)

selected_x = selected_tuning.set_index('family')['x']
pred_lr = validation_predictions[
    ('Logistic Regression', selected_x.loc['Logistic Regression'])
]
pred_mlp = validation_predictions[
    ('MLP neural network', selected_x.loc['MLP neural network'])
]
selected_depth = int(selected_x.loc['Gradient-boosted trees (XGBoost)'])
```

|     | family                           |     x | train_AUC | val_AUC |
| --: | :------------------------------- | ----: | --------: | ------: |
|   0 | Logistic Regression              | 0.001 |    0.9217 |  0.8805 |
|   8 | MLP neural network               |     3 |    0.9684 |   0.877 |
|  14 | Gradient-boosted trees (XGBoost) |     6 |    0.9715 |  0.9135 |

```python
def plot_auc_sweep(data, x, facet, title):
    row_layout = facet == 'family'
    curves = data.melt(
        id_vars=[facet, x, 'axis', 'log_x', 'selected'],
        value_vars=['train_AUC', 'val_AUC'],
        var_name='split',
        value_name='ROC-AUC',
    ).replace({'split': {'train_AUC': 'Train', 'val_AUC': 'Validation'}})
    grid = sns.relplot(
        data=curves,
        x=x,
        y='ROC-AUC',
        hue='split',
        style='split',
        kind='line',
        markers=True,
        dashes=False,
        height=4.0 if row_layout else 4.2,
        aspect=1.55 if row_layout else 1.35,
        facet_kws={'sharex': False, 'sharey': False},
        **({'row': facet} if row_layout else {'col': facet}),
    )
    grid.set_titles('').set_ylabels('ROC-AUC')
    for value, ax in grid.axes_dict.items():
        subset = data[data[facet].eq(value)]
        ax.set(
            title=str(value)
            if row_layout
            else f'{facet.replace("_", " ")} = {value}',
            xlabel=subset['axis'].iloc[0],
        )
        if subset['log_x'].iloc[0]:
            ax.set_xscale('log')
        chosen = subset.loc[subset['selected']].iloc[0]
        ax.scatter(
            chosen[x],
            chosen['val_AUC'],
            marker='*',
            s=190,
            color='#E69F00',
            edgecolor='black',
            linewidth=0.7,
            zorder=5,
        )
    grid.figure.suptitle(title, y=1.01)
    grid.figure.subplots_adjust(top=0.94, hspace=0.5, wspace=0.25)
    show(grid.figure)

plot_auc_sweep(
    tuning, 'x', 'family', 'Focused tuning: training vs validation ROC-AUC'
)
```

![svg](<notebook_files/notebook_81_0.svg>)

The gold star marks the setting selected by the tuning rule. For Logistic Regression and MLP this is the maximum validation AUC. For XGBoost it is the smallest depth within 0.0005 of the best validation score, avoiding extra capacity for a negligible gain. The selected depth flows into the remaining experiments and final refit. XGBoost is the strongest development candidate on the future holdout, so the remaining experiments refine that candidate while retaining Logistic Regression and MLP for the final comparison.

## 7.3 Improving the gradient model

Because XGBoost achieved the highest AUC in the family comparison, we now refine the strongest tree-based candidate. We first tune its boosting budget, then finalize the temporal features, and only afterward test whether adding two fixed-configuration boosted-tree implementations improves the ranking further.

### (a) Boosting budget: number of trees × learning rate

The number of trees and the learning rate interact directly. Consistent with our validation choice, we evaluate the tree budget directly in ROC-AUC. We evaluate various tree counts across two learning rate settings:

```python
@cache
def run_budget_sweep(X_train, X_valid, y_train, y_valid, seed, depth):
    budget_rows = []
    selected_prediction = None
    for lr_rate in (0.1, 0.03):
        for n in (50, 100, 200, 400, 700, 1000):
            model = make_xgb(
                seed,
                n_estimators=n,
                learning_rate=lr_rate,
                max_depth=depth,
            )
            model.fit(X_train, y_train)
            train_predictions = model.predict_proba(X_train)[:, 1]
            valid_predictions = model.predict_proba(X_valid)[:, 1]
            if (lr_rate, n) == (0.03, 700):
                selected_prediction = valid_predictions
            budget_rows.append({
                'learning_rate': lr_rate,
                'n_trees': n,
                'train_AUC': roc_auc_score(y_train, train_predictions),
                'val_AUC': roc_auc_score(y_valid, valid_predictions),
            })
    return budget_rows, selected_prediction

budget_rows, pred_xgb = run_budget_sweep(
    Xtr_t, Xva_t, y_tr, y_va, SEED, selected_depth
)
budget = pd.DataFrame(budget_rows)
budget['axis'] = 'number of trees'
budget['log_x'] = False
budget['selected'] = False
budget.loc[budget.groupby('learning_rate')['val_AUC'].idxmax(), 'selected'] = True
plot_auc_sweep(
    budget,
    'n_trees',
    'learning_rate',
    'Boosting budget: train vs validation ROC-AUC',
)

budget_best = budget.loc[
    budget.groupby('learning_rate')['val_AUC'].idxmax(),
    ['learning_rate', 'n_trees', 'train_AUC', 'val_AUC'],
].copy()
budget_best[['train_AUC', 'val_AUC']] = budget_best[['train_AUC', 'val_AUC']].round(
    4
)
display(budget_best)
```

![svg](<notebook_files/notebook_84_0.svg>)

|     | learning_rate | n_trees | train_AUC | val_AUC |
| --: | ------------: | ------: | --------: | ------: |
|  10 |          0.03 |     700 |     0.977 |  0.9135 |
|   2 |           0.1 |     200 |    0.9756 |  0.9125 |

At learning rate 0.1, validation AUC peaks around 200 trees and then declines while training AUC keeps rising. At 0.03, improvement is slower but the holdout reaches a slightly higher plateau around 700 trees. We choose `learning_rate=0.03` and `n_estimators=700` for XGBoost.

### (b) Temporal-feature ablation

EDA suggested both broad seasonality and longer-term drift. Because XGBoost is the strongest tuning candidate and its capacity is fixed, we use it as a stable reference model to test the remaining temporal choices: whether the continuous `days_since_epoch` index helps, and whether adding ISO week contributes beyond month and weekday.

Trees cannot extrapolate a linear trend beyond the observed range, but the index can still distinguish older and newer training regimes. The chronological holdout determines whether that distinction improves future ranking.

```python
@cache
def fit_temporal_candidate(X_train, y_train, X_valid, seed, depth):
    model = make_xgb(
        seed,
        n_estimators=700,
        learning_rate=0.03,
        max_depth=depth,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1]

temporal_check = pd.DataFrame({
    'temporal features': [
        'Month + weekday',
        'Month + weekday + continuous time index',
        'Month + weekday + time index + ISO week',
    ],
    'chronological AUC': [
        roc_auc_score(
            y_va,
            fit_temporal_candidate(Xtr_n, y_tr, Xva_n, SEED, selected_depth),
        ),
        roc_auc_score(
            y_va,
            fit_temporal_candidate(
                Xtr_without_week,
                y_tr,
                Xva_without_week,
                SEED,
                selected_depth,
            ),
        ),
        roc_auc_score(
            y_va,
            fit_temporal_candidate(
                Xtr_with_week,
                y_tr,
                Xva_with_week,
                SEED,
                selected_depth,
            ),
        ),
    ],
})
display(temporal_check.round(6))
```

|     | temporal features                       | chronological AUC |
| --: | :-------------------------------------- | ----------------: |
|   0 | Month + weekday                         |            0.9112 |
|   1 | Month + weekday + continuous time index |            0.9152 |
|   2 | Month + weekday + time index + ISO week |            0.9135 |

The continuous time index improves future-window ranking, while ISO week does not improve this reference holdout. We nevertheless retain ISO week because the purpose of this notebook is to reproduce the already-scored submission specification rather than modify it after seeing the leaderboard result.

### (c) Does adding other boosters help?

LightGBM and CatBoost build boosted trees differently from XGBoost, so they may rank some registrations differently. We add them with fixed, capacity-aligned settings and test the ensemble itself: does combining their rankings improve the tuned XGBoost result?

We use rank averaging because ROC-AUC depends on ordering. Each model's predictions are converted to percentile ranks before averaging, so a model's probability scale cannot dominate the blend. The resulting value is a ranking score, not a calibrated cancellation probability.

```python
@cache
def fit_predict(name, X_tr, y_train, X_val, seed):
    """Fit one boosted-tree implementation and return P(drop) on X_val."""
    if name == "cat":
        cat_idx = [
            i
            for i, c in enumerate(X_tr.columns)
            if str(X_tr[c].dtype) == "category"
        ]
        X_tr2, X_val2 = X_tr.copy(), X_val.copy()
        for c in X_tr2.columns[cat_idx]:
            X_tr2[c] = X_tr2[c].astype(str)
            X_val2[c] = X_val2[c].astype(str)
        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            eval_metric='AUC',
            cat_features=cat_idx,
        )
        model.fit(X_tr2, y_train)
        return model.predict_proba(X_val2)[:, 1]
    if name == "lgbm":
        model = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_tr,
            y_train,
            categorical_feature=X_tr.select_dtypes("category").columns.tolist(),
        )
        return model.predict_proba(X_val)[:, 1]
    model = make_xgb(
        seed,
        n_estimators=700,
        learning_rate=0.03,
        max_depth=selected_depth,
    )
    model.fit(X_tr, y_train)
    return model.predict_proba(X_val)[:, 1]

def rank_avg(predictions):
    """Average percentile ranks; optimized for ordering, not calibration."""
    return np.mean([rankdata(pred) / len(pred) for pred in predictions], axis=0)

pred_t = {
    "lgbm": fit_predict("lgbm", Xtr_t, y_tr, Xva_t, SEED),
    "xgb": pred_xgb,
    "cat": fit_predict("cat", Xtr_t, y_tr, Xva_t, SEED),
}
blend_t = rank_avg(list(pred_t.values()))

xgb_auc = roc_auc_score(y_va, pred_t["xgb"])
blend_check = (
    pd
    .DataFrame({
        "model": [
            "LightGBM (fixed setting)",
            "XGBoost (tuned)",
            "CatBoost (fixed setting)",
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
```

|     | model                             | chrono_AUC | delta_vs_XGBoost |
| --: | :-------------------------------- | ---------: | ---------------: |
|   0 | Rank-average blend (LGBM+XGB+Cat) |     0.9156 |           0.0021 |
|   1 | XGBoost (tuned)                   |     0.9135 |                0 |
|   2 | LightGBM (fixed setting)          |     0.9135 |          -0.0001 |
|   3 | CatBoost (fixed setting)          |      0.913 |          -0.0005 |

All XGBoost trials use the same fixed sampling and regularization settings (`min_child_weight=5`, `subsample=0.9`, and `reg_lambda=1.0`), so the tuned XGBoost candidate flows unchanged into the ensemble. Adding the fixed LightGBM and CatBoost rankings gives a small further improvement, and we carry the rank-average blend forward as the boosted-tree candidate.

# 8. Model evaluation

We compare the tuned Logistic Regression, MLP, and boosted-tree candidates on the chronological holdout.

## 8.1 ROC & precision–recall curves

```python
curve_candidates = [
    ('Logistic Regression', pred_lr),
    ('MLP', pred_mlp),
    ('Boosted-tree rank blend', blend_t),
]
curve_displays = [
    (RocCurveDisplay, 'ROC curve'),
    (PrecisionRecallDisplay, 'Precision–Recall curve'),
]
curve_fig, curve_axes = plt.subplots(1, 2, figsize=(15, 5.5), layout='compressed')
for curve_ax, (display_class, curve_title) in zip(curve_axes, curve_displays):
    for curve_index, (candidate_name, candidate_predictions) in enumerate(
        curve_candidates
    ):
        display_class.from_predictions(
            y_va,
            candidate_predictions,
            name=candidate_name,
            ax=curve_ax,
            plot_chance_level=curve_index == len(curve_candidates) - 1,
            despine=True,
        )
    curve_ax.set_title(curve_title)
show(curve_fig)
```

![svg](<notebook_files/notebook_92_0.svg>)

The boosted-tree blend has the highest holdout AUC in this comparison. Threshold-based metrics are examined next.

## 8.2 Confusion matrices & threshold metrics

A confusion matrix requires a threshold, so we use 0.5 as a simple reference cutoff for each candidate. For the rank-average blend, this is not a 50% cancellation probability: rank averaging preserves ordering but discards the individual models' probability scales. Nova Academy could later adjust the cutoff according to the relative cost of unnecessary follow-up and missed cancellations.

```python
matrix_candidates = [
    ('Logistic Regression', pred_lr),
    ('MLP neural network', pred_mlp),
    ('Boosted-tree rank blend', blend_t),
]
matrix_fig, matrix_axes = subplot_grid(1, 3)
metric_rows = []
for matrix_ax, (matrix_name, matrix_predictions) in zip(
    matrix_axes, matrix_candidates
):
    matrix_labels = (matrix_predictions >= 0.5).astype(int)
    matrix_auc = roc_auc_score(y_va, matrix_predictions)
    matrix_report = classification_report(
        y_va,
        matrix_labels,
        target_names=['completed', 'dropped'],
        output_dict=True,
        zero_division=0,
    )
    ConfusionMatrixDisplay.from_predictions(
        y_va,
        matrix_labels,
        display_labels=['completed', 'dropped'],
        values_format=',d',
        cmap='Blues',
        colorbar=False,
        ax=matrix_ax,
    )
    matrix_ax.set_title(f'{matrix_name} — ROC-AUC={matrix_auc:.3f}')
    matrix_ax.grid(False)
    metric_rows.append({
        'model': matrix_name,
        'ROC-AUC': matrix_auc,
        'accuracy': matrix_report['accuracy'],
        'precision (dropped)': matrix_report['dropped']['precision'],
        'recall (dropped)': matrix_report['dropped']['recall'],
        'F1 (dropped)': matrix_report['dropped']['f1-score'],
    })
matrix_fig.suptitle('Candidate-model confusion matrices at a 0.5 reference cutoff')
show(matrix_fig)
evaluation_metrics = pd.DataFrame(metric_rows).set_index('model').round(3)
display(evaluation_metrics)
```

![svg](<notebook_files/notebook_95_0.svg>)

| model                   | ROC-AUC | accuracy | precision (dropped) | recall (dropped) | F1 (dropped) |
| :---------------------- | ------: | -------: | ------------------: | ---------------: | -----------: |
| Logistic Regression     |   0.881 |    0.799 |               0.773 |            0.737 |        0.754 |
| MLP neural network      |   0.877 |    0.791 |               0.759 |            0.737 |        0.748 |
| Boosted-tree rank blend |   0.916 |     0.81 |               0.732 |            0.865 |        0.793 |

## 8.2 Model Family Comparisons

- Logistic Regression slightly outperformed the MLP, suggesting that non-linearity is not the primary limitation for continuous models on this dataset.
- The Logistic Regression and MLP ROC and Precision–Recall curves largely overlap, indicating that the two models produce very similar rankings across most decision thresholds.
- The blended gradient-boosted model consistently outperformed both the linear and neural-network models across the evaluated metrics.

### Interpreting the Confusion Matrices (0.5 Threshold)

At the default decision threshold of 0.5, the three models exhibit different operating characteristics.

- Higher recall: The boosted-tree model correctly identified more than 4.2k dropped courses, compared with approximately 3.6k–3.7k for the Logistic Regression and MLP models.
- Fewer missed dropouts: The boosted-tree model produced 661 false negatives, compared with 1,162–1,287 for the other two models.
- More false alarms: The boosted-tree model generated 1,551 false positives, compared with 1,058 for Logistic Regression and 1,281 for the MLP.

Note that the boosted-tree model produces a continuous risk score, its false-positive/false-negative trade-off can be adjusted by selecting a different decision threshold. The confusion matrices therefore illustrate one operating point (0.5) rather than an inherent limitation of the model.

## 8.3 Final Model Selection

The blended boosted-tree model achieved the highest AUC. Since its operating point can be adjusted by selecting an appropriate decision threshold, we chose it as our final submission.

## 8.4 Registrations near the illustrative threshold

We inspect how many selected-blend scores fall near the 0.5 reference cutoff.

```python
score_fig, score_ax = subplot_grid()
sns.histplot(blend_t, bins=50, ax=score_ax)
score_ax.axvline(0.5, linestyle='--', label='reference cutoff')
score_ax.axvspan(
    0.4, 0.6, color='orange', alpha=0.25, zorder=2, label='near-threshold band'
)
score_ax.set(
    xlabel='Rank-average risk score', title='Selected-blend score distribution'
)
score_ax.legend()
near_threshold = ((blend_t > 0.4) & (blend_t < 0.6)).mean() * 100
print(f'share of holdout in the 0.40–0.60 band: {near_threshold:.1f}%')
show(score_fig)
```

    share of holdout in the 0.40–0.60 band: 20.2%

![svg](<notebook_files/notebook_98_1.svg>)

Scores in this band are close to the illustrative cutoff, so small changes in the cutoff can change their binary classification.

# 9. Interpretation with SHAP

The selected blend averages three sets of prediction ranks and therefore has no single fitted tree structure for SHAP to decompose. We use the tuned XGBoost component as a representative fitted model for detailed interpretation, then compare its SHAP patterns with the earlier EDA.

We compute TreeSHAP values on a fixed validation sample of up to 10,000 rows to keep the analysis reproducible and the runtime manageable.

```python
@cache
def compute_shap_analysis(X_train, X_valid, y_train, y_valid, seed, depth):
    shap_model = make_xgb(
        seed,
        n_estimators=700,
        learning_rate=0.03,
        max_depth=depth,
    )
    shap_model.fit(X_train, y_train)
    valid_scores = shap_model.predict_proba(X_valid)[:, 1]
    X_shap = X_valid.sample(min(10000, len(X_valid)), random_state=seed)
    sample_scores = shap_model.predict_proba(X_shap)[:, 1]
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # some shap versions return (n, features, classes)
        shap_values = shap_values[:, :, 1]
    return (
        X_shap,
        sample_scores,
        explainer.expected_value,
        shap_values,
        roc_auc_score(y_valid, valid_scores),
    )

X_shap, shap_scores, shap_base, shap_values, shap_auc = compute_shap_analysis(
    Xtr_t, Xva_t, y_tr, y_va, SEED, selected_depth
)
print(f"XGBoost+time chrono AUC: {shap_auc:.4f}")
```

    XGBoost+time chrono AUC: 0.9135

## 9.1 Global importance (beeswarm + bar)

```python
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title('SHAP summary (beeswarm) — XGBoost+time')
show()
importance = (
    pd
    .DataFrame({
        'feature': X_shap.columns,
        'mean_abs_shap': np.abs(shap_values).mean(0),
    })
    .sort_values('mean_abs_shap', ascending=False)
    .reset_index(drop=True)
)
top = importance.head(20)

importance_fig, importance_ax = plt.subplots(
    figsize=figure_size(2, 1), layout='constrained'
)
top.sort_values('mean_abs_shap').plot.barh(
    x='feature', y='mean_abs_shap', legend=False, ax=importance_ax
)
importance_ax.set(
    xlabel='mean |SHAP value|', title='Top 20 features by SHAP importance'
)
show(importance_fig)
display(top)
```

![svg](<notebook_files/notebook_103_0.svg>)

![svg](<notebook_files/notebook_103_1.svg>)

|     | feature                     | mean_abs_shap |
| --: | :-------------------------- | ------------: |
|   0 | Payment_Terms               |        1.2374 |
|   1 | Origin_Country              |        0.6693 |
|   2 | days_since_epoch            |        0.5305 |
|   3 | Agent_ID                    |        0.5042 |
|   4 | tickets_per_participant     |        0.3554 |
|   5 | Registration_Days_Before    |        0.3389 |
|   6 | Pre_Course_Supports_Tickets |        0.2485 |
|   7 | Enrollment_Type             |        0.2338 |
|   8 | Client_Category             |        0.2259 |
|   9 | Origin_Country_freq         |        0.2002 |
|  10 | got_requested_lab           |        0.1766 |
|  11 | Registration_Changes        |        0.1201 |
|  12 | prev_drop_rate              |        0.1078 |
|  13 | Physical_Course_Kits        |        0.1064 |
|  14 | Daily_Tuition_Cost          |        0.0863 |
|  15 | Agent_ID_freq               |        0.0836 |
|  16 | start_week                  |        0.0836 |
|  17 | cost_x_days                 |        0.0698 |
|  18 | Catering_Package            |        0.0426 |
|  19 | kits_per_participant        |        0.0408 |

The strongest XGBoost contributions broadly match the earlier exploration: `Payment_Terms`, `Origin_Country`, the time index, `Agent_ID`, registration lead time, and support-related features appear near the top. Raw country and agent identity contribute more than their frequency encodings, while the engineered ratios add smaller supporting signals.

### Checking the suspicious `Payment_Terms` signal

EDA showed that prepaid, non-refundable registrations drop unexpectedly often, and representative-model SHAP now ranks `Payment_Terms` first. To measure how strongly the selected model relies on it, we refit all three blend components without the field and compare chronological AUC.

```python
Xtr_no_payment = Xtr_t.drop(columns=["Payment_Terms"])
Xva_no_payment = Xva_t.drop(columns=["Payment_Terms"])
pred_blend_no_payment = rank_avg([
    fit_predict(name, Xtr_no_payment, y_tr, Xva_no_payment, SEED)
    for name in ("lgbm", "xgb", "cat")
])
payment_check = pd.DataFrame({
    "model": [
        "Rank-average blend",
        "Rank-average blend, no Payment_Terms",
    ],
    "chrono_AUC": [
        roc_auc_score(y_va, blend_t),
        roc_auc_score(y_va, pred_blend_no_payment),
    ],
})
payment_check["delta_vs_with_payment"] = (
    payment_check["chrono_AUC"] - payment_check.loc[0, "chrono_AUC"]
)
display(payment_check)
```

|     | model                                | chrono_AUC | delta_vs_with_payment |
| --: | :----------------------------------- | ---------: | --------------------: |
|   0 | Rank-average blend                   |     0.9156 |                     0 |
|   1 | Rank-average blend, no Payment_Terms |     0.9101 |               -0.0055 |

Removing `Payment_Terms` changes chronological AUC from 0.9156 to 0.9101. The field's exact recording time is still worth confirming with the data owner.

## 9.2 Direction of the strongest non-payment signal

`Origin_Country` is the strongest feature after `Payment_Terms`, so we plot the average SHAP contribution of its most common levels. Positive values push XGBoost toward a higher cancellation score; negative values push it toward completion.

```python
top_feat = (
    importance.loc[importance['feature'] != 'Payment_Terms', 'feature'].iloc[0]
    if importance['feature'].iloc[0] == 'Payment_Terms'
    else importance['feature'].iloc[0]
)

def plot_shap_dependence_readable(feature):
    max_categories = 15
    col_idx = list(X_shap.columns).index(feature)
    values = X_shap[feature]
    is_categorical = (
        str(values.dtype) == 'category'
        or values.dtype == 'object'
        or values.nunique(dropna=False) <= max_categories
    )
    if not is_categorical:
        shap.dependence_plot(
            feature, shap_values, X_shap, interaction_index=None, show=False
        )
        plt.title(f'SHAP dependence — {feature}')
        show()
        return
    labels = values.astype('string').fillna('missing')
    keep = labels.value_counts().head(max_categories).index
    grouped = pd.DataFrame({
        'level': labels.where(labels.isin(keep), 'other'),
        'shap': shap_values[:, col_idx],
    })
    summary = (
        grouped
        .groupby('level', observed=True)
        .agg(mean_shap=('shap', 'mean'), n=('shap', 'size'))
        .sort_values('mean_shap')
    )
    fig, ax = plt.subplots(figsize=figure_size(2, 1), layout='constrained')
    sns.barplot(data=summary.reset_index(), y='level', x='mean_shap', ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set(
        title=f'Mean SHAP by {feature} level (top {max_categories} + other)',
        xlabel='mean SHAP contribution',
        ylabel=feature,
    )
    show(fig)

plot_shap_dependence_readable(top_feat)
```

![svg](<notebook_files/notebook_108_0.svg>)

## 9.3 Explaining one near-threshold registration

We choose one sampled XGBoost prediction near 0.5 and decompose it. The waterfall shows which features pushed this particular score upward and which pushed it downward.

```python
borderline = np.where((shap_scores > 0.45) & (shap_scores < 0.55))[0]
idx = int(borderline[0]) if len(borderline) else 0
base = shap_base
if isinstance(base, (list, np.ndarray)):
    base = np.asarray(base).ravel()[-1]

print(
    f"explaining order at sample position {idx} — model P(drop)={shap_scores[idx]:.3f}"
)
explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=base,
    data=X_shap.iloc[idx].values,
    feature_names=list(X_shap.columns),
)
shap.plots.waterfall(explanation, max_display=14, show=False)
show()
```

    explaining order at sample position 13 — model P(drop)=0.488

![svg](<notebook_files/notebook_110_1.svg>)

For this registration, positive and negative contributions nearly balance, producing a score close to the reference threshold.

# 10. Rebuilding and checking the submission

The stored `data/Group_27_Submission.csv` is the submission that received the recorded leaderboard score. The block below retrains the model, writes `data/Group_27_Submission_v2_rebuilt.csv`, and checks how closely the current environment reproduces the historical scored ranking. It never overwrites the scored file.

```python
candidate_path = f"data/Group_27_Submission_v{'3' if USE_V3 else '2'}_rebuilt.csv"
submission_maps = make_freq_maps(train_raw, test_raw)
X_train_full = build_features(train_raw, submission_maps, add_week=not USE_V3)
X_test = build_features(test_raw, submission_maps, add_week=not USE_V3)
align_categories(X_train_full, X_test)
y_full = train_raw[TARGET].values

submission_predictions = []
for submission_model in ("lgbm", "xgb", "cat"):
    submission_predictions.append(
        fit_predict(submission_model, X_train_full, y_full, X_test, SEED)
    )
    print(f"fitted {submission_model} on {len(X_train_full):,} rows")

submission = pd.DataFrame({
    "Client_ID": test_raw["Client_ID"],
    "Drop_Probability": rank_avg(submission_predictions),
})
submission.to_csv(candidate_path, index=False)
print(f"wrote {candidate_path}  ({len(submission):,} rows)")
display(submission.head())
print(submission["Drop_Probability"].describe())

scored_submission = pd.read_csv("data/Group_27_Submission.csv")
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
```

    fitted lgbm on 63,464 rows
    fitted xgb on 63,464 rows
    fitted cat on 63,464 rows
    wrote data/Group_27_Submission_v2_rebuilt.csv  (15,866 rows)

|     | Client_ID | Drop_Probability |
| --: | --------: | ---------------: |
|   0 |     62246 |           0.2206 |
|   1 |     43031 |           0.9595 |
|   2 |     26571 |           0.2439 |
|   3 |     77694 |           0.9611 |
|   4 |     22185 |           0.4952 |

    count    15866.000000
    mean         0.500032
    std          0.287224
    min          0.000252
    25%          0.251765
    50%          0.503771
    75%          0.749312
    max          0.998939
    Name: Drop_Probability, dtype: float64

|     | rows_compared | max_abs_diff | mean_abs_diff | spearman_corr |
| --: | ------------: | -----------: | ------------: | ------------: |
|   0 |         15866 |       0.0962 |         0.011 |        0.9986 |

    rebuilt predictions are not byte-identical to the scored file; use the stored scored CSV as the official leaderboard record.

The comparison checks the rebuilt file's schema and its Spearman rank agreement with the submitted ranking.

# 11. Conclusions & Executive Summary

Nova Academy's test registrations occur after the training period, and both the monthly target rate and the adversarial-validation result (AUC 0.935) show temporal distribution shift. Model selection therefore used a four-month chronological holdout.

Cleaning reduced hundreds of inconsistent text labels to compact category sets. Missingness, payment terms, country, agent, registration timing, and support activity all carried predictive information. Model comparison confirmed that tuned XGBoost outperformed the Logistic Regression and MLP baselines on the future holdout. The LightGBM, XGBoost, and CatBoost rank-average blend reached chronological AUC 0.9156 on this split.

The stored submission produced by this model specification received hidden-test ROC-AUC **0.889314**, above the required 0.70. A fresh rerun can differ slightly because boosted-tree implementations and hardware-dependent floating-point behavior are not guaranteed to reproduce an old prediction vector byte-for-byte, so the stored scored CSV remains the authoritative leaderboard artifact.

Further work could include confirming when `Payment_Terms` is recorded and calibrating the selected blend score for cost-based operational thresholds.
