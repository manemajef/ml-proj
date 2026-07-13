# Group 27 — Course-Drop Prediction (Nova Academy)

**Submitters:** Rotem David Semah (ID: `211396593`) · Ron Drach (ID: `213915499`)

---

This notebook follows the project from understanding the data through preparation, modelling, evaluation, and interpretation.

> **About the notebook's**
>
> The notebook was developed as a `marimo` notebook and exported to Jupyter format for submission.Some code patterns may look wierd (such as wrappers functions for ploting etc) but that ensures compatibility with both `jupyter` and `marimo` format.
>
>Additionaly you may notice expensive functions (such as shap, and hyper tuners) are decorated with `@cache` which as the name suggest - caches the the expensive claulations to keep the notebook runable :)


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
from IPython.display import display

def cache(fn):
    cache_dir = Path('.cache/joblib') / fn.__name__
    source_hash = joblib_hash(dedent(getsource(fn)))

    @wraps(fn)
    def wrapped(*args, **kwargs):
        path = cache_dir / f'{joblib_hash((source_hash, args, kwargs))}.joblib'
        if path.exists():
            return load(path)
        result = fn(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(result, path)
        return result

    return wrapped

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 220
plt.rcParams['svg.fonttype'] = 'path'
pd.set_option('display.max_columns', None)

TRAIN_PATH = 'data/Train_Data.csv'
TEST_PATH = 'data/Test_Data_No_Target.csv'
TARGET = 'Dropped_Course'
SEED = 42

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Course_Start_Date"])
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

# The CSV parser reads these ID fields as numbers because their non-missing
# values look numeric. They are labels, not measurable quantities, so from
# this point onward we keep them as strings.
def cast_id_columns(df):
    for col in ("Agent_ID", "Company_ID"):
        df[col] = df[col].astype("string")

for df in (train_raw, test_raw):
    cast_id_columns(df)
```

    train: 63,464 rows x 29 cols
    test : 15,866 rows x 28 cols



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dtype</th>
      <th>n_missing</th>
      <th>missing_%</th>
      <th>n_unique</th>
      <th>n_zero</th>
      <th>most_frequent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Client_ID</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>63464</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Professionals_Count</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>338</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Students_Count</th>
      <td>float64</td>
      <td>4</td>
      <td>0.01</td>
      <td>5</td>
      <td>59578</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Observers_Count</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>5</td>
      <td>63149</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Course_Start_Date</th>
      <td>datetime64[ns]</td>
      <td>0</td>
      <td>0.00</td>
      <td>666</td>
      <td>0</td>
      <td>2015-10-16 00:00:00</td>
    </tr>
    <tr>
      <th>Practical_Hours</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>18</td>
      <td>30671</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Theory_Hours</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>29</td>
      <td>4045</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Registration_Days_Before</th>
      <td>float64</td>
      <td>2666</td>
      <td>4.20</td>
      <td>423</td>
      <td>2610</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Origin_Country</th>
      <td>object</td>
      <td>557</td>
      <td>0.88</td>
      <td>721</td>
      <td>0</td>
      <td>PRT</td>
    </tr>
    <tr>
      <th>Catering_Package</th>
      <td>object</td>
      <td>407</td>
      <td>0.64</td>
      <td>321</td>
      <td>0</td>
      <td>Standard (Coffee Only)</td>
    </tr>
    <tr>
      <th>Welcome_Gift_Type</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>4</td>
      <td>0</td>
      <td>Branded Notebook</td>
    </tr>
    <tr>
      <th>Requested_Lab_Config</th>
      <td>object</td>
      <td>1736</td>
      <td>2.74</td>
      <td>8</td>
      <td>0</td>
      <td>Standard PC (Windows)</td>
    </tr>
    <tr>
      <th>Assigned_Lab_Config</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>9</td>
      <td>0</td>
      <td>Standard PC (Windows)</td>
    </tr>
    <tr>
      <th>Prev_Course_Dropouts</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>10</td>
      <td>58184</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Prev_Course_Attended</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>62</td>
      <td>62188</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Pre_Course_Supports_Tickets</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>6</td>
      <td>39830</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Physical_Course_Kits</th>
      <td>float64</td>
      <td>1040</td>
      <td>1.64</td>
      <td>4</td>
      <td>60790</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Waiting_List_Days</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>107</td>
      <td>60089</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Registration_Changes</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>19</td>
      <td>55478</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Enrollment_Type</th>
      <td>object</td>
      <td>719</td>
      <td>1.13</td>
      <td>298</td>
      <td>0</td>
      <td>General Admission</td>
    </tr>
    <tr>
      <th>Lanyard_Color</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>240</td>
      <td>0</td>
      <td>Blue</td>
    </tr>
    <tr>
      <th>Client_Category</th>
      <td>object</td>
      <td>0</td>
      <td>0.00</td>
      <td>505</td>
      <td>0</td>
      <td>SaaS &amp; Software Houses</td>
    </tr>
    <tr>
      <th>Submission_Source</th>
      <td>object</td>
      <td>605</td>
      <td>0.95</td>
      <td>328</td>
      <td>0</td>
      <td>B2B Platforms &amp; Resellers</td>
    </tr>
    <tr>
      <th>Returning_Client</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>2</td>
      <td>61742</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Agent_ID</th>
      <td>float64</td>
      <td>11173</td>
      <td>17.61</td>
      <td>203</td>
      <td>0</td>
      <td>184.0</td>
    </tr>
    <tr>
      <th>Company_ID</th>
      <td>float64</td>
      <td>60344</td>
      <td>95.08</td>
      <td>184</td>
      <td>0</td>
      <td>5181.0</td>
    </tr>
    <tr>
      <th>Payment_Terms</th>
      <td>object</td>
      <td>587</td>
      <td>0.92</td>
      <td>236</td>
      <td>0</td>
      <td>Pay Upon Start</td>
    </tr>
    <tr>
      <th>Daily_Tuition_Cost</th>
      <td>float64</td>
      <td>79</td>
      <td>0.12</td>
      <td>4780</td>
      <td>1079</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>Dropped_Course</th>
      <td>int64</td>
      <td>0</td>
      <td>0.00</td>
      <td>2</td>
      <td>37165</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


**What the dictionary tells us.**

- `Client_ID` is unique per row — an identifier, never a feature.
- `Agent_ID` and `Company_ID` were inferred as numeric even though they are identifiers, so we convert them to strings after this first inspection. `Company_ID` is also missing for most rows.
- Several text fields have unexpectedly high cardinality. We inspect their raw values later before deciding whether that reflects real variety or inconsistent spelling.
- The numeric summary below lets us look for suspicious ranges and extreme values.


```python
train_raw.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Client_ID</th>
      <th>Professionals_Count</th>
      <th>Students_Count</th>
      <th>Observers_Count</th>
      <th>Course_Start_Date</th>
      <th>Practical_Hours</th>
      <th>Theory_Hours</th>
      <th>Registration_Days_Before</th>
      <th>Prev_Course_Dropouts</th>
      <th>Prev_Course_Attended</th>
      <th>Pre_Course_Supports_Tickets</th>
      <th>Physical_Course_Kits</th>
      <th>Waiting_List_Days</th>
      <th>Registration_Changes</th>
      <th>Returning_Client</th>
      <th>Daily_Tuition_Cost</th>
      <th>Dropped_Course</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>63460.000000</td>
      <td>63464.000000</td>
      <td>63464</td>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>60798.000000</td>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>62424.000000</td>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>63464.000000</td>
      <td>63385.000000</td>
      <td>63464.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39761.752616</td>
      <td>1.835214</td>
      <td>8.751718</td>
      <td>0.005326</td>
      <td>2016-06-23 05:17:23.287533056</td>
      <td>6.609054</td>
      <td>2.164392</td>
      <td>102.894470</td>
      <td>0.095991</td>
      <td>0.122967</td>
      <td>0.513330</td>
      <td>0.026224</td>
      <td>3.983676</td>
      <td>0.180039</td>
      <td>0.027133</td>
      <td>98.847963</td>
      <td>0.414392</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2015-07-01 00:00:00</td>
      <td>-5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19959.750000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2016-02-13 00:00:00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>75.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39819.500000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2016-07-01 00:00:00</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>65.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59570.250000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2016-11-11 00:00:00</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>117.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79330.000000</td>
      <td>4.000000</td>
      <td>9999.000000</td>
      <td>10.000000</td>
      <td>2017-04-26 00:00:00</td>
      <td>10000.000000</td>
      <td>41.000000</td>
      <td>629.000000</td>
      <td>21.000000</td>
      <td>61.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>391.000000</td>
      <td>21.000000</td>
      <td>1.000000</td>
      <td>5400.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>22878.980699</td>
      <td>0.508607</td>
      <td>294.238584</td>
      <td>0.089662</td>
      <td>NaN</td>
      <td>215.502929</td>
      <td>1.469854</td>
      <td>109.178824</td>
      <td>0.448526</td>
      <td>1.535201</td>
      <td>0.763563</td>
      <td>0.160202</td>
      <td>23.195495</td>
      <td>0.592577</td>
      <td>0.162474</td>
      <td>41.855391</td>
      <td>0.492621</td>
    </tr>
  </tbody>
</table>
</div>



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

def plot_target_balance():
    ax = target_rate.mul(100).plot.bar(color=['#4c72b0', '#c44e52'])
    ax.set_xticklabels(['completed (0)', 'dropped (1)'], rotation=0)
    ax.set_ylabel('share of orders (%)')
    ax.set_title('Target balance — Dropped_Course')
    plt.tight_layout()
    plt.show()

plot_target_balance()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>rate_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0 = completed</th>
      <td>37165</td>
      <td>58.6</td>
    </tr>
    <tr>
      <th>1 = dropped</th>
      <td>26299</td>
      <td>41.4</td>
    </tr>
  </tbody>
</table>
</div>



    
![svg](notebook_files/notebook_9_1.svg)
    


The classes are reasonably balanced: about 59% completed and 41% dropped

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

def plot_monthly_drop_rate():
    ax = monthly.plot(marker='o', figsize=(12, 4))
    ax.axhline(
        train_raw[TARGET].mean() * 100,
        ls='--',
        color='grey',
        label='train average',
    )
    ax.axvline(
        train_end,
        ls='--',
        color='green',
        label=f'train ends ({train_end.date()})',
    )
    ax.axvline(
        test_end, ls=':', color='red', label=f'test ends ({test_end.date()})'
    )
    ax.set_xlim(train_raw['Course_Start_Date'].min(), test_end)
    ax.set_ylabel('drop rate (%)')
    ax.set_title(
        'Drop rate over time — training period and the hidden test horizon'
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_monthly_drop_rate()
```

    train dates: 2015-07-01 -> 2017-04-26
    test  dates: 2017-04-26 -> 2017-08-31



    
![svg](notebook_files/notebook_13_1.svg)
    


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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train_missing_%</th>
      <th>test_missing_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Company_ID</th>
      <td>95.08</td>
      <td>96.41</td>
    </tr>
    <tr>
      <th>Agent_ID</th>
      <td>17.61</td>
      <td>17.61</td>
    </tr>
    <tr>
      <th>Registration_Days_Before</th>
      <td>4.20</td>
      <td>4.05</td>
    </tr>
    <tr>
      <th>Requested_Lab_Config</th>
      <td>2.74</td>
      <td>3.01</td>
    </tr>
    <tr>
      <th>Physical_Course_Kits</th>
      <td>1.64</td>
      <td>1.43</td>
    </tr>
    <tr>
      <th>Enrollment_Type</th>
      <td>1.13</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>Submission_Source</th>
      <td>0.95</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>Payment_Terms</th>
      <td>0.92</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>Origin_Country</th>
      <td>0.88</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>Catering_Package</th>
      <td>0.64</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>Daily_Tuition_Cost</th>
      <td>0.12</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>Students_Count</th>
      <td>0.01</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


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

def summarize_missingness():
    rows = []
    for col in missingness_cols:
        stats = (
            train_raw
            .assign(is_missing=train_raw[col].isna())
            .groupby('is_missing')[TARGET]
            .agg(count='size', drop_rate='mean')
        )
        for is_missing, row in stats.iterrows():
            rows.append({
                'column': col,
                'is_missing': is_missing,
                'count': int(row['count']),
                'drop_rate_%': round(row['drop_rate'] * 100, 1),
            })
    return pd.DataFrame(rows)

display(summarize_missingness())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>is_missing</th>
      <th>count</th>
      <th>drop_rate_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Company_ID</td>
      <td>False</td>
      <td>3120</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Company_ID</td>
      <td>True</td>
      <td>60344</td>
      <td>42.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Agent_ID</td>
      <td>False</td>
      <td>52291</td>
      <td>43.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agent_ID</td>
      <td>True</td>
      <td>11173</td>
      <td>33.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Registration_Days_Before</td>
      <td>False</td>
      <td>60798</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Registration_Days_Before</td>
      <td>True</td>
      <td>2666</td>
      <td>41.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Physical_Course_Kits</td>
      <td>False</td>
      <td>62424</td>
      <td>41.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Physical_Course_Kits</td>
      <td>True</td>
      <td>1040</td>
      <td>39.3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Daily_Tuition_Cost</td>
      <td>False</td>
      <td>63385</td>
      <td>41.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Daily_Tuition_Cost</td>
      <td>True</td>
      <td>79</td>
      <td>51.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Payment_Terms</td>
      <td>False</td>
      <td>62877</td>
      <td>41.5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Payment_Terms</td>
      <td>True</td>
      <td>587</td>
      <td>35.4</td>
    </tr>
  </tbody>
</table>
</div>


Rows without a `Company_ID` have a noticeably higher drop rate, and `Agent_ID` presence also separates groups. This motivates explicit presence flags instead of replacing missing identifiers with a typical value.

## 3.3 Inspecting categorical values

Several text columns have far more distinct values than their meanings suggest: hundreds of payment terms, colors, and enrollment types would be surprising. We inspect the raw labels before deciding whether the cardinality is real.


```python
TEXT_COLS = list(train_raw.select_dtypes(include=['object']).columns)

def count_unique_vals(df, col):
    return df[col].nunique()

def most_common_cats(df, col, n=15):
    return df[col].value_counts(normalize=True).head(n)

N_COUNT = 8

def print_category_summaries():
    for col in TEXT_COLS:
        top_values = most_common_cats(train_raw, col, N_COUNT)
        cats = [
            f'{value!r}: ({share * 100:.1f}%)'
            for value, share in top_values.items()
        ]
        cats_str = '\n'.join(
            (' | '.join(cats[i : i + 3]) for i in range(0, len(cats), 3))
        )
        print(
            f"\n{'=' * 80}\n\nColumn: {col}\nUnique values: {count_unique_vals(train_raw, col)}\n\nTop {len(top_values)} categories:\n\n{cats_str}\n"
        )

print_category_summaries()
```

    
    ================================================================================
    
    Column: Origin_Country
    Unique values: 721
    
    Top 8 categories:
    
    'PRT': (38.6%) | 'FRA': (10.2%) | 'DEU': (6.4%)
    'ESP': (5.7%) | 'GBR': (5.2%) | 'ITA': (4.0%)
    'BRA': (2.0%) | 'BEL': (2.0%)
    
    
    ================================================================================
    
    Column: Catering_Package
    Unique values: 321
    
    Top 8 categories:
    
    'Standard (Coffee Only)': (71.9%) | 'No Food Plan': (10.5%) | 'Lunch Included': (7.5%)
    'standard (coffee only)': (1.8%) | 'STANDARD (COFFEE ONLY)': (1.7%) | ' Standard (Coffee Only)  ': (0.8%)
    '  Standard (Coffee Only) ': (0.8%) | ' Standard (Coffee Only) ': (0.8%)
    
    
    ================================================================================
    
    Column: Welcome_Gift_Type
    Unique values: 4
    
    Top 4 categories:
    
    'Branded Notebook': (50.8%) | 'Water Bottle': (29.0%) | 'USB Drive': (16.0%)
    'Portable Charger': (4.2%)
    
    
    ================================================================================
    
    Column: Requested_Lab_Config
    Unique values: 8
    
    Top 8 categories:
    
    'Standard PC (Windows)': (80.5%) | 'Linux Workstation': (13.6%) | 'Dual Monitor Setup': (2.1%)
    'MacOS Station': (1.6%) | 'Laptop Docking Station': (1.6%) | 'High-GPU Unit': (0.5%)
    'Touch Screen Interface': (0.0%) | 'VR/AR Station': (0.0%)
    
    
    ================================================================================
    
    Column: Assigned_Lab_Config
    Unique values: 9
    
    Top 8 categories:
    
    'Standard PC (Windows)': (72.4%) | 'Linux Workstation': (18.4%) | 'Laptop Docking Station': (2.9%)
    'MacOS Station': (2.5%) | 'Dual Monitor Setup': (2.5%) | 'High-GPU Unit': (0.8%)
    'Server Access Terminal': (0.4%) | 'Touch Screen Interface': (0.2%)
    
    
    ================================================================================
    
    Column: Enrollment_Type
    Unique values: 298
    
    Top 8 categories:
    
    'General Admission': (64.6%) | 'Affiliated Admission': (21.6%) | 'Contractual Agreement': (3.2%)
    'general admission': (1.6%) | 'GENERAL ADMISSION': (1.6%) | ' General Admission  ': (0.8%)
    ' General Admission ': (0.7%) | '  General Admission ': (0.7%)
    
    
    ================================================================================
    
    Column: Lanyard_Color
    Unique values: 240
    
    Top 8 categories:
    
    'Blue': (49.6%) | 'Black': (21.0%) | 'Red': (10.1%)
    'Orange': (5.2%) | 'Green': (3.9%) | 'BLUE': (1.2%)
    'blue': (1.2%) | '  Blue  ': (0.6%)
    
    
    ================================================================================
    
    Column: Client_Category
    Unique values: 505
    
    Top 8 categories:
    
    'SaaS & Software Houses': (41.4%) | 'Traditional IT & Telecomm': (20.4%) | 'Big Tech & Multinationals': (16.8%)
    'FinTech & Banking': (6.6%) | 'Industrial Tech & IoT': (3.7%) | 'saas & software houses': (1.1%)
    'SAAS & SOFTWARE HOUSES': (1.0%) | 'Non-Profit & EduTech': (0.7%)
    
    
    ================================================================================
    
    Column: Submission_Source
    Unique values: 328
    
    Top 8 categories:
    
    'B2B Platforms & Resellers': (77.4%) | 'Direct Website Registration': (7.4%) | 'Dedicated Sales Team': (4.1%)
    'B2B PLATFORMS & RESELLERS': (2.0%) | 'b2b platforms & resellers': (1.9%) | ' B2B Platforms & Resellers  ': (0.9%)
    '  B2B Platforms & Resellers ': (0.9%) | ' B2B Platforms & Resellers ': (0.8%)
    
    
    ================================================================================
    
    Column: Payment_Terms
    Unique values: 236
    
    Top 8 categories:
    
    'Pay Upon Start': (73.8%) | 'Prepaid (Non-Refundable)': (15.3%) | 'PAY UPON START': (1.9%)
    'pay upon start': (1.8%) | ' Pay Upon Start ': (0.9%) | '  Pay Upon Start  ': (0.8%)
    ' Pay Upon Start  ': (0.8%) | '  Pay Upon Start ': (0.8%)
    


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
COUNTRY_ALIASES = {'cn': 'chn'}

def canonicalize(s: pd.Series) -> pd.Series:
    """Map dirty categorical text to its canonical form: lowercase, injected
    punctuation stripped, single-spaced. No NaN masking — that is normalize_cats' job."""
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>raw_unique</th>
      <th>clean_unique</th>
      <th>collapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Origin_Country</th>
      <td>721</td>
      <td>153</td>
      <td>568</td>
    </tr>
    <tr>
      <th>Client_Category</th>
      <td>505</td>
      <td>7</td>
      <td>498</td>
    </tr>
    <tr>
      <th>Submission_Source</th>
      <td>328</td>
      <td>4</td>
      <td>324</td>
    </tr>
    <tr>
      <th>Catering_Package</th>
      <td>321</td>
      <td>4</td>
      <td>317</td>
    </tr>
    <tr>
      <th>Enrollment_Type</th>
      <td>298</td>
      <td>4</td>
      <td>294</td>
    </tr>
    <tr>
      <th>Lanyard_Color</th>
      <td>240</td>
      <td>5</td>
      <td>235</td>
    </tr>
    <tr>
      <th>Payment_Terms</th>
      <td>236</td>
      <td>3</td>
      <td>233</td>
    </tr>
    <tr>
      <th>Welcome_Gift_Type</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Requested_Lab_Config</th>
      <td>8</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Assigned_Lab_Config</th>
      <td>9</td>
      <td>9</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


The before/after table confirms that most of the apparent variety was formatting noise: `Payment_Terms` falls from 236 raw labels to 3 cleaned levels, and `Client_Category` from 505 to 7. Columns that were already consistent remain unchanged.

## 3.4 Which categories actually relate to dropping?

We start with business fields that have only a few cleaned levels, where a direct plot remains readable. Country and identifiers need separate treatment because hundreds of levels would make the same plot misleading.


```python
def plot_dropout_by_category(df, col, min_count=50, top_n=10, ax=None):
    stats = df.groupby(col, dropna=False)[TARGET].agg(
        drop_rate='mean', count='size'
    )
    stats = (
        stats[stats['count'] >= min_count]
        .sort_values('count', ascending=False)
        .head(top_n)
        .sort_values('drop_rate')
    )
    labels = [f"{i} (n={int(r['count'])})" for i, r in stats.iterrows()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, stats['drop_rate'] * 100, color='#4c72b0')
    overall = df[TARGET].mean() * 100
    ax.axvline(overall, ls='--', color='red', label=f'mean ({overall:.1f}%)')
    ax.set_xlabel('drop rate (%)')
    ax.set_title(f'Drop rate by {col}')
    ax.legend()
    return stats

def plot_categorical_overview():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_dropout_by_category(
        clean_train, 'Payment_Terms', min_count=20, top_n=5, ax=axes[0, 0]
    )
    plot_dropout_by_category(
        clean_train, 'Client_Category', min_count=100, top_n=8, ax=axes[0, 1]
    )
    plot_dropout_by_category(
        clean_train, 'Submission_Source', min_count=100, top_n=6, ax=axes[1, 0]
    )
    plot_dropout_by_category(
        clean_train, 'Enrollment_Type', min_count=100, top_n=6, ax=axes[1, 1]
    )
    plt.tight_layout()
    plt.show()

plot_categorical_overview()
```


    
![svg](notebook_files/notebook_29_0.svg)
    


The clearest surprise is `Payment_Terms`: prepaid, non-refundable registrations drop more often than pay-on-start registrations, the opposite of what we expected. Two possible explanations are that these terms are assigned to riskier deals in advance or that the field is updated later in the registration process. We return to this question after fitting the model.

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
    colors = np.where(stats['lift_pp'] >= 0, '#c44e52', '#4c72b0')
    ax.barh(labels, stats['drop_rate_pct'], color=colors)
    ax.axvline(
        overall_drop * 100,
        ls='--',
        color='black',
        lw=1,
        label=f'overall ({overall_drop * 100:.1f}%)',
    )
    ax.set_xlabel('drop rate (%)')
    ax.set_title(title)
    ax.legend()

def plot_country_overview():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_country_dropout(
        top_by_size, f'Drop rate by largest {country_top_n} countries', axes[0]
    )
    plot_country_dropout(
        extreme_by_lift,
        f'Most unusual country drop rates (n >= {country_min_n})',
        axes[1],
    )
    plt.tight_layout()
    plt.show()

plot_country_overview()
display(
    country_stats
    .sort_values('count', ascending=False)
    .head(country_top_n)[['count', 'drop_rate_pct', 'lift_pp']]
    .round(2)
)  # ignore countries with too few rows for a stable rate  # sort by distance from the overall drop rate
```


    
![svg](notebook_files/notebook_32_0.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>drop_rate_pct</th>
      <th>lift_pp</th>
    </tr>
    <tr>
      <th>Origin_Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prt</th>
      <td>26429</td>
      <td>63.78</td>
      <td>22.34</td>
    </tr>
    <tr>
      <th>fra</th>
      <td>6961</td>
      <td>17.28</td>
      <td>-24.16</td>
    </tr>
    <tr>
      <th>deu</th>
      <td>4400</td>
      <td>16.70</td>
      <td>-24.73</td>
    </tr>
    <tr>
      <th>esp</th>
      <td>3896</td>
      <td>27.31</td>
      <td>-14.13</td>
    </tr>
    <tr>
      <th>gbr</th>
      <td>3514</td>
      <td>27.80</td>
      <td>-13.64</td>
    </tr>
    <tr>
      <th>ita</th>
      <td>2726</td>
      <td>35.88</td>
      <td>-5.56</td>
    </tr>
    <tr>
      <th>bra</th>
      <td>1402</td>
      <td>38.02</td>
      <td>-3.42</td>
    </tr>
    <tr>
      <th>bel</th>
      <td>1324</td>
      <td>19.18</td>
      <td>-22.25</td>
    </tr>
    <tr>
      <th>nld</th>
      <td>1222</td>
      <td>19.97</td>
      <td>-21.47</td>
    </tr>
    <tr>
      <th>usa</th>
      <td>1072</td>
      <td>22.39</td>
      <td>-19.05</td>
    </tr>
    <tr>
      <th>chn</th>
      <td>1054</td>
      <td>42.79</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>che</th>
      <td>935</td>
      <td>22.78</td>
      <td>-18.66</td>
    </tr>
  </tbody>
</table>
</div>


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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>drop_rate_pct</th>
    </tr>
    <tr>
      <th>country_group</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Other countries</th>
      <td>37035</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>Portugal</th>
      <td>26429</td>
      <td>63.8</td>
    </tr>
  </tbody>
</table>
</div>


Compared with all other countries, Portugal remains clearly different. We next inspect the identifier fields as categories, not numbers, to see whether they show related structure.


```python
company_presence = train_raw.groupby(train_raw['Company_ID'].notna())[TARGET].agg(
    count='size', drop_rate='mean'
)
company_presence.index = ['no company_id', 'has company_id']

def plot_agent_company_overview():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_dropout_by_category(
        clean_train, 'Agent_ID', min_count=150, top_n=12, ax=axes[0]
    )
    axes[1].bar(
        company_presence.index,
        company_presence['drop_rate'] * 100,
        color=['#c44e52', '#55a868'],
    )
    axes[1].set_ylabel('drop rate (%)')
    axes[1].set_title('Drop rate by Company_ID presence')
    plt.tight_layout()
    plt.show()

plot_agent_company_overview()
display(company_presence)
```


    
![svg](notebook_files/notebook_36_0.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>drop_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no company_id</th>
      <td>60344</td>
      <td>0.424848</td>
    </tr>
    <tr>
      <th>has company_id</th>
      <td>3120</td>
      <td>0.212179</td>
    </tr>
  </tbody>
</table>
</div>


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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>check</th>
      <th>accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>majority country baseline</td>
      <td>0.391</td>
    </tr>
    <tr>
      <th>1</th>
      <td>agent modal country</td>
      <td>0.421</td>
    </tr>
  </tbody>
</table>
</div>


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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>missing_%</th>
      <th>corr_target</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>min</th>
      <th>max</th>
      <th>skew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Registration_Days_Before</td>
      <td>4.2</td>
      <td>0.351</td>
      <td>102.89</td>
      <td>65.0</td>
      <td>109.18</td>
      <td>0.0</td>
      <td>629.0</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pre_Course_Supports_Tickets</td>
      <td>0.0</td>
      <td>-0.301</td>
      <td>0.51</td>
      <td>0.0</td>
      <td>0.76</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Prev_Course_Dropouts</td>
      <td>0.0</td>
      <td>0.199</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>0.45</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>15.70</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Registration_Changes</td>
      <td>0.0</td>
      <td>-0.148</td>
      <td>0.18</td>
      <td>0.0</td>
      <td>0.59</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>7.36</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Physical_Course_Kits</td>
      <td>1.6</td>
      <td>-0.138</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Waiting_List_Days</td>
      <td>0.0</td>
      <td>0.068</td>
      <td>3.98</td>
      <td>0.0</td>
      <td>23.20</td>
      <td>0.0</td>
      <td>391.0</td>
      <td>9.26</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Returning_Client</td>
      <td>0.0</td>
      <td>-0.059</td>
      <td>0.03</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.82</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Professionals_Count</td>
      <td>0.0</td>
      <td>0.057</td>
      <td>1.84</td>
      <td>2.0</td>
      <td>0.51</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>-0.47</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Prev_Course_Attended</td>
      <td>0.0</td>
      <td>-0.052</td>
      <td>0.12</td>
      <td>0.0</td>
      <td>1.54</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>21.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Theory_Hours</td>
      <td>0.0</td>
      <td>0.045</td>
      <td>2.16</td>
      <td>2.0</td>
      <td>1.47</td>
      <td>0.0</td>
      <td>41.0</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Observers_Count</td>
      <td>0.0</td>
      <td>-0.031</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.09</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>45.38</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Daily_Tuition_Cost</td>
      <td>0.1</td>
      <td>-0.024</td>
      <td>98.85</td>
      <td>94.5</td>
      <td>41.86</td>
      <td>0.0</td>
      <td>5400.0</td>
      <td>32.55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Practical_Hours</td>
      <td>0.0</td>
      <td>0.005</td>
      <td>6.61</td>
      <td>1.0</td>
      <td>215.50</td>
      <td>-5.0</td>
      <td>10000.0</td>
      <td>40.45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Students_Count</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>8.75</td>
      <td>0.0</td>
      <td>294.24</td>
      <td>0.0</td>
      <td>9999.0</td>
      <td>33.92</td>
    </tr>
  </tbody>
</table>
</div>


The maximum values reveal several likely data errors: `Students_Count` reaches 9999, and `Practical_Hours` contains both negative values and values up to 10000. We leave the raw values unchanged for this first inspection and decide how to handle them in the outlier section.


```python
corr = train_raw[num_cols + [TARGET]].corr()
plt.figure(figsize=(13, 10))
sns.heatmap(
    corr, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm", center=0
)
plt.title("Numeric correlation heatmap (incl. target)")
plt.tight_layout()
plt.show()
```


    
![svg](notebook_files/notebook_43_0.svg)
    


No raw numeric feature has an extremely strong Pearson correlation with the target. `Registration_Days_Before` and `Pre_Course_Supports_Tickets` stand out most, while inter-feature correlations are generally modest. Because Pearson correlation measures linear association and is sensitive to extremes, we next use binned drop rates to inspect the shape of the strongest relationships.

## 3.6 Numeric drop-rate profiles

Binning a couple of the more predictive numeric features shows _how_ risk moves with them (not just whether they correlate linearly).


```python
def plot_dropout_by_bins(df, col, bins=8, ax=None):
    tmp = df[[col, TARGET]].dropna().copy()
    tmp['bin'] = pd.qcut(tmp[col], q=bins, duplicates='drop')
    stats = tmp.groupby('bin', observed=True)[TARGET].mean().mul(100)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    stats.plot.bar(ax=ax, color='#4c72b0')
    ax.axhline(df[TARGET].mean() * 100, ls='--', color='red', label='mean')
    ax.set_ylabel('drop rate (%)')
    ax.set_title(f'Drop rate by {col} bins')
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    return stats

def plot_numeric_bins():
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    plot_dropout_by_bins(train_raw, 'Registration_Days_Before', bins=8, ax=axes[0])
    plot_dropout_by_bins(
        train_raw, 'Pre_Course_Supports_Tickets', bins=6, ax=axes[1]
    )
    plt.tight_layout()
    plt.show()

plot_numeric_bins()
```


    
![svg](notebook_files/notebook_46_0.svg)
    


Drop rate rises across longer registration lead times, which suggests that plans are more likely to change when courses are booked far in advance. More pre-course support tickets are associated with lower dropping, suggesting that early engagement may reflect stronger commitment.

## 3.7 EDA conclusions

Several observations now guide preparation and modelling:

- Missing `Company_ID`, support activity, registration channel, enrollment type, and lead time all separate groups with different drop rates. Together, these patterns suggest a broader difference in buyer commitment.
- `Payment_Terms` is unusually strong and counter-intuitive. We include it and later test how much XGBoost depends on it.
- Country and agent both contain signal and overlap slightly. Their many levels require a compact encoding instead of a large one-hot expansion.
- The later test window and changing monthly rates make time-aware validation important. We derive calendar and trend features, then test the time index after selecting a model.
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



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>min</th>
      <th>max</th>
      <th>q99</th>
      <th>why</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Students_Count</td>
      <td>0.0</td>
      <td>9999.0</td>
      <td>2.0</td>
      <td>max=9999 &gt;&gt; q99=2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Practical_Hours</td>
      <td>-5.0</td>
      <td>10000.0</td>
      <td>3.0</td>
      <td>negative values; max=10000 &gt;&gt; q99=3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Prev_Course_Dropouts</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>max=21 &gt;&gt; q99=1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Prev_Course_Attended</td>
      <td>0.0</td>
      <td>61.0</td>
      <td>3.0</td>
      <td>max=61 &gt;&gt; q99=3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Registration_Changes</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>2.0</td>
      <td>max=21 &gt;&gt; q99=2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Daily_Tuition_Cost</td>
      <td>0.0</td>
      <td>5400.0</td>
      <td>209.7</td>
      <td>max=5400 &gt;&gt; q99=209.7</td>
    </tr>
  </tbody>
</table>
</div>


    Suspect columns — TEST



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>min</th>
      <th>max</th>
      <th>q99</th>
      <th>why</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Students_Count</td>
      <td>0.0</td>
      <td>9999.0</td>
      <td>2.0</td>
      <td>max=9999 &gt;&gt; q99=2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Practical_Hours</td>
      <td>-5.0</td>
      <td>10000.0</td>
      <td>2.0</td>
      <td>negative values; max=10000 &gt;&gt; q99=2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Prev_Course_Attended</td>
      <td>0.0</td>
      <td>72.0</td>
      <td>3.0</td>
      <td>max=72 &gt;&gt; q99=3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Waiting_List_Days</td>
      <td>0.0</td>
      <td>183.0</td>
      <td>0.0</td>
      <td>max=183 &gt;&gt; q99=0</td>
    </tr>
  </tbody>
</table>
</div>


The test set introduces no new forms of corruption, suggesting the same cleaning policy can be safely shared. Comparing the maximum values to the 99th percentile helps identify columns with extreme outliers:


```python
import math

TAIL_CHECK_COLS = [
    "Students_Count",
    "Practical_Hours",
    "Daily_Tuition_Cost",
    "Prev_Course_Attended",
    "Waiting_List_Days",
    "Registration_Changes",
]

def build_tail_data():
    frames = []
    for split, df in [("train", train_raw), ("test", test_raw)]:
        for col in TAIL_CHECK_COLS:
            frame = df[col].dropna().rename("value").to_frame()
            frame["split"] = split
            frame["column"] = col
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)

tail_long = build_tail_data()

def plot_tail_checks():
    n_cols = min(3, len(TAIL_CHECK_COLS))
    n_rows = math.ceil(len(TAIL_CHECK_COLS) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.6 * n_rows),
        sharey=False,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, col in zip(axes, TAIL_CHECK_COLS):
        subset = tail_long[tail_long["column"] == col]
        sns.boxplot(
            data=subset,
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
            split_values = subset.loc[subset["split"] == split, "value"]
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
        positive = subset.loc[subset["value"] > 0, "value"]
        if not positive.empty:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(positive.min() * 0.7, 0.1))
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylabel("Raw value (log scale)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(TAIL_CHECK_COLS) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Raw tail check for candidate capped columns",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    plt.show()

plot_tail_checks()
```


    
![svg](notebook_files/notebook_53_0.svg)
    


As seen in the box-plots,  top 3 (student count, practical hours and tuition) justify a  cap. We therefore apply:

- `Students_Count <= 10`: the values beyond the observed low-count support are repeated `9999` placeholders in both train and test. The cap keeps those rows as large groups without treating 9999 as a real count.
- `Practical_Hours` in `[0, 12]`: negative values are impossible, and `5000`/`10000` are clear placeholders. A 12-hour upper bound still allows a long practical day and prevents corrupted placeholder values from distorting the feature space.
- `Daily_Tuition_Cost <= 600`: train has a single `5400` value, while the test maximum is 510. A cap of 600 leaves the observed test range untouched and prevents one corrupted training value from dominating cost calculations.

Other flagged count columns (`Prev_Course_Dropouts`, `Prev_Course_Attended`, `Registration_Changes`, and test-side `Waiting_List_Days`) have long but plausible tails (as seen in the box-plots), so we leave them unchanged and restrict clipping to the three apparent data-entry errors above.


```python
CAP_RULES = {
    'Students_Count': {
        'lower': None,
        'upper': 10,
        'problem': '9999 placeholder',
        'reason': 'repeated 9999 values are isolated placeholders beyond the observed support',
    },
    'Practical_Hours': {
        'lower': 0,
        'upper': 12,
        'problem': 'negative values and 10000',
        'reason': 'course hours cannot be negative; 12 covers a long practical day',
    },
    'Daily_Tuition_Cost': {
        'lower': None,
        'upper': 600,
        'problem': '5400 value',
        'reason': '5400 is far beyond the valid fee range; 600 keeps the high-cost tail',
    },
}

def apply_cap(s, lower=None, upper=None):
    if lower is not None:
        s = s.clip(lower=lower)
    if upper is not None:
        s = s.clip(upper=upper)
    return s

def build_cap_summary():
    rows = []
    for col, rule in CAP_RULES.items():
        lo, hi = (rule['lower'], rule['upper'])
        train_changed = (
            train_raw[col].notna()
            & apply_cap(train_raw[col], lo, hi).ne(train_raw[col])
        ).sum()
        test_changed = (
            test_raw[col].notna()
            & apply_cap(test_raw[col], lo, hi).ne(test_raw[col])
        ).sum()
        action = f'clip to [{lo}, {hi}]' if lo is not None else f'clip to <= {hi}'
        rows.append({
            'column': col,
            'raw_train_min': train_raw[col].min(),
            'raw_train_max': train_raw[col].max(),
            'problem': rule['problem'],
            'action': action,
            'train_rows_affected': int(train_changed),
            'test_rows_affected': int(test_changed),
            'reason': rule['reason'],
        })
    return pd.DataFrame(rows)

display(build_cap_summary())

def plot_cap_effects():
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row')
    for j, (col, rule) in enumerate(CAP_RULES.items()):
        before = train_raw[col].dropna()
        after = apply_cap(train_raw[col], rule['lower'], rule['upper']).dropna()
        axes[0, j].hist(before, bins=50, color='#c44e52')
        axes[0, j].set_yscale('log')
        axes[0, j].set_title(f'{col}: raw')
        axes[0, j].set_xlabel(f'max={before.max():g}')
        axes[1, j].hist(after, bins=30, color='#55a868')
        axes[1, j].set_yscale('log')
        axes[1, j].set_title(f'{col}: clipped')
        axes[1, j].set_xlabel(f'max={after.max():g}')
    axes[0, 0].set_ylabel('count (log)')
    axes[1, 0].set_ylabel('count (log)')
    plt.tight_layout()
    plt.show()

plot_cap_effects()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>raw_train_min</th>
      <th>raw_train_max</th>
      <th>problem</th>
      <th>action</th>
      <th>train_rows_affected</th>
      <th>test_rows_affected</th>
      <th>reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Students_Count</td>
      <td>0.0</td>
      <td>9999.0</td>
      <td>9999 placeholder</td>
      <td>clip to &lt;= 10</td>
      <td>55</td>
      <td>12</td>
      <td>repeated 9999 values are isolated placeholders...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Practical_Hours</td>
      <td>-5.0</td>
      <td>10000.0</td>
      <td>negative values and 10000</td>
      <td>clip to [0, 12]</td>
      <td>121</td>
      <td>23</td>
      <td>course hours cannot be negative; 12 covers a l...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Daily_Tuition_Cost</td>
      <td>0.0</td>
      <td>5400.0</td>
      <td>5400 value</td>
      <td>clip to &lt;= 600</td>
      <td>1</td>
      <td>0</td>
      <td>5400 is far beyond the valid fee range; 600 ke...</td>
    </tr>
  </tbody>
</table>
</div>



    
![svg](notebook_files/notebook_55_1.svg)
    


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

| Raw signal                | Engineered feature(s)                                        | Reason for testing it                                                                                                                                                |
| ------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Course_Start_Date`       | `start_month`, `start_week`, `start_dow`, `days_since_epoch` | Month/week represent possible yearly seasonality, weekday represents scheduling patterns, and the linear index represents the longer-term shift seen in Section 3.1. |
| Participant counts        | `total_participants`, `prof_share`                           | Total size and professional share describe group composition more directly than three separate counts.                                                               |
| Practical/theory hours    | `total_hours`, `practical_share`                             | Total duration and hands-on share distinguish courses with the same raw hour count but different structure.                                                          |
| Client history            | `prev_drop_rate = dropouts / (attended + 1)`                 | Combines previous dropouts and attendance into one history signal; the `+1` handles clients with no attended courses.                                                |
| Tuition cost and hours    | `cost_x_days`                                                | Combines price and course length so the model can consider their interaction.                                                                                        |
| Requested vs assigned lab | `got_requested_lab`                                          | Captures whether the assigned lab configuration matches the original request.                                                                                        |
| Missing company/agent IDs | `has_company_id`, `has_agent_id`                             | Preserves the presence differences observed in Section 3.2 even when a raw identifier is removed.                                                                    |
| Agent/company/country IDs | frequency encodings and native categories                    | Retains identity and commonness information while avoiding a wide dummy matrix.                                                                                      |

## 5.2 Dimensionality

The main expansion risk comes from identifiers: `Agent_ID` has 204 cleaned levels and `Origin_Country` has 154. Different model families therefore need different preparation paths.

Models that require numeric inputs receive rare-level grouping, one-hot encoding, training-median imputation, and scaling. Boosted-tree implementations with native categorical support can work with category labels directly, so the matrix does not need one dummy column per agent or country. We also add one frequency feature per high-cardinality identifier and remove raw `Company_ID`, retaining only its frequency and presence flag.

During validation, frequency maps are learned from the earlier training partition. For the final submission, we compute frequencies across the available train and test features, without using `Dropped_Course`. This transductive step gives each identifier one consistent frequency at scoring time.


```python
def make_freq_maps(*dfs):
    """Label-free frequency of each ID value across the supplied frames."""
    combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
    return {
        col: combined[col].value_counts(normalize=True)
        for col in ('Agent_ID', 'Company_ID', 'Origin_Country')
    }

freq_maps = make_freq_maps(train_raw, test_raw)

def build_features(
    df: pd.DataFrame, freq_maps: dict, add_time: bool = True
) -> pd.DataFrame:
    """Cleaning + feature engineering. Identical transform for train and test.

    Mirrors the feature values used by the scored ``pipeline.py`` transform."""
    df = normalize_cats(df)
    out = pd.DataFrame(index=df.index)
    out['Professionals_Count'] = df['Professionals_Count']
    out['Students_Count'] = df['Students_Count'].clip(upper=10)
    out['Observers_Count'] = df['Observers_Count']
    out['Practical_Hours'] = df['Practical_Hours'].clip(0, 12)
    out['Theory_Hours'] = df['Theory_Hours']
    out['Registration_Days_Before'] = df['Registration_Days_Before']
    out['Prev_Course_Dropouts'] = df['Prev_Course_Dropouts']
    out['Prev_Course_Attended'] = df['Prev_Course_Attended']
    out['Pre_Course_Supports_Tickets'] = df['Pre_Course_Supports_Tickets']
    out['Physical_Course_Kits'] = df[
        'Physical_Course_Kits'
    ]  # numeric passthrough with sanity caps (Section 4)
    out['Waiting_List_Days'] = df['Waiting_List_Days']
    out['Registration_Changes'] = df['Registration_Changes']
    out['Returning_Client'] = df['Returning_Client']
    out['Daily_Tuition_Cost'] = df['Daily_Tuition_Cost'].clip(upper=600)
    d = df['Course_Start_Date']
    out['start_month'] = d.dt.month
    out['start_dow'] = d.dt.dayofweek
    out['start_week'] = d.dt.isocalendar().week.astype(float)
    if add_time:
        out['days_since_epoch'] = (d - pd.Timestamp('2015-01-01')).dt.days
    total = (
        df['Professionals_Count'].fillna(0)
        + df['Students_Count'].clip(upper=10).fillna(0)
        + df['Observers_Count'].fillna(0)
    )
    out['total_participants'] = total
    out['prof_share'] = df['Professionals_Count'] / total.replace(0, np.nan)
    out['total_hours'] = df['Practical_Hours'].clip(0, 12) + df['Theory_Hours']
    out['practical_share'] = df['Practical_Hours'].clip(0, 12) / out[
        'total_hours'
    ].replace(0, np.nan)
    out['cost_x_days'] = (
        df['Daily_Tuition_Cost'].clip(upper=600) * out['total_hours']
    )  # legacy name: this is a cost-hours interaction, not a number of days
    out['prev_drop_rate'] = df['Prev_Course_Dropouts'] / (
        df['Prev_Course_Attended'] + 1
    )
    out['kits_per_participant'] = df['Physical_Course_Kits'] / total.replace(
        0, np.nan
    )
    out['tickets_per_participant'] = df[
        'Pre_Course_Supports_Tickets'
    ] / total.replace(0, np.nan)
    out['got_requested_lab'] = (
        df['Requested_Lab_Config'] == df['Assigned_Lab_Config']
    ).astype(float)
    out['has_company_id'] = df['Company_ID'].notna().astype(int)
    out['has_agent_id'] = df['Agent_ID'].notna().astype(int)
    for col in ('Agent_ID', 'Company_ID', 'Origin_Country'):
        out[f'{col}_freq'] = (
            df[col].map(freq_maps[col]).fillna(0).astype(float)
        )  # group composition & ratios
    for col in (
        'Origin_Country',
        'Catering_Package',
        'Welcome_Gift_Type',
        'Requested_Lab_Config',
        'Enrollment_Type',
        'Lanyard_Color',
        'Client_Category',
        'Submission_Source',
        'Payment_Terms',
        'Agent_ID',
    ):
        out[col] = df[col].fillna('missing').astype('category')
    return out

def align_categories(train_X, *others):
    """Give every frame identical category levels so the boosters agree."""
    for col in train_X.select_dtypes('category').columns:
        cats = train_X[col].cat.categories
        for o in others:
            cats = cats.union(o[col].cat.categories)
        train_X[col] = train_X[col].cat.set_categories(cats)
        for o in others:
            o[col] = o[col].cat.set_categories(cats)

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
Xtr_t = build_features(tr_raw, freq_maps_chrono, add_time=True)
Xva_t = build_features(va_raw, freq_maps_chrono, add_time=True)
align_categories(Xtr_t, Xva_t)
Xtr_n = build_features(tr_raw, freq_maps_chrono, add_time=False)
Xva_n = build_features(va_raw, freq_maps_chrono, add_time=False)
align_categories(Xtr_n, Xva_n)
```

    chrono split -> fit=51,822  validate=11,642  (val drop rate=0.420)


# 7. Model experiments & tuning

The assignment requires at least three models and hyperparameter tuning. We compare one linear model, one neural network, and one boosted-tree model on the same chronological holdout.

## 7.1 The model families

We tune one important capacity or regularization parameter for each family and compare the selected settings on the same holdout. The strongest family becomes the focus of the next experiments.

Each model family is paired with its appropriate preprocessing pipeline: bounded one-hot and scaling for the continuous baselines, and native categorical handling for the tree boosters.

- **Logistic Regression** provides an interpretable linear reference. Its main tuning parameter here is `C`, the inverse regularization strength.
- **MLP** can learn nonlinear combinations but requires a complete, scaled numeric matrix. We tune its L2 penalty `alpha` while keeping a small two-layer architecture fixed.
- **XGBoost** builds trees sequentially so later trees correct earlier errors. It can represent thresholds and interactions directly; we tune tree depth and then the learning-rate/tree-count budget.


```python
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

def fit_predict_xgb_pair(X_tr, y_tr, X_va, sample_weight=None, **model_params):
    """Fit XGBoost and return P(drop) on train and validation."""
    m = get_xgb(**model_params)
    m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]

def fit_predict_xgb(X_tr, y_tr, X_va, sample_weight=None, **model_params):
    """Fit XGBoost and return P(drop) on validation."""
    _, pred_va = fit_predict_xgb_pair(
        X_tr, y_tr, X_va, sample_weight, **model_params
    )
    return pred_va

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

For each required family, we vary one parameter that controls capacity or regularization and select it using chronological validation ROC-AUC, the project metric. Training AUC is shown beside it so we can see when extra capacity improves fit without improving the future holdout.

For XGBoost, this first sweep varies `max_depth` while holding the boosting budget fixed. A second experiment then tunes the interaction between learning rate and number of trees.


```python
Xtr_enc, Xva_enc = encode_for_continuous_models(Xtr_t, Xva_t)
# We scale continuous inputs using StandardScaler to preserve the variance of naturally skewed columns (like Daily_Tuition_Cost) without compressing normal-range observations.
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
    def loss_auc(train_predictions, valid_predictions):
        return {
            'train_logloss': log_loss(y_train, train_predictions),
            'val_logloss': log_loss(y_valid, valid_predictions),
            'train_AUC': roc_auc_score(y_train, train_predictions),
            'val_AUC': roc_auc_score(y_valid, valid_predictions),
        }

    def fit_xgb_pair(depth):
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=depth,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            enable_categorical=True,
            tree_method='hist',
            eval_metric='auc',
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train_tree, y_train)
        return (
            model.predict_proba(X_train_tree)[:, 1],
            model.predict_proba(X_valid_tree)[:, 1],
        )

    tuning_rows = []
    selected_predictions = {}
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        model = LogisticRegression(C=C, max_iter=2000).fit(X_train_scaled, y_train)
        train_predictions = model.predict_proba(X_train_scaled)[:, 1]
        valid_predictions = model.predict_proba(X_valid_scaled)[:, 1]
        tuning_rows.append({
            'family': 'Logistic Regression',
            'axis': 'C  (less regularisation →)',
            'x': C,
            **loss_auc(train_predictions, valid_predictions),
        })
        if np.isclose(C, 0.001):
            selected_predictions['lr'] = valid_predictions
    for alpha in (1.0, 0.1, 0.01, 0.001, 0.0001):
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        ).fit(X_train_scaled, y_train)
        train_predictions = model.predict_proba(X_train_scaled)[:, 1]
        valid_predictions = model.predict_proba(X_valid_scaled)[:, 1]
        tuning_rows.append({
            'family': 'MLP neural network',
            'axis': '1 / alpha  (less regularisation →)',
            'x': 1.0 / alpha,
            **loss_auc(train_predictions, valid_predictions),
        })
        if np.isclose(alpha, 0.1):
            selected_predictions['mlp'] = valid_predictions
    for depth in (2, 3, 4, 5, 6, 8, 10):
        train_predictions, valid_predictions = fit_xgb_pair(depth)
        tuning_rows.append({
            'family': 'Gradient-boosted trees (XGBoost)',
            'axis': 'max_depth  (more capacity →)',
            'x': depth,
            **loss_auc(train_predictions, valid_predictions),
        })
    return tuning_rows, selected_predictions

tuning_rows, selected_predictions = hyper_tune(
    Xtr_scaled, Xva_scaled, Xtr_t, Xva_t, y_tr, y_va, SEED
)
# Linear baseline: inverse-regularisation C (higher C -> less regularisation).
tuning = pd.DataFrame(tuning_rows)
selected_x = {
    'Logistic Regression': 0.001,
    'MLP neural network': 10.0,
    'Gradient-boosted trees (XGBoost)': 6,
}
tuning['selected'] = tuning.apply(
    lambda r: bool(np.isclose(r['x'], selected_x[r['family']])), axis=1
)

def plot_tuning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (ax, family) in enumerate(zip(axes, selected_x)):
        data = tuning[tuning['family'] == family].sort_values('x')
        ax.plot(
            data['x'],
            data['train_AUC'],
            color='#2b5c8f',
            linestyle='--',
            marker='o',
            alpha=0.7,
            label='train AUC',
        )
        ax.plot(
            data['x'],
            data['val_AUC'],
            color='#1b9e77',
            linestyle='-',
            marker='s',
            linewidth=2,
            label='val AUC',
        )
        star = data[data['selected']].iloc[0]
        ax.scatter(
            star['x'],
            star['val_AUC'],
            marker='*',
            s=250,
            color='#ffd700',
            edgecolor='black',
            linewidths=1.5,
            zorder=10,
            label='selected setting',
        )
        ax.set_title(family, fontsize=12, pad=12, fontweight='bold')
        ax.set_xlabel(data['axis'].iloc[0], fontsize=10)
        if i == 0:
            ax.set_ylabel('ROC-AUC Score', fontsize=10)
        if family != 'Gradient-boosted trees (XGBoost)':
            ax.set_xscale('log')
        ax.grid(True, linestyle=':', alpha=0.6)
        loc = 'center right' if family == 'Logistic Regression' else 'lower left'
        if family == 'Gradient-boosted trees (XGBoost)':
            loc = 'lower right'
        ax.legend(
            loc=loc,
            fontsize=8,
            frameon=True,
            facecolor='white',
            framealpha=0.9,
        )
    fig.suptitle(
        'Focused tuning: training vs validation ROC-AUC',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

plot_tuning_curves()
selected_tuning = tuning.loc[
    tuning['selected'], ['family', 'x', 'train_AUC', 'val_AUC']
].copy()
selected_tuning[['train_AUC', 'val_AUC']] = selected_tuning[
    ['train_AUC', 'val_AUC']
].round(4)
display(selected_tuning)
pred_lr = selected_predictions['lr']
pred_mlp = selected_predictions['mlp']
```


    
![svg](notebook_files/notebook_74_0.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>family</th>
      <th>x</th>
      <th>train_AUC</th>
      <th>val_AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.001</td>
      <td>0.9217</td>
      <td>0.8805</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MLP neural network</td>
      <td>10.000</td>
      <td>0.9645</td>
      <td>0.8762</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Gradient-boosted trees (XGBoost)</td>
      <td>6.000</td>
      <td>0.9715</td>
      <td>0.9135</td>
    </tr>
  </tbody>
</table>
</div>


Logistic Regression performs best with strong regularization (`C=0.001`); increasing `C` improves training fit slightly but reduces holdout AUC. The MLP performs best at `alpha=0.1`, after which its training/validation gap grows. XGBoost holdout AUC rises through depth 6 and then levels off while training AUC continues upward, so we keep depth 6 as the best trade-off in this sweep.

The selected settings reach validation AUC 0.8805 for Logistic Regression, 0.8762 for MLP, and 0.9135 for XGBoost. XGBoost's advantage indicates that thresholds, categories, and interactions matter for this problem.

## 7.3 Improving the gradient model

Because XGBoost achieved the highest AUC in the first sweep, we further tune it. We first tune its boosting budget, then test whether adding two fixed-configuration boosted-tree implementations improves the ranking further.

### (a) Boosting budget: number of trees × learning rate

The number of trees and the learning rate interact directly. Consistent with our validation choice, we evaluate the tree budget directly in ROC-AUC. We evaluate various tree counts across two learning rate settings:


```python
@cache
def run_budget_sweep(X_train, X_valid, y_train, y_valid, seed):
    budget_rows = []
    budget_predictions = {}
    for lr_rate in (0.1, 0.03):
        for n in (50, 100, 200, 400, 700, 1000):
            model = XGBClassifier(
                n_estimators=n,
                learning_rate=lr_rate,
                max_depth=6,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                enable_categorical=True,
                tree_method='hist',
                eval_metric='auc',
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            train_predictions = model.predict_proba(X_train)[:, 1]
            valid_predictions = model.predict_proba(X_valid)[:, 1]
            budget_predictions[(lr_rate, n)] = valid_predictions
            budget_rows.append({
                'learning_rate': lr_rate,
                'n_trees': n,
                'train_logloss': log_loss(y_train, train_predictions),
                'val_logloss': log_loss(y_valid, valid_predictions),
                'train_AUC': roc_auc_score(y_train, train_predictions),
                'val_AUC': roc_auc_score(y_valid, valid_predictions),
            })
    return budget_rows, budget_predictions

budget_rows, budget_predictions = run_budget_sweep(Xtr_t, Xva_t, y_tr, y_va, SEED)
budget = pd.DataFrame(budget_rows)

def plot_budget_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (lr_rate, subset) in zip(axes, budget.groupby('learning_rate')):
        subset = subset.sort_values('n_trees')
        ax.plot(
            subset['n_trees'],
            subset['train_AUC'],
            marker='o',
            ls='--',
            color='#2b5c8f',
            label='train AUC',
        )
        ax.plot(
            subset['n_trees'],
            subset['val_AUC'],
            marker='o',
            color='#d95f02',
            linewidth=2,
            label='val AUC',
        )
        best_row = subset.loc[subset['val_AUC'].idxmax()]
        ax.scatter(
            best_row['n_trees'],
            best_row['val_AUC'],
            marker='*',
            s=250,
            color='#ffd700',
            edgecolor='black',
            linewidths=1.5,
            zorder=10,
            label=f"best AUC ({int(best_row['n_trees'])} trees)",
        )
        ax.set_xlabel('number of trees (n_estimators)', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('ROC-AUC Score', fontsize=10)
        ax.set_title(f'Learning Rate = {lr_rate}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='lower right')
    fig.suptitle(
        'Boosting budget: ROC-AUC vs number of trees, by learning rate (star = best AUC)',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

plot_budget_curves()
budget_best = budget.loc[
    budget.groupby('learning_rate')['val_AUC'].idxmax(),
    ['learning_rate', 'n_trees', 'train_AUC', 'val_AUC'],
].copy()
budget_best[['train_AUC', 'val_AUC']] = budget_best[['train_AUC', 'val_AUC']].round(
    4
)
display(budget_best)
pred_xgb = budget_predictions[(0.03, 700)]
```


    
![svg](notebook_files/notebook_77_0.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>learning_rate</th>
      <th>n_trees</th>
      <th>train_AUC</th>
      <th>val_AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>0.03</td>
      <td>700</td>
      <td>0.9770</td>
      <td>0.9135</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.10</td>
      <td>200</td>
      <td>0.9756</td>
      <td>0.9125</td>
    </tr>
  </tbody>
</table>
</div>


At learning rate 0.1, validation AUC peaks around 200 trees and then declines while training AUC keeps rising. At 0.03, improvement is slower but the holdout reaches a slightly higher plateau around 700 trees. We choose `learning_rate=0.03` and `n_estimators=700` for XGBoost.

### (b) Does adding other boosters help?

LightGBM and CatBoost build boosted trees differently from XGBoost, so they may rank some registrations differently. We add them with fixed, capacity-aligned settings and test the ensemble itself: does combining their rankings improve the tuned XGBoost result?

We use rank averaging because ROC-AUC depends on ordering. Each model's predictions are converted to percentile ranks before averaging, so a model's probability scale cannot dominate the blend. The resulting value is a ranking score, not a calibrated cancellation probability.


```python
@cache
def fit_predict(name, X_tr, y_train, X_val, seed, sample_weight=None):
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
        model.fit(X_tr2, y_train, sample_weight=sample_weight)
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
            sample_weight=sample_weight,
            categorical_feature=X_tr.select_dtypes("category").columns.tolist(),
        )
        return model.predict_proba(X_val)[:, 1]
    model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        enable_categorical=True,
        tree_method='hist',
        eval_metric='auc',
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train, sample_weight=sample_weight)
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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>chrono_AUC</th>
      <th>delta_vs_XGBoost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rank-average blend (LGBM+XGB+Cat)</td>
      <td>0.915618</td>
      <td>0.0021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>XGBoost (tuned)</td>
      <td>0.913536</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM (fixed setting)</td>
      <td>0.913464</td>
      <td>-0.0001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CatBoost (fixed setting)</td>
      <td>0.913025</td>
      <td>-0.0005</td>
    </tr>
  </tbody>
</table>
</div>


All three boosters perform similarly on this holdout. Adding the fixed LightGBM and CatBoost rankings raises AUC from 0.9135 for tuned XGBoost to 0.9156 for the rank-average blend, which we carry forward as the boosted-tree candidate.

## 7.4 Returning to the time-index hypothesis

EDA suggested that the long-term time position might help. We test `days_since_epoch` on the rank-average blend.

Trees cannot extrapolate a linear trend beyond the observed range, but the index can still separate older and more recent training regimes. Whether that helps is an empirical question answered by the chronological holdout below.


```python
model_names = ("lgbm", "xgb", "cat")
pred_blend_no_time = rank_avg([
    fit_predict(name, Xtr_n, y_tr, Xva_n, SEED) for name in model_names
])

ablation = pd.DataFrame({
    "configuration": [
        "Rank-average blend, no time index",
        "Rank-average blend, + time index",
    ],
    "AUC": [
        roc_auc_score(y_va, pred_blend_no_time),
        roc_auc_score(y_va, blend_t),
    ],
    "validation": ["chrono", "chrono"],
})
display(ablation)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>configuration</th>
      <th>AUC</th>
      <th>validation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rank-average blend, no time index</td>
      <td>0.912226</td>
      <td>chrono</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rank-average blend, + time index</td>
      <td>0.915618</td>
      <td>chrono</td>
    </tr>
  </tbody>
</table>
</div>


The table tests the time index on the boosted-tree blend using the chronological holdout.

# 8. Model evaluation

We compare the tuned Logistic Regression, MLP, and boosted-tree candidates on the chronological holdout.

## 8.1 ROC & precision–recall curves


```python
def plot_evaluation_curves():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for predictions, name in [
        (pred_lr, 'Logistic Regression'),
        (pred_mlp, 'MLP'),
        (blend_t, 'Boosted-tree rank blend'),
    ]:
        fpr, tpr, roc_thresholds = roc_curve(y_va, predictions)
        axes[0].plot(
            fpr,
            tpr,
            label=f'{name} (AUC={auc(fpr, tpr):.3f})',
            lw=2,
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_va, predictions)
        axes[1].plot(
            recall,
            precision,
            label=f'{name} (AP={average_precision_score(y_va, predictions):.3f})',
            lw=2,
        )
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set(
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='ROC curve',
    )
    axes[0].legend(loc='lower right', fontsize=8)
    axes[1].axhline(y_va.mean(), ls='--', color='grey', alpha=0.7)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–Recall curve')
    axes[1].legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.show()

plot_evaluation_curves()
```


    
![svg](notebook_files/notebook_86_0.svg)
    


The boosted-tree blend has the highest holdout AUC in this comparison. Threshold-based metrics are examined next.

## 8.2 Confusion matrices & threshold metrics

A confusion matrix requires a threshold, so we use 0.5 as a simple reference cutoff for each candidate. For the rank-average blend, this is not a 50% cancellation probability: rank averaging preserves ordering but discards the individual models' probability scales. Nova Academy could later adjust the cutoff according to the relative cost of unnecessary follow-up and missed cancellations.


```python
def evaluate_candidates():
    candidates = [
        ('Logistic Regression', pred_lr),
        ('MLP neural network', pred_mlp),
        ('Boosted-tree rank blend', blend_t),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metric_rows = []
    for ax, (name, predictions) in zip(axes, candidates):
        labels = (predictions >= 0.5).astype(int)
        matrix = confusion_matrix(y_va, labels)
        auc_score = roc_auc_score(y_va, predictions)
        report = classification_report(
            y_va,
            labels,
            target_names=['completed', 'dropped'],
            output_dict=True,
            zero_division=0,
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=',d',
            cmap='Blues',
            cbar=False,
            square=True,
            linewidths=0.5,
            xticklabels=['pred completed', 'pred dropped'],
            yticklabels=['true completed', 'true dropped'],
            annot_kws={'fontsize': 12},
            ax=ax,
        )
        ax.set_title(f'{name}\nROC-AUC={auc_score:.3f}', fontweight='bold')
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('Actual class')
        metric_rows.append({
            'model': name,
            'ROC-AUC': auc_score,
            'accuracy': report['accuracy'],
            'precision (dropped)': report['dropped']['precision'],
            'recall (dropped)': report['dropped']['recall'],
            'F1 (dropped)': report['dropped']['f1-score'],
        })
    fig.suptitle(
        'Candidate-model confusion matrices at a 0.5 reference cutoff',
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
    return pd.DataFrame(metric_rows).set_index('model')

evaluation_metrics = evaluate_candidates().round(3)
display(evaluation_metrics)
```


    
![svg](notebook_files/notebook_89_0.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ROC-AUC</th>
      <th>accuracy</th>
      <th>precision (dropped)</th>
      <th>recall (dropped)</th>
      <th>F1 (dropped)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Logistic Regression</th>
      <td>0.881</td>
      <td>0.799</td>
      <td>0.773</td>
      <td>0.737</td>
      <td>0.754</td>
    </tr>
    <tr>
      <th>MLP neural network</th>
      <td>0.876</td>
      <td>0.790</td>
      <td>0.744</td>
      <td>0.762</td>
      <td>0.753</td>
    </tr>
    <tr>
      <th>Boosted-tree rank blend</th>
      <td>0.916</td>
      <td>0.810</td>
      <td>0.732</td>
      <td>0.865</td>
      <td>0.793</td>
    </tr>
  </tbody>
</table>
</div>


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
plt.figure(figsize=(9, 4.5))
sns.histplot(blend_t, bins=50, kde=True, color="teal")
plt.axvline(0.5, color="red", ls="--", label="reference cutoff")
plt.axvspan(0.40, 0.60, color="orange", alpha=0.2, label="near-threshold band")
plt.xlabel("Rank-average risk score")
plt.title("Selected-blend score distribution")
plt.legend()
plt.tight_layout()
plt.show()

near_threshold = ((blend_t > 0.40) & (blend_t < 0.60)).mean() * 100
print(f"share of holdout in the 0.40–0.60 band: {near_threshold:.1f}%")
```


    
![svg](notebook_files/notebook_92_0.svg)
    


    share of holdout in the 0.40–0.60 band: 20.2%


Scores in this band are close to the illustrative cutoff, so small changes in the cutoff can change their binary classification.

# 9. Interpretation with SHAP

The selected blend averages three sets of prediction ranks and therefore has no single fitted tree structure for SHAP to decompose. We use the tuned XGBoost component as a representative fitted model for detailed interpretation, then compare its SHAP patterns with the earlier EDA.

We compute TreeSHAP values on a fixed validation sample of up to 10,000 rows to keep the analysis reproducible and the runtime manageable.


```python
@cache
def compute_shap_analysis(X_train, X_valid, y_train, y_valid, seed):
    shap_model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        enable_categorical=True,
        tree_method='hist',
        eval_metric='auc',
        random_state=seed,
        n_jobs=-1,
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
    Xtr_t, Xva_t, y_tr, y_va, SEED
)
print(f"XGBoost+time chrono AUC: {shap_auc:.4f}")
```

    XGBoost+time chrono AUC: 0.9135


## 9.1 Global importance (beeswarm + bar)


```python
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title('SHAP summary (beeswarm) — XGBoost+time')
plt.tight_layout()
plt.show()
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

def plot_shap_importance():
    ax = top.sort_values('mean_abs_shap').plot.barh(
        x='feature',
        y='mean_abs_shap',
        legend=False,
        figsize=(8, 7),
        color='#4c72b0',
    )
    ax.set_xlabel('mean |SHAP value|')
    ax.set_title('Top 20 features by SHAP importance')
    plt.tight_layout()
    plt.show()

plot_shap_importance()
display(top)
```


    
![svg](notebook_files/notebook_97_0.svg)
    



    
![svg](notebook_files/notebook_97_1.svg)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>mean_abs_shap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Payment_Terms</td>
      <td>1.237425</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Origin_Country</td>
      <td>0.669338</td>
    </tr>
    <tr>
      <th>2</th>
      <td>days_since_epoch</td>
      <td>0.530472</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Agent_ID</td>
      <td>0.504151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>tickets_per_participant</td>
      <td>0.355377</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Registration_Days_Before</td>
      <td>0.338935</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pre_Course_Supports_Tickets</td>
      <td>0.248548</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Enrollment_Type</td>
      <td>0.233831</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Client_Category</td>
      <td>0.225936</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Origin_Country_freq</td>
      <td>0.200234</td>
    </tr>
    <tr>
      <th>10</th>
      <td>got_requested_lab</td>
      <td>0.176595</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Registration_Changes</td>
      <td>0.120067</td>
    </tr>
    <tr>
      <th>12</th>
      <td>prev_drop_rate</td>
      <td>0.107810</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Physical_Course_Kits</td>
      <td>0.106446</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Daily_Tuition_Cost</td>
      <td>0.086305</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Agent_ID_freq</td>
      <td>0.083634</td>
    </tr>
    <tr>
      <th>16</th>
      <td>start_week</td>
      <td>0.083600</td>
    </tr>
    <tr>
      <th>17</th>
      <td>cost_x_days</td>
      <td>0.069765</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Catering_Package</td>
      <td>0.042550</td>
    </tr>
    <tr>
      <th>19</th>
      <td>kits_per_participant</td>
      <td>0.040752</td>
    </tr>
  </tbody>
</table>
</div>


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


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>chrono_AUC</th>
      <th>delta_vs_with_payment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rank-average blend</td>
      <td>0.915618</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rank-average blend, no Payment_Terms</td>
      <td>0.910071</td>
      <td>-0.005546</td>
    </tr>
  </tbody>
</table>
</div>


Removing `Payment_Terms` changes chronological AUC from 0.9156 to 0.9101. The field's exact recording time is still worth confirming with the data owner.

## 9.2 Direction of the strongest non-payment signal

`Origin_Country` is the strongest feature after `Payment_Terms`, so we plot the average SHAP contribution of its most common levels. Positive values push XGBoost toward a higher cancellation score; negative values push it toward completion.


```python
top_feat = (
    importance.loc[importance['feature'] != 'Payment_Terms', 'feature'].iloc[0]
    if importance['feature'].iloc[0] == 'Payment_Terms'
    else importance['feature'].iloc[0]
)

def plot_shap_dependence_readable(feature, max_categories=15):
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
        plt.tight_layout()
        plt.show()
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
    fig_h = max(4, min(7, 0.32 * len(summary) + 1.2))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.barplot(
        data=summary.reset_index(),
        y='level',
        x='mean_shap',
        color='#4c72b0',
        ax=ax,
    )
    ax.axvline(0, color='black', lw=1, alpha=0.5)
    ax.set_title(f'Mean SHAP by {feature} level (top {max_categories} + other)')
    ax.set_xlabel('mean SHAP contribution')
    ax.set_ylabel(feature)
    plt.tight_layout()
    plt.show()

try:
    plot_shap_dependence_readable(top_feat)
except Exception as e:
    print(f'(dependence view skipped for {top_feat}: {e})')
```


    
![svg](notebook_files/notebook_102_0.svg)
    


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
plt.tight_layout()
plt.show()
```

    explaining order at sample position 13 — model P(drop)=0.488



    
![svg](notebook_files/notebook_104_1.svg)
    


For this registration, positive and negative contributions nearly balance, producing a score close to the reference threshold.

# 10. Rebuilding and checking the submission

The stored `data/Group_27_Submission.csv` is the submission that received the leaderboard score. The block below can retrain the three-model blend and compare the rebuilt ranking with that submission.


```python
def build_submission_candidate(
    out_path="data/Group_27_Submission_candidate.csv", write=False
):
    fm = make_freq_maps(train_raw, test_raw)
    X_train_full = build_features(train_raw, fm)
    X_test = build_features(test_raw, fm)
    align_categories(X_train_full, X_test)
    y_full = train_raw[TARGET].values

    preds = []
    for name in ("lgbm", "xgb", "cat"):
        preds.append(fit_predict(name, X_train_full, y_full, X_test, SEED))
        print(f"fitted {name} on {len(X_train_full):,} rows")

    submission = pd.DataFrame({
        "Client_ID": test_raw["Client_ID"],
        "Drop_Probability": rank_avg(preds),
    })
    if write:
        submission.to_csv(out_path, index=False)
        print(f"wrote {out_path}  ({len(submission):,} rows)")
    return submission

def rebuild_and_compare():
    submission = build_submission_candidate(write=False)
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

RUN_REBUILD_AND_COMPARE = False  # expensive; keep disabled during routine editing

if RUN_REBUILD_AND_COMPARE:
    rebuild_and_compare()
```

The comparison checks the rebuilt file's schema and its Spearman rank agreement with the submitted ranking.

# 11. Conclusions & Executive Summary

Nova Academy's test registrations occur after the training period, and both the monthly target rate and the adversarial-validation result (AUC 0.935) show temporal distribution shift. Model selection therefore used a four-month chronological holdout.

Cleaning reduced hundreds of inconsistent text labels to compact category sets. Missingness, payment terms, country, agent, registration timing, and support activity all carried predictive information. Model comparison confirmed that tuned XGBoost outperformed the Logistic Regression and MLP baselines on the future holdout. Adding fixed LightGBM and CatBoost rankings produced a small further gain, from XGBoost AUC 0.9135 to blend AUC 0.9156 on that split.

The stored rank-average submission received test ROC-AUC **0.889314**, above the required 0.70.

Further work could include rolling temporal validation, confirming when `Payment_Terms` is recorded, and calibrating the selected blend score for cost-based operational thresholds.
