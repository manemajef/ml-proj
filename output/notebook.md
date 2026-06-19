# ML Project Notebook

Rotem David Semah



```python
# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import RocCurveDisplay

TARGET_NAME = "Dropped_Course"

data = pd.read_csv("data/Train_Data.csv")
official_test_data = pd.read_csv("data/Test_Data_No_Target.csv")

```

### Helper functions



```python
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

```

## 1. EDA





```python
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", None)
print("data shape: ", data.shape)
data.head()
```

    data shape:  (63464, 29)







|   Unnamed: 0 |   Client_ID |   Professionals_Count |   Students_Count |   Observers_Count | Course_Start_Date   |   Practical_Hours |   Theory_Hours |   Registration_Days_Before | Origin_Country   | Catering_Package   | Welcome_Gift_Type   | Requested_Lab_Config   | Assigned_Lab_Config   |   Prev_Course_Dropouts |   Prev_Course_Attended |   Pre_Course_Supports_Tickets |   Physical_Course_Kits |   Waiting_List_Days |   Registration_Changes | Enrollment_Type   | Lanyard_Color   | Client_Category           | Submission_Source         |   Returning_Client |   Agent_ID |   Company_ID | Payment_Terms   |   Daily_Tuition_Cost |   Dropped_Course |
|-------------:|------------:|----------------------:|-----------------:|------------------:|:--------------------|------------------:|---------------:|---------------------------:|:-----------------|:-------------------|:--------------------|:-----------------------|:----------------------|-----------------------:|-----------------------:|------------------------------:|-----------------------:|--------------------:|-----------------------:|:------------------|:----------------|:--------------------------|:--------------------------|-------------------:|-----------:|-------------:|:----------------|---------------------:|-----------------:|
|            0 |       13766 |                     2 |                0 |                 0 | 2015-07-01          |                 0 |              2 |                        257 | PRT              | Lunch Included     | Branded Notebook    | Standard PC (Windows)  | Standard PC (Windows) |                      0 |                      0 |                             0 |                      0 |                   0 |                      0 | General Admission | Blue            | Traditional IT & Telecomm | B2B Platforms & Resellers |                  0 |        219 |          nan | Pay Upon Start  |                101.5 |                0 |
|            1 |       78660 |                     1 |                0 |                 0 | 2015-07-01          |                 0 |              2 |                        257 | PRT              | Lunch Included     | Branded Notebook    | Standard PC (Windows)  | Standard PC (Windows) |                      0 |                      0 |                             0 |                      0 |                   0 |                      1 | General Admission | Blue            | Traditional IT & Telecomm | B2B Platforms & Resellers |                  0 |        219 |          nan | Pay Upon Start  |                 80   |                0 |
|            2 |       51396 |                     1 |                0 |                 0 | 2015-07-01          |                 0 |              2 |                        257 | PRT              | Lunch Included     | USB Drive           | Standard PC (Windows)  | Standard PC (Windows) |                      0 |                      0 |                             0 |                      0 |                   0 |                      1 | General Admission | Red             | Traditional IT & Telecomm | B2B Platforms & Resellers |                  0 |        219 |          nan | Pay Upon Start  |                 80   |                0 |
|            3 |       34000 |                     2 |                0 |                 0 | 2015-07-01          |                 0 |              2 |                        257 | PRT              | Lunch Included     | Branded Notebook    | Standard PC (Windows)  | Standard PC (Windows) |                      0 |                      0 |                             0 |                      0 |                   0 |                      0 | General Admission | Red             | Traditional IT & Telecomm | B2B Platforms & Resellers |                  0 |        219 |          nan | Pay Upon Start  |                101.5 |                0 |
|            4 |       69025 |                     1 |                0 |                 0 | 2015-07-01          |                 0 |              2 |                        257 | PRT              | Lunch Included     | Branded Notebook    | Standard PC (Windows)  | Standard PC (Windows) |                      0 |                      0 |                             0 |                      0 |                   0 |                      1 | General Admission | Orange          | Traditional IT & Telecomm | B2B Platforms & Resellers |                  0 |        219 |          nan | Pay Upon Start  |                 80   |                0 |





Split the labeled training file into a development train set and a validation set. I avoid calling the validation split `test_data`, because the assignment also provides a separate `Test_Data_No_Target.csv` file that must be predicted at the end.



```python
train_data, valid_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data[TARGET_NAME],
)

print(f"train_data shape: {train_data.shape}")
print(f"valid_data shape: {valid_data.shape}")
train_data.info()

```

    train_data shape: (50771, 29)
    valid_data shape: (12693, 29)
    <class 'pandas.core.frame.DataFrame'>
    Index: 50771 entries, 8022 to 41357
    Data columns (total 29 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   Client_ID                    50771 non-null  int64  
     1   Professionals_Count          50771 non-null  int64  
     2   Students_Count               50767 non-null  float64
     3   Observers_Count              50771 non-null  int64  
     4   Course_Start_Date            50771 non-null  object 
     5   Practical_Hours              50771 non-null  int64  
     6   Theory_Hours                 50771 non-null  int64  
     7   Registration_Days_Before     48645 non-null  float64
     8   Origin_Country               50333 non-null  object 
     9   Catering_Package             50434 non-null  object 
     10  Welcome_Gift_Type            50771 non-null  object 
     11  Requested_Lab_Config         49377 non-null  object 
     12  Assigned_Lab_Config          50771 non-null  object 
     13  Prev_Course_Dropouts         50771 non-null  int64  
     14  Prev_Course_Attended         50771 non-null  int64  
     15  Pre_Course_Supports_Tickets  50771 non-null  int64  
     16  Physical_Course_Kits         49952 non-null  float64
     17  Waiting_List_Days            50771 non-null  int64  
     18  Registration_Changes         50771 non-null  int64  
     19  Enrollment_Type              50208 non-null  object 
     20  Lanyard_Color                50771 non-null  object 
     21  Client_Category              50771 non-null  object 
     22  Submission_Source            50297 non-null  object 
     23  Returning_Client             50771 non-null  int64  
     24  Agent_ID                     41843 non-null  float64
     25  Company_ID                   2476 non-null   float64
     26  Payment_Terms                50308 non-null  object 
     27  Daily_Tuition_Cost           50706 non-null  float64
     28  Dropped_Course               50771 non-null  int64  
    dtypes: float64(6), int64(12), object(11)
    memory usage: 11.6+ MB


I use `df` as shorthand for the development training split during EDA. Target-based EDA should use only this labeled training split. The official test file has no target, so it is useful for checking schema and missing-value patterns, but not for drawing dropout-rate conclusions.



```python
df = train_data
```

### Dataset Scope And Target Balance

Before looking at individual features, I check the target distribution and compare missing-value patterns between the labeled data and the official test file. I do not use the official test file for target-based conclusions, because it has no `Dropped_Course` label.



```python
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

```




|   ('Unnamed: 0_level_0', 'Dropped_Course') |   ('count', 'Unnamed: 1_level_1') |   ('rate_%', 'Unnamed: 2_level_1') |
|-------------------------------------------:|----------------------------------:|-----------------------------------:|
|                                          0 |                             29732 |                               58.6 |
|                                          1 |                             21039 |                               41.4 |







| Unnamed: 0               |   train_missing_% |   official_test_missing_% |   train_missing_count |   official_test_missing_count |
|:-------------------------|------------------:|--------------------------:|----------------------:|------------------------------:|
| Company_ID               |             95.12 |                     96.41 |                 48295 |                         15297 |
| Agent_ID                 |             17.58 |                     17.61 |                  8928 |                          2794 |
| Registration_Days_Before |              4.19 |                      4.05 |                  2126 |                           642 |
| Requested_Lab_Config     |              2.75 |                      3.01 |                  1394 |                           477 |
| Physical_Course_Kits     |              1.61 |                      1.43 |                   819 |                           227 |
| Enrollment_Type          |              1.11 |                      1.12 |                   563 |                           177 |
| Submission_Source        |              0.93 |                      0.94 |                   474 |                           149 |
| Payment_Terms            |              0.91 |                      0.91 |                   463 |                           144 |
| Origin_Country           |              0.86 |                      1.01 |                   438 |                           160 |
| Catering_Package         |              0.66 |                      0.7  |                   337 |                           111 |
| Daily_Tuition_Cost       |              0.13 |                      0.01 |                    65 |                             1 |
| Students_Count           |              0.01 |                      0    |                     4 |                             0 |




The target is not severely imbalanced: roughly 59% of the training examples are not dropped and 41% are dropped. This makes AUC a reasonable evaluation metric and means a trivial majority-class model is not enough.

The official test file has similar missingness patterns to the labeled data. This means missing values must be handled by the preprocessing pipeline; dropping rows is not a valid final strategy because the submission requires one prediction for every official test row.


#### Missingness Versus Target

For columns with meaningful missingness, I also check whether the fact that a value is missing is itself related to dropout. This helps decide whether to add missingness indicators.


```python
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
        df.assign(is_missing=df[col].isna())
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

missingness_target_summary[[
    "column",
    "is_missing",
    "count",
    "dropout_rate_%",
]]

```






|   Unnamed: 0 | column                   | is_missing   |   count |   dropout_rate_% |
|-------------:|:-------------------------|:-------------|--------:|-----------------:|
|            0 | Agent_ID                 | False        |   41843 |             43.1 |
|            1 | Agent_ID                 | True         |    8928 |             33.8 |
|            2 | Company_ID               | False        |    2476 |             21.4 |
|            3 | Company_ID               | True         |   48295 |             42.5 |
|            4 | Registration_Days_Before | False        |   48645 |             41.4 |
|            5 | Registration_Days_Before | True         |    2126 |             41.9 |
|            6 | Physical_Course_Kits     | False        |   49952 |             41.5 |
|            7 | Physical_Course_Kits     | True         |     819 |             38   |
|            8 | Daily_Tuition_Cost       | False        |   50706 |             41.4 |
|            9 | Daily_Tuition_Cost       | True         |      65 |             52.3 |
|           10 | Requested_Lab_Config     | False        |   49377 |             41.5 |
|           11 | Requested_Lab_Config     | True         |    1394 |             40.3 |
|           12 | Payment_Terms            | False        |   50308 |             41.5 |
|           13 | Payment_Terms            | True         |     463 |             36.9 |





Missingness is informative for some columns. Missing `Company_ID` is strongly associated with a higher dropout rate, which supports using a `has_company_id` feature. Missing `Agent_ID` also behaves differently from non-missing values. For `Registration_Days_Before` and `Physical_Course_Kits`, missingness itself is not the main signal, but adding missingness flags is still cheap and protects the model from treating imputed values as observed values.

### Checking Cat cols


Lets see whats going on with none-numeric columns



```python
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
```

    Number of none numeric cols:  11
    The % of none numeric cols:  39.285714285714285







|   Unnamed: 0 | column               |   missing_% |   missing_count |   unique_count | top_8_cat                                                                                                                                                                                                                                                 |
|-------------:|:---------------------|------------:|----------------:|---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|            4 | Requested_Lab_Config |         2.7 |            1394 |              8 | [Standard PC (Windows) (78.4%), Linux Workstation (13.2%), nan (2.7%), Dual Monitor Setup (2.1%), MacOS Station (1.6%), Laptop Docking Station (1.5%), High-GPU Unit (0.5%), Touch Screen Interface (0.0%)]                                               |
|            6 | Enrollment_Type      |         1.1 |             563 |            263 | [General Admission (63.8%), Affiliated Admission (21.4%), Contractual Agreement (3.2%), general admission (1.7%), GENERAL ADMISSION (1.6%), nan (1.1%), General Admission (0.8%), General Admission (0.8%)]                                               |
|            9 | Submission_Source    |         0.9 |             474 |            289 | [B2B Platforms & Resellers (76.8%), Direct Website Registration (7.3%), Dedicated Sales Team (4.0%), B2B PLATFORMS & RESELLERS (1.9%), b2b platforms & resellers (1.9%), nan (0.9%), B2B Platforms & Resellers (0.9%), B2B Platforms & Resellers (0.9%)]  |
|           10 | Payment_Terms        |         0.9 |             463 |            214 | [Pay Upon Start (73.1%), Prepaid (Non-Refundable) (15.2%), PAY UPON START (1.9%), pay upon start (1.8%), nan (0.9%), Pay Upon Start (0.9%), Pay Upon Start (0.8%), Pay Upon Start (0.8%)]                                                                 |
|            1 | Origin_Country       |         0.9 |             438 |            666 | [PRT (38.2%), FRA (10.1%), DEU (6.3%), ESP (5.7%), GBR (5.1%), ITA (3.9%), BRA (2.0%), BEL (2.0%)]                                                                                                                                                        |
|            2 | Catering_Package     |         0.7 |             337 |            299 | [Standard (Coffee Only) (71.3%), No Food Plan (10.3%), Lunch Included (7.5%), standard (coffee only) (1.8%), STANDARD (COFFEE ONLY) (1.7%), Standard (Coffee Only) (0.8%), Standard (Coffee Only) (0.8%), Standard (Coffee Only) (0.8%)]                  |
|            0 | Course_Start_Date    |         0   |               0 |            666 | [2015-10-16 (0.5%), 2016-11-07 (0.5%), 2015-09-18 (0.5%), 2016-10-13 (0.5%), 2015-08-14 (0.5%), 2016-06-17 (0.4%), 2016-06-24 (0.4%), 2016-06-15 (0.4%)]                                                                                                  |
|            3 | Welcome_Gift_Type    |         0   |               0 |              4 | [Branded Notebook (50.8%), Water Bottle (29.1%), USB Drive (15.9%), Portable Charger (4.1%)]                                                                                                                                                              |
|            5 | Assigned_Lab_Config  |         0   |               0 |              9 | [Standard PC (Windows) (72.4%), Linux Workstation (18.4%), Laptop Docking Station (2.9%), MacOS Station (2.5%), Dual Monitor Setup (2.4%), High-GPU Unit (0.8%), Server Access Terminal (0.4%), Touch Screen Interface (0.2%)]                            |
|            7 | Lanyard_Color        |         0   |               0 |            225 | [Blue (49.5%), Black (20.9%), Red (10.2%), Orange (5.2%), Green (3.9%), BLUE (1.2%), blue (1.2%), Blue (0.6%)]                                                                                                                                            |
|            8 | Client_Category      |         0   |               0 |            455 | [SaaS & Software Houses (41.3%), Traditional IT & Telecomm (20.5%), Big Tech & Multinationals (16.7%), FinTech & Banking (6.6%), Industrial Tech & IoT (3.6%), saas & software houses (1.1%), SAAS & SOFTWARE HOUSES (1.0%), Non-Profit & EduTech (0.7%)] |





#### A few conclusions

We can clearly see that someone tried to get tricky and sneak in some Unicode into items such as “blue” with a hash in the middle, etc. So we’ll just take care of that with rejects. We also see a lot of question marks, which is weird, and some null values are labeled as “unknown.” Let’s use a function that fixes all of that.

**Lets try to normlize them**



```python
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

```






|   Unnamed: 0 | column               |   missing_% |   missing_count |   unique_count | top_8_cat                                                                                                                                                                                                                         |
|-------------:|:---------------------|------------:|----------------:|---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|            4 | Requested_Lab_Config |         2.7 |            1394 |              8 | [standard pc (windows) (78.4%), linux workstation (13.2%), <NA> (2.7%), dual monitor setup (2.1%), macos station (1.6%), laptop docking station (1.5%), highgpu unit (0.5%), touch screen interface (0.0%)]                       |
|            9 | Submission_Source    |         1.6 |             803 |              4 | [b2b platforms & resellers (85.5%), direct website registration (8.1%), dedicated sales team (4.5%), <NA> (1.6%), government procurement system (0.2%)]                                                                           |
|           10 | Payment_Terms        |         1.6 |             802 |              3 | [pay upon start (81.6%), prepaid (nonrefundable) (16.9%), <NA> (1.6%), refundable deposit (0.0%)]                                                                                                                                 |
|            6 | Enrollment_Type      |         1.1 |             563 |              4 | [general admission (71.1%), affiliated admission (23.8%), contractual agreement (3.6%), <NA> (1.1%), organizational arrangement (0.4%)]                                                                                           |
|            1 | Origin_Country       |         0.9 |             438 |            148 | [prt (41.6%), fra (11.0%), deu (6.9%), esp (6.2%), gbr (5.5%), ita (4.3%), bra (2.2%), bel (2.1%)]                                                                                                                                |
|            2 | Catering_Package     |         0.7 |             337 |              4 | [standard (coffee only) (79.4%), no food plan (11.6%), lunch included (8.3%), <NA> (0.7%), all inclusive (0.1%)]                                                                                                                  |
|            8 | Client_Category      |         0   |               2 |              7 | [saas & software houses (46.0%), traditional it & telecomm (22.8%), big tech & multinationals (18.8%), fintech & banking (7.4%), industrial tech & iot (4.1%), nonprofit & edutech (0.7%), defense & govtech (0.2%), <NA> (0.0%)] |
|            0 | Course_Start_Date    |         0   |               0 |            666 | [20151016 (0.5%), 20161107 (0.5%), 20150918 (0.5%), 20161013 (0.5%), 20150814 (0.5%), 20160617 (0.4%), 20160624 (0.4%), 20160615 (0.4%)]                                                                                          |
|            3 | Welcome_Gift_Type    |         0   |               0 |              4 | [branded notebook (50.8%), water bottle (29.1%), usb drive (15.9%), portable charger (4.1%)]                                                                                                                                      |
|            5 | Assigned_Lab_Config  |         0   |               0 |              9 | [standard pc (windows) (72.4%), linux workstation (18.4%), laptop docking station (2.9%), macos station (2.5%), dual monitor setup (2.4%), highgpu unit (0.8%), server access terminal (0.4%), touch screen interface (0.2%)]     |
|            7 | Lanyard_Color        |         0   |               0 |              5 | [blue (55.2%), black (23.3%), red (11.4%), orange (5.7%), green (4.3%)]                                                                                                                                                           |





MUCH BETTER :)



```python
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

```


    
![svg](<notebook_files/notebook_23_0.svg>)
    




Lets look closer at intresting ones



```python
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

```






|   Unnamed: 0 | column               | top_1                                              | top_2                                               | top_3                                             | top_4                                      | top_5                                                |
|-------------:|:---------------------|:---------------------------------------------------|:----------------------------------------------------|:--------------------------------------------------|:-------------------------------------------|:-----------------------------------------------------|
|            0 | Origin_Country       | prt: drop%=64.0, count=21098                       | fra: drop%=17.4, count=5581                         | deu: drop%=16.7, count=3496                       | esp: drop%=27.3, count=3129                | gbr: drop%=27.9, count=2803                          |
|            1 | Catering_Package     | standard (coffee only): drop%=42.3, count=40305    | no food plan: drop%=35.9, count=5868                | lunch included: drop%=40.9, count=4225            | <NA>: drop%=38.6, count=337                | all inclusive: drop%=80.6, count=36                  |
|            2 | Requested_Lab_Config | standard pc (windows): drop%=43.3, count=39793     | linux workstation: drop%=34.0, count=6698           | <NA>: drop%=40.3, count=1394                      | dual monitor setup: drop%=38.9, count=1050 | macos station: drop%=29.8, count=792                 |
|            3 | Assigned_Lab_Config  | standard pc (windows): drop%=47.5, count=36779     | linux workstation: drop%=26.3, count=9327           | laptop docking station: drop%=23.3, count=1471    | macos station: drop%=21.1, count=1268      | dual monitor setup: drop%=34.6, count=1242           |
|            4 | Enrollment_Type      | general admission: drop%=45.1, count=36093         | affiliated admission: drop%=29.9, count=12098       | contractual agreement: drop%=48.7, count=1831     | <NA>: drop%=42.3, count=563                | organizational arrangement: drop%=10.8, count=186    |
|            5 | Client_Category      | saas & software houses: drop%=35.9, count=23378    | traditional it & telecomm: drop%=42.9, count=11566  | big tech & multinationals: drop%=68.3, count=9527 | fintech & banking: drop%=17.2, count=3741  | industrial tech & iot: drop%=22.2, count=2065        |
|            6 | Submission_Source    | b2b platforms & resellers: drop%=44.6, count=43432 | direct website registration: drop%=17.7, count=4136 | dedicated sales team: drop%=24.6, count=2280      | <NA>: drop%=44.1, count=803                | government procurement system: drop%=19.2, count=120 |
|            7 | Payment_Terms        | pay upon start: drop%=29.4, count=41404            | prepaid (nonrefundable): drop%=99.8, count=8560     | <NA>: drop%=38.9, count=802                       | nan                                        | nan                                                  |





The compact table above is meant as a screening view: for each categorical feature it first keeps the frequent categories, then reports their dropout rate. I use it to decide which categorical variables deserve a cleaner final plot. `Lanyard_Color` and `Welcome_Gift_Type` do not show a meaningful business pattern, while `Payment_Terms`, `Client_Category`, `Submission_Source`, `Enrollment_Type`, and `Agent_ID` show large enough differences to keep investigating.


**A big red flag**
The data for payment terms doesn't make sense since the dropout rate is almost 100% for prepaid and nonrefundable. So this is either data leakage or something intentional, but it's definitely weird.



```python
normed_catted_df.groupby("Payment_Terms")["Dropped_Course"].agg(
    count="size",
    mean="mean",
)
```






| ('Unnamed: 0_level_0', 'Payment_Terms')   |   ('count', 'Unnamed: 1_level_1') |   ('mean', 'Unnamed: 2_level_1') |
|:------------------------------------------|----------------------------------:|---------------------------------:|
| pay upon start                            |                             41404 |                         0.294295 |
| prepaid (nonrefundable)                   |                              8560 |                         0.997547 |
| refundable deposit                        |                                 5 |                         0.6      |





**Lets see the dates**



```python
plot_dropout_over_time(df)

```


    
![svg](<notebook_files/notebook_31_0.svg>)
    







| ('Unnamed: 0_level_0', 'Course_Start_Date')   |   ('dropout_rate', 'Unnamed: 1_level_1') |   ('count', 'Unnamed: 2_level_1') |
|:----------------------------------------------|-----------------------------------------:|----------------------------------:|
| 2015-07                                       |                                 0.672337 |                              1117 |
| 2015-08                                       |                                 0.501007 |                              1986 |
| 2015-09                                       |                                 0.438793 |                              2851 |
| 2015-10                                       |                                 0.393498 |                              2676 |
| 2015-11                                       |                                 0.24368  |                               989 |
| 2015-12                                       |                                 0.404655 |                              1332 |
| 2016-01                                       |                                 0.325709 |                              1093 |
| 2016-02                                       |                                 0.384575 |                              1906 |
| 2016-03                                       |                                 0.366999 |                              2406 |
| 2016-04                                       |                                 0.429124 |                              2843 |
| 2016-05                                       |                                 0.38193  |                              2922 |
| 2016-06                                       |                                 0.440446 |                              3140 |
| 2016-07                                       |                                 0.334932 |                              2502 |
| 2016-08                                       |                                 0.365699 |                              2688 |
| 2016-09                                       |                                 0.406686 |                              3081 |
| 2016-10                                       |                                 0.462155 |                              3369 |
| 2016-11                                       |                                 0.439676 |                              2470 |
| 2016-12                                       |                                 0.433249 |                              1985 |
| 2017-01                                       |                                 0.431514 |                              1942 |
| 2017-02                                       |                                 0.369378 |                              2090 |
| 2017-03                                       |                                 0.38     |                              2750 |
| 2017-04                                       |                                 0.488416 |                              2633 |





#### Categorical Conclusions

The categorical analysis belongs in EDA because it reveals data-quality problems and predictive patterns. The actual normalization function should later be reused inside the preprocessing pipeline so the validation and official test rows receive the same cleaning.

Key conclusions:

- `Payment_Terms` is the strongest categorical warning sign. `prepaid (nonrefundable)` has almost 100% dropout in the training split, which may represent a real business rule or a leakage-like artifact. I will keep it, but I should later verify whether models rely on it too heavily.
- `Client_Category`, `Submission_Source`, and `Enrollment_Type` show meaningful differences in dropout rates and should be encoded.
- `Lanyard_Color` and `Welcome_Gift_Type` look weak or artificial, so they are candidates to drop unless later model evaluation shows value.
- `Course_Start_Date` has too many exact categories for the first categorical bar chart, but the time plot shows a clear period effect. I will treat the date carefully: it may improve AUC, but it may also capture period-specific conditions rather than a stable causal feature.
- Categorical missing values should be encoded explicitly, either as an `unknown/missing` category or through missingness indicators, because missingness is part of the registration process.


### Numeric cols


First, we have to separate numeric features into three subsets:

- the target
- true numeric features
- identifier numbers, such as client ID or agent ID, etc., which shouldn't really be treated as a number.



```python
all_num_cols = df.select_dtypes(include=["int64", "float64"]).columns
IDE_COLNAMES = ["Agent_ID", "Company_ID", "Client_ID"]
TARGET = "Dropped_Course"
ide_cols = [col for col in all_num_cols if col in IDE_COLNAMES]

num_cols = [col for col in all_num_cols if col not in IDE_COLNAMES and col != TARGET]
```

#### Identifier columns


We will analyze the `ide_cols` as categorial ones



```python
get_cat_smr(df, ide_cols)
```






|   Unnamed: 0 | column     |   missing_% |   missing_count |   unique_count | top_8_cat                                                                                                              |
|-------------:|:-----------|------------:|----------------:|---------------:|:-----------------------------------------------------------------------------------------------------------------------|
|            2 | Company_ID |        95.1 |           48295 |            169 | [nan (95.1%), 5181.0 (1.2%), 5013.0 (0.4%), 5194.0 (0.3%), 5119.0 (0.2%), 5024.0 (0.2%), 5185.0 (0.2%), 5025.0 (0.2%)] |
|            1 | Agent_ID   |        17.6 |            8928 |            196 | [184.0 (34.8%), nan (17.6%), 218.0 (10.4%), 104.0 (4.1%), 264.0 (3.7%), 219.0 (3.1%), 224.0 (2.0%), 205.0 (1.7%)]      |
|            0 | Client_ID  |         0   |               0 |          50771 | [72948 (0.0%), 17739 (0.0%), 32579 (0.0%), 46415 (0.0%), 12303 (0.0%), 19224 (0.0%), 73044 (0.0%), 45587 (0.0%)]       |






```python
df["Client_ID"].nunique() == len(df)
```




    True



So far, we see a few interesting patterns. One is that client ID is a unique identifier, so even if a client is returning, it wouldn’t have the same client ID. Anyway, that one is useless for us.

**About Agent & Company ID**

We have to remember that the clients we talk about are enterprise clients, meaning groups. The data says agent ID is an identifier for the agent or salesperson who listed the group, and company ID is an identifier for the company that did that. So we can assume that Nova also has internal agents but also works with external companies that handle it for them. That is why company ID is probably missing 95 % of the time.

That also explains why not all of them have an agent ID—some may have joined the program on their own, through a company, or through a documented reason. Even though we have a lot of nulls, that’s okay, because a missing value also tells us something. A dummy would probably solve it.



```python
ide_cols = [col for col in ide_cols if col != "Client_ID"]
```


```python
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

```


```python
plot_id_target_rate(df, "Agent_ID")
```


    
![svg](<notebook_files/notebook_43_0.svg>)
    







|   ('Unnamed: 0_level_0', 'Agent_ID') |   ('dropout_rate', 'Unnamed: 1_level_1') |   ('count', 'Unnamed: 2_level_1') |
|-------------------------------------:|-----------------------------------------:|----------------------------------:|
|                                  224 |                                 0.053846 |                              1040 |
|                                  104 |                                 0.126018 |                              2087 |
|                                  264 |                                 0.175726 |                              1895 |
|                                  220 |                                 0.257212 |                               416 |
|                                  158 |                                 0.323404 |                               705 |
|                                  nan |                                 0.338262 |                              8928 |
|                                  219 |                                 0.393308 |                              1584 |
|                                  184 |                                 0.400193 |                             17644 |
|                                  258 |                                 0.51005  |                               398 |
|                                  205 |                                 0.566362 |                               874 |
|                                  320 |                                 0.578624 |                               814 |
|                                  138 |                                 0.60066  |                               606 |
|                                  218 |                                 0.734562 |                              5263 |
|                                  139 |                                 0.740864 |                               602 |
|                                  129 |                                 0.794297 |                               491 |





**Agent Matters A lot!!!**

So we see that agents matter a lot. Some explanation could be that some are more aggressive sellers than others, or some work by different methods, et cetera, but it definitely has a big predictive value.



```python
plot_id_target_rate(
    df,
    "Company_ID",
    min_count=50,
    top_n=10,
)
```


    
![svg](<notebook_files/notebook_45_0.svg>)
    







|   ('Unnamed: 0_level_0', 'Company_ID') |   ('dropout_rate', 'Unnamed: 1_level_1') |   ('count', 'Unnamed: 2_level_1') |
|---------------------------------------:|-----------------------------------------:|----------------------------------:|
|                                   5035 |                                 0.02     |                                50 |
|                                   5185 |                                 0.073684 |                                95 |
|                                   5181 |                                 0.086677 |                               623 |
|                                   5024 |                                 0.09375  |                                96 |
|                                   5194 |                                 0.104294 |                               163 |
|                                   5025 |                                 0.109756 |                                82 |
|                                   5119 |                                 0.1875   |                               112 |
|                                   5010 |                                 0.424242 |                                66 |
|                                    nan |                                 0.424682 |                             48295 |
|                                   5013 |                                 0.669683 |                               221 |





So we see that generally booking through a company reduces the rate risk, except for this one company, but the problem is that the stats are so low that it's not really reliable.

Let's test the rate with a company or without a company to compare.



```python
df.groupby(df.Company_ID.notna()).Dropped_Course.agg(count="size", drop_rate="mean")
```






| ('Unnamed: 0_level_0', 'Company_ID')   |   ('count', 'Unnamed: 1_level_1') |   ('drop_rate', 'Unnamed: 2_level_1') |
|:---------------------------------------|----------------------------------:|--------------------------------------:|
| False                                  |                             48295 |                              0.424682 |
| True                                   |                              2476 |                              0.213651 |





Yeah. So definitely groups coming through a company are way less likely to drop, even though it's quite a small subset of the data.


#### Numeric Columns



```python
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

```


```python
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
```






|   Unnamed: 0 | column                      |   missing_% |   missing_count |   corr_target |   mean |   median |    std |   min |   max |   q25 |   q75 |   skew |
|-------------:|:----------------------------|------------:|----------------:|--------------:|-------:|---------:|-------:|------:|------:|------:|------:|-------:|
|            5 | Registration_Days_Before    |         4.2 |            2126 |         0.351 | 103.2  |     65   | 109.55 |     0 |   629 |    19 |   150 |   1.5  |
|            8 | Pre_Course_Supports_Tickets |         0   |               0 |        -0.3   |   0.51 |      0   |   0.76 |     0 |     5 |     0 |     1 |   1.47 |
|            6 | Prev_Course_Dropouts        |         0   |               0 |         0.204 |   0.1  |      0   |   0.44 |     0 |    21 |     0 |     0 |  15.55 |
|           11 | Registration_Changes        |         0   |               0 |        -0.147 |   0.18 |      0   |   0.59 |     0 |    21 |     0 |     0 |   6.9  |
|            9 | Physical_Course_Kits        |         1.6 |             819 |        -0.138 |   0.03 |      0   |   0.16 |     0 |     3 |     0 |     0 |   5.99 |
|           10 | Waiting_List_Days           |         0   |               0 |         0.068 |   4    |      0   |  23.21 |     0 |   391 |     0 |     0 |   9.24 |
|           12 | Returning_Client            |         0   |               0 |        -0.059 |   0.03 |      0   |   0.16 |     0 |     1 |     0 |     0 |   5.87 |
|            0 | Professionals_Count         |         0   |               0 |         0.056 |   1.83 |      2   |   0.51 |     0 |     4 |     2 |     2 |  -0.47 |
|            7 | Prev_Course_Attended        |         0   |               0 |        -0.05  |   0.12 |      0   |   1.55 |     0 |    61 |     0 |     0 |  22.04 |
|            4 | Theory_Hours                |         0   |               0 |         0.044 |   2.17 |      2   |   1.47 |     0 |    41 |     1 |     3 |   3.38 |
|            2 | Observers_Count             |         0   |               0 |        -0.032 |   0.01 |      0   |   0.09 |     0 |    10 |     0 |     0 |  47.66 |
|           13 | Daily_Tuition_Cost          |         0.1 |              65 |        -0.024 |  98.87 |     94.5 |  43.11 |     0 |  5400 |    75 |   117 |  37.13 |
|            3 | Practical_Hours             |         0   |               0 |         0.006 |   6.79 |      1   | 220.69 |    -5 | 10000 |     0 |     1 |  39.99 |
|            1 | Students_Count              |         0   |               4 |         0     |   8.55 |      0   | 290.88 |     0 |  9999 |     0 |     0 |  34.32 |





**Numeric missing values**

The numeric summary table gives the first pass: missingness, correlation with the target, central tendency, spread, and extreme values.

Main missing-value decisions:

- `Registration_Days_Before` has the most numeric missingness, about 4%. Its missingness does not materially change the dropout rate by itself, but the non-missing values are strongly related to dropout. I will impute it rather than drop rows. KNN is a reasonable candidate, but it should be compared against simpler median imputation inside validation.
- `Physical_Course_Kits` has moderate missingness and a weak direct relationship with the target. Median imputation plus a missingness flag is enough.
- `Daily_Tuition_Cost` has very few missing values, but the missing rows have a higher dropout rate. I should not drop official test rows; use a missingness flag and either median imputation or a simple train-fitted regression imputer if it improves validation.
- For other low-missingness numeric columns, use train-fitted imputation rather than row dropping.

Important leakage rule: the imputer values or imputer model must be fitted only on `train_data` / `X_train` and then applied to validation and official test data.



```python
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

```

    Waiting_List_Days
    391
    
    Prev_Course_Attended
    61, 60, 59, 58, 57, 56, 55, 54, 49, 48, 47, 46, 45, 44, 43, 42, 41
    
    Registration_Days_Before
    629.0, 626.0
    
    Practical_Hours
    10000, 5000
    
    Students_Count
    9999.0
    
    Daily_Tuition_Cost
    5400.0, 451.5, 375.5, 365.0, 349.63, 345.0, 332.57, 316.0, 314.1, 312.5, 309.5, 306.0, 300.0, 299.33, 295.0, 294.35
    


`Prev_Course_Attended` has high values but they look like a smooth tail rather than a clear error. `Daily_Tuition_Cost = 5400` is suspicious and should be flagged for later outlier handling, but most of the other high tuition values look plausible. `Students_Count = 9999` and extreme `Practical_Hours` values look like data-entry or placeholder errors rather than real observations.



```python
sus_cols = [
    col for col in sus_cols if col not in ["Prev_Course_Attended", "Daily_Tuition_Cost"]
]
for col in sus_cols:
    print(f"\n{col}")
    print(df[col].quantile([0.9999, 0.999, 0.995, 0.99, 0.95, 0.9]))
    br()

```

    
    Waiting_List_Days
    0.9999    391.0
    0.9990    330.0
    0.9950    176.0
    0.9900    100.3
    0.9500     17.0
    0.9000      0.0
    Name: Waiting_List_Days, dtype: float64
    
    
    Registration_Days_Before
    0.9999    629.0
    0.9990    622.0
    0.9950    531.0
    0.9900    447.0
    0.9500    326.0
    0.9000    277.0
    Name: Registration_Days_Before, dtype: float64
    
    
    Practical_Hours
    0.9999    10000.00
    0.9990        8.23
    0.9950        4.00
    0.9900        3.00
    0.9500        2.00
    0.9000        2.00
    Name: Practical_Hours, dtype: float64
    
    
    Students_Count
    0.9999    9999.0
    0.9990       3.0
    0.9950       2.0
    0.9900       2.0
    0.9500       1.0
    0.9000       0.0
    Name: Students_Count, dtype: float64
    


**Outlier candidates**

The clear invalid/suspicious values are:

- `Students_Count`: values above the 99.9th percentile jump from normal small counts to `9999`, so these should be treated as invalid placeholders.
- `Practical_Hours`: negative values and very large values such as `5000`/`10000` are not plausible course-hour values.

`Waiting_List_Days` and `Registration_Days_Before` have long tails, but they are not automatically errors: long waits or early registrations can happen. I will keep them for now and let binned target-rate plots show whether the tail carries useful signal.



```python
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

```




|   Unnamed: 0 | column          |   cap |   values_set_missing |
|-------------:|:----------------|------:|---------------------:|
|            0 | Students_Count  |  3    |                   43 |
|            1 | Practical_Hours |  8.23 |                   51 |





```python
corr_matrix = df_capped[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)

```




    <Axes: >




    
![svg](<notebook_files/notebook_58_1.svg>)
    



```python
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

corr_pairs = corr_pairs[corr_pairs["corr"].abs() >= CORR_THRESH].sort_values(
    "corr",
    key=abs,
    ascending=False,
)
corr_pairs
```






|   Unnamed: 0 | feature_1                | feature_2            |     corr |
|-------------:|:-------------------------|:---------------------|---------:|
|           74 | Prev_Course_Attended     | Returning_Client     | 0.443719 |
|           63 | Prev_Course_Dropouts     | Prev_Course_Attended | 0.367656 |
|           24 | Students_Count           | Daily_Tuition_Cost   | 0.294568 |
|           68 | Prev_Course_Dropouts     | Returning_Client     | 0.245439 |
|           12 | Professionals_Count      | Daily_Tuition_Cost   | 0.226151 |
|           59 | Registration_Days_Before | Waiting_List_Days    | 0.221273 |
|           36 | Practical_Hours          | Theory_Hours         | 0.207055 |





### Final EDA Plots

These are the curated plots I would keep for the report. Earlier tables and screening plots are useful during exploration, but these final plots communicate the main EDA conclusions more clearly.



```python
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

```


    
![svg](<notebook_files/notebook_61_0.svg>)
    



    
![svg](<notebook_files/notebook_61_1.svg>)
    



    
![svg](<notebook_files/notebook_61_2.svg>)
    



    
![svg](<notebook_files/notebook_61_3.svg>)
    



    
![svg](<notebook_files/notebook_61_4.svg>)
    



    
![svg](<notebook_files/notebook_61_5.svg>)
    



    
![svg](<notebook_files/notebook_61_6.svg>)
    



    
![svg](<notebook_files/notebook_61_7.svg>)
    


#### Final Plot Interpretation

- `Payment_Terms`: `prepaid (nonrefundable)` is an extreme signal, with dropout close to 100%, while `pay upon start` is far lower. This is the strongest categorical pattern, but it should be treated carefully because it may encode a business process that is very close to the target.
- `Client_Category`: dropout differs strongly by segment. `big tech & multinationals` is much higher than the dataset mean, while `nonprofit & edutech`, `fintech & banking`, and `industrial tech & iot` are lower. This supports keeping the business segment as a feature.
- `Submission_Source`: direct website and dedicated sales registrations have lower dropout than B2B reseller/platform traffic. Missing submission source behaves closer to the high-risk B2B platform group, so missingness should not be silently discarded.
- `Enrollment_Type`: organizational arrangements and affiliated admissions are lower risk than general admission and contractual agreement. This feature should be encoded.
- `Agent_ID`: agents differ dramatically, from very low dropout to very high dropout among frequent agents. `Agent_ID` should be treated as a categorical feature, not as a numeric quantity. To reduce overfitting, rare agents should be grouped or handled with regularized encoding.
- `Registration_Days_Before`: dropout rises as registration happens further before the course. The highest bin is far above the dataset mean, so this is a strong numeric signal. Missing values should be imputed, not dropped.
- `Pre_Course_Supports_Tickets`: rows with more support tickets have lower dropout in this data. This may mean engaged clients are more likely to stay, so the feature is useful even though the direction is not intuitive at first.
- `Course_Start_Date`: dropout changes over time, which suggests seasonality or period-specific business conditions. It can help prediction, but it should be handled carefully because exact dates can overfit to the historical period.

Overall EDA conclusion: the strongest visible signals are payment terms, registration timing, agent/company registration path, client segment, source/enrollment channel, and support-ticket engagement. The next step is to convert these EDA decisions into a train-fitted preprocessing pipeline: categorical normalization/encoding, missing-value imputation with missingness flags, and outlier handling for the clearly invalid numeric placeholders.


### Boundary Between EDA, Missing Values, And Later Sections

My interpretation of the assignment is:

- EDA should analyze the labeled training data and justify cleaning decisions. It is fine to define category normalization here because discovering dirty categories is part of data understanding.
- Missing-value completion belongs in Part A, but it should be implemented in a way that can be reused later in the modeling pipeline. The important point is not where the function is written; it is that fill statistics are fitted only on the training split.
- The official test file should not be used for target-based analysis because it has no target. It is okay to inspect its schema and missingness so the final pipeline can handle all rows.
- Outlier analysis is formally Part B, but identifying suspicious values during EDA is normal. The final outlier decisions and feature-engineering choices should be summarized again in Part B.

So the notebook can include exploratory tables, but the final report should be curated: show the few plots/tables that support decisions, then explain what changed in preprocessing and why.

