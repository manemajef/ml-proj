# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import marimo as mo


    df = pd.read_csv("Train_Data.csv")
    return df, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Helper functions
    """)
    return


@app.cell
def _(pd, plt, sns):
    TARGET_NAME = 'Dropped_Course'


    def plot_feature_dist(df: pd.DataFrame, colname: str) -> None:
        if colname not in df.columns:
            print(f"Error: Column '{colname}' not found in the DataFrame.")
            return

        s = df[colname]
        if pd.api.types.is_numeric_dtype(s):
            fig, axes = plt.subplots(1, 2, figsize=(8, 5))
            sns.boxenplot(data=df, y=colname, x=TARGET_NAME, ax=axes[0])
            axes[0].set_title(f'{colname} vs {TARGET_NAME}')
            sns.histplot(
                data=df,
                x=colname,
                hue=TARGET_NAME,
                kde=True,
                ax=axes[1],
                multiple="layer",
                alpha=0.5,
            )
            axes[1].set_title(f'{colname} Distribution Profile')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            top_categories = s.value_counts().head(20).index
            df_filtered = df[df[colname].isin(top_categories)]
            sns.countplot(
                data=df_filtered,
                y=colname,
                order=top_categories,
                ax=axes[0],
                palette="blues_r",
            )
            axes[0].set_title(f'Top 20 {colname} Overall Counts')
            sns.countplot(
                data=df_filtered,
                y=colname,
                hue=TARGET_NAME,
                order=top_categories,
                ax=axes[1],
            )
            axes[1].set_title(f'{colname} Split by Attrition')
        plt.suptitle(f'Exploratory Data Analysis for: {colname}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    return (plot_feature_dist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. First insights
    """)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df['Dropped_Course'].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we see a Huge percent of drops, over $40\%$ !
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Missing values
    """)
    return


@app.cell
def _(df, pd):
    def get_null_summary(df: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame({
            "null_count": df.isnull().sum(),
            "null_percent": df.isnull().mean() * 100,
        }).sort_values("null_count", ascending=False)
        return frame[frame["null_count"] > 0]


    null_summary = get_null_summary(df)
    null_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Conclusions so far

    - `Company_ID` is $95\%$ null, should probably drop it, or turn into a dummy variable (`ordered_through_company`)

    - `Agent_ID` Has a lot of nulls, but this feature might still hold usefull data due to different agents could use different "Selling" which if done too aggresivly could result in high cancelation rate per specific agent. needs to test this theory.

    - The next 3 features need additional exploration inorder to decide weather theyre nulls are a big problem or not, or weather they are usefull.

    - for the rest, could probably just drop the samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 `Company_ID`

    **Lets see whatsup with `Company_ID`**

    We will split the Data set into the 95% that dosnt have a companyID and the 5% that do and see weather the mean differes by a lot in regards to drop rate.
    """)
    return


@app.cell
def _(df):
    print(df.groupby(df['Company_ID'].isna())['Dropped_Course'].describe())
    print(
        "unique company_id list is too big: ", len(df['Company_ID'].unique().tolist()) > 10
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data suggests people invited by a Company are 2 times less likely to drop then ones who are not. since 3120 is a large enough $n$ to get a resnable mean, we should probably create a new boolean feature as suggested before, and drop this one. Before that let us check whats going on inside those 3120 samples. We also see that unique company_id's list is too big to do anything usefull with, sicne to start with we only have like 3k samples only.

    - [ ] TODO: test hypothesis first

    **Decision**: We will drop the `Company_ID` feature and use a dummy variable instead. Later, we could decide weather that dummy itself is usefull or not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 `Agent_ID`

    **Lets check out Agent_ID**

    1. start by seeing if any specific agent has a meaningfull difference in mean of drops
    2. see if the 20% who didnt use any agent has differenc ein drop means
    """)
    return


@app.cell
def _(df):
    unique_agents = len(df['Agent_ID'].unique().tolist())
    print("number of agents + 1 = ", unique_agents)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    yiches, we have a lot of agents ... Lets keep only the ones who has a lot of samples
    """)
    return


@app.cell
def _(df):
    agent_counts = df['Agent_ID'].value_counts().sort_values(ascending=False)
    K = 100
    large_agents = agent_counts[agent_counts >= K]
    large_agents
    return (large_agents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    cool we have a lot of agents to work with.
    """)
    return


@app.cell
def _(df):
    DROP_MEAN = float(df['Dropped_Course'].mean())
    DROP_MEAN
    return (DROP_MEAN,)


@app.cell
def _(DROP_MEAN, df, large_agents, np):
    df_large_agents = df[df['Agent_ID'].isin(large_agents.index)]
    agent_summary = df_large_agents.groupby('Agent_ID')['Dropped_Course'].agg([
        "count",
        "mean",
    ])
    # agent_summary['drop_percent'] = (agent_summary['mean'] * 100)
    agent_summary['se'] = np.sqrt(DROP_MEAN * (1 - DROP_MEAN) / agent_summary['count'])
    agent_summary['diff_from_global'] = agent_summary['mean'] - DROP_MEAN
    agent_summary["z_score"] = agent_summary["diff_from_global"] / agent_summary["se"]
    agent_summary = agent_summary.sort_values("z_score", key=np.abs, ascending=False)
    agent_summary_display = agent_summary.copy()

    percent_cols = ["mean", "se", "diff_from_global"]

    agent_summary_display[percent_cols] = (agent_summary_display[percent_cols] * 100).round(
        1
    )

    agent_summary_display = agent_summary_display.rename(
        columns={
            "mean": "drop_rate_%",
            "se": "se_%",
            "diff_from_global": "diff_from_global_%",
        }
    )

    agent_summary_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Big finding -- Agents matter.**

    Apperently some agents are terrible (as suspected) whilte some perform extremly well. we must use the data then. as for all the nulls, ill soon check thyre mean.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 `Registration_Days_Before`
    """)
    return


@app.cell
def _(df, plot_feature_dist):
    plot_feature_dist(df, 'Registration_Days_Before')
    return


@app.cell
def _(df, plt, sns):
    fig, axes = plt.subplots(1, 2)
    sns.histplot(data=df, x='Registration_Days_Before', hue='Dropped_Course', kde=True)
    axes[0].set_title('Distribution of Registration Days Before')
    sns.boxenplot(data=df, y='Registration_Days_Before', x='Dropped_Course', ax=axes[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that this feature is relevant. The treshhold where the mean changes is around $\approx 150$ days. Lets check to see whatsup with missing values
    """)
    return


@app.cell
def _(df):
    drops_150_plus = df[df['Registration_Days_Before'] >= 150]['Dropped_Course'].mean()
    print('mean for eearly regs', float(drops_150_plus))
    drops_days_null = df[df['Registration_Days_Before'].isna()]['Dropped_Course'].mean()
    print('mean for missing vals', float(drops_days_null))
    drops_up_2_150 = df[
        (df['Registration_Days_Before'] < 150) & (df['Registration_Days_Before'].notna())
    ]['Dropped_Course'].mean()
    print('mean for rest', float(drops_up_2_150))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we see that missing values rows have larger drop rate then late registrations, but not as high as early registration. we should probably fill it with median, and add a dummy variable to track weather it has reg day
    """)
    return


@app.cell
def _():
    # median_days_before = df['Registration_Days_Before'].median()
    # print(median_days_before)
    # df['Registration_Days_Before_IsNa'] = df['Registration_Days_Before'].isna().astype(int)
    # df['Registration_Days_Before'] = df['Registration_Days_Before'].fillna(
    #     median_days_before
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Filling missing values

    **So far weve:**

    - droppes `Company_ID` and switched it to a dummy
    - filled `Registration_Days_Before` with median

    **For the rest**
    since the rest dont take a large enough portion, we will use autmoated methods to fill the nulls.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Preprocessing Pipeline
    """)
    return


@app.cell
def _(pd):
    def basic_preproccess(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # add missing indicators for features with meaningful missingness
        for col in [
            "Agent_ID",
            "Registration_Days_Before",
            "Requested_Lab_Config",
            "Physical_Course_Kits",
        ]:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)

        # Company_ID is mostly missing, so keep only whether it exists
        df["has_company_id"] = df["Company_ID"].notna().astype(int)
        df = df.drop(columns=["Company_ID"])

        # Date features // check later if its actualy needed during feature engeenering phase
        df["Course_Start_Date"] = pd.to_datetime(df["Course_Start_Date"])
        df["Course_start_month"] = df["Course_Start_Date"].dt.month
        df["Course_start_weekday"] = df["Course_Start_Date"].dt.weekday
        df = df.drop(columns=["Course_Start_Date"])

        # Normalize strings
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip().str.lower()

        # Fill missing values
        # numeric_cols = df.select_dtypes(include=["number"]).columns
        # cat_cols = df.select_dtypes(include=["object"]).columns
        # df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median()) fill seperatly
        # df[cat_cols] = df[cat_cols].fillna("unknown") fill seperatly

        return df

    return (basic_preproccess,)


@app.cell
def _(basic_preproccess, pd):
    train_raw = pd.read_csv("Train_Data.csv")
    test_raw = pd.read_csv("Test_Data_No_Target.csv")

    train_clean = basic_preproccess(train_raw)
    test_clean = basic_preproccess(test_raw)
    return train_clean, train_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. First shot models so we have a baseline
    """)
    return


@app.cell
def _(train_clean):
    import random

    from sklearn.model_selection import train_test_split

    X = train_clean.drop(columns=["Dropped_Course", "Client_ID"])
    y = train_clean["Dropped_Course"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,  # important so we get same res every time
        stratify=y,
    )
    return X_train, X_val, train_test_split, y, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1. Raw Logistic Regression
    """)
    return


@app.cell
def _(X_train, X_val, pd, y_train, y_val):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import (
        SimpleImputer,
    )  # fill nas using this instead of in basoc_preproccess()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score


    def get_preprocess(X: pd.DataFrame):
        """
        Apply simple imputing for nulls, and standart scaler.
        """
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

        numeric_transofrmer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        return ColumnTransformer([
            ('num', numeric_transofrmer, numeric_cols),
            ('cat', cat_transformer, cat_cols),
        ])


    def get_lg_model(X: pd.DataFrame):
        """
        Return lg (Logistic Regression) Model.
        """
        lg_preprocess = get_preprocess(X)
        lg_model = Pipeline([
            ("preprocess", lg_preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        return lg_model


    lg_model = get_lg_model(X_train)
    lg_model.fit(X_train, y_train)
    lg_y_proba = lg_model.predict_proba(X_val)[:, 1]
    RAW_LG_SCORE = roc_auc_score(y_val, lg_y_proba)

    print(f"\n\nLogsitic Regression first score: {RAW_LG_SCORE}\n\n")
    return Pipeline, get_lg_model, get_preprocess, roc_auc_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Now just test to see if the work we did in basic pre_proccess actualy gave value**
    """)
    return


@app.cell
def _(get_lg_model, roc_auc_score, train_raw, train_test_split, y):
    def test_preprocessed_data():
        """
        We wrap it in a function since we dont want all these *_raw variables in main memory
        """
        X_raw, y_raw = (
            train_raw.drop(columns=['Dropped_Course', 'Client_ID']),
            train_raw["Dropped_Course"],
        )
        X_raw_train, X_raw_val, y_raw_train, y_raw_val = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=42, stratify=y
        )
        lg_raw_model = get_lg_model(X_raw_train)
        lg_raw_model.fit(X_raw_train, y_raw_train)

        print(
            "The raw score before preprocess is: ",
            roc_auc_score(y_raw_val, lg_raw_model.predict_proba(X_raw_val)[:, 1]),
        )


    test_preprocessed_data()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1.1 CONCLUSIONS FROM LOGISTIC REGRESSION

    - **Processed Data Logistic Model Score:** 0.89875

    - **Unprocessed Data Logistic Model Score:** 0.90675

    - The logistic regression scored very well, even without feature engineering or complex transformations. This suggests the data is highly linear.

    - The unprocessed data performed slightly higher than the processed data, which could be due to the data processing.
      - This will be addressed during the feature engineering phase. KEY CHANGES MADE
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Random Forest (Raw)
    """)
    return


@app.cell
def _(
    Pipeline,
    X_train,
    X_val,
    get_preprocess,
    pd,
    roc_auc_score,
    y_train,
    y_val,
):
    from sklearn.ensemble import RandomForestClassifier


    def get_rf_model(X: pd.DataFrame) -> Pipeline:
        return Pipeline([
            ("preprocess", get_preprocess(X)),
            (
                'model',
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42
                ),
            ),
        ])


    rf_model = get_rf_model(X_train)
    rf_model.fit(X_train, y_train)
    rf_y_proba = rf_model.predict_proba(X_val)[:, 1]

    RAW_RF_SCORE = roc_auc_score(y_val, rf_y_proba)
    print(f"\n\nRandom Forest Basline AOC: {RAW_RF_SCORE}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2.1. Conclusions from Random Forest

    **Random Forest Basline AOC:** 0.89430

    - It scores almost exaclty the same as the Logistic Regression model.
    - Since Random Forest are less sensitive to data scale feature engeenring etc, I expect that it will be difficult to get the Forest score much higher, and therefor put most effort in opreparing for logistic regression, Which is likely to score better after data prep.
    """)
    return


if __name__ == "__main__":
    app.run()
