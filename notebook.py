import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Group 27 — Course-Drop Prediction (Nova Academy)

    **Submitters:** Rotem David Semah (ID: `211396593`) · Ron Drach (ID: `213915499`)

    ---

    This notebook follows the project from understanding the data through preparation, modelling, evaluation, and interpretation.
    """)
    return


@app.cell
def _():
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

    from IPython.display import display

    warnings.filterwarnings('ignore')
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 220
    plt.rcParams['svg.fonttype'] = 'path'

    TRAIN_PATH = 'data/Train_Data.csv'
    TEST_PATH = 'data/Test_Data_No_Target.csv'
    TARGET = 'Dropped_Course'
    SEED = 42
    pd.set_option('display.max_columns', None)  # noqa: B018


    # pragma: no cover
    def load_raw(path: str) -> pd.DataFrame:
        return pd.read_csv(path, parse_dates=["Course_Start_Date"])

    return (
        CatBoostClassifier,
        LGBMClassifier,
        LogisticRegression,
        MLPClassifier,
        OneHotEncoder,
        Path,
        SEED,
        StandardScaler,
        TARGET,
        TEST_PATH,
        TRAIN_PATH,
        XGBClassifier,
        auc,
        average_precision_score,
        classification_report,
        confusion_matrix,
        display,
        load_raw,
        log_loss,
        np,
        pd,
        plt,
        precision_recall_curve,
        rankdata,
        roc_auc_score,
        roc_curve,
        shap,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Business understanding

    Nova Academy prepares cloud environments, catering, equipment, and classroom capacity before each B2B course begins. A cancellation therefore wastes prepared resources and can leave capacity that could have been offered to another group.

    Our goal is to estimate cancellation risk for new registrations early enough to support operational decisions. The assignment requires a continuous `Drop_Probability` output and evaluates its ranking quality with ROC-AUC; the minimum required AUC is 0.70.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Data loading & first look

    Two files are provided:

    - `Train_Data.csv` — historical registrations **with** the `Dropped_Course`
      label.
    - `Test_Data_No_Target.csv` — registrations to score, **without** the label.

    Each row is one registration, identified by `Client_ID`. We first inspect inferred types, missingness, cardinality, common values, and zeros before deciding how any column should be treated.
    """)
    return


@app.cell
def _(TEST_PATH, TRAIN_PATH, display, load_raw, pd):
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
    for _df in (train_raw, test_raw):
        for _col in ("Agent_ID", "Company_ID"):
            _df[_col] = _df[_col].astype("string")
    return test_raw, train_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the dictionary tells us.**

    - `Client_ID` is unique per row — an identifier, never a feature.
    - `Agent_ID` and `Company_ID` were inferred as numeric even though they are identifiers, so we convert them to strings after this first inspection. `Company_ID` is also missing for most rows.
    - Several text fields have unexpectedly high cardinality. We inspect their raw values later before deciding whether that reflects real variety or inconsistent spelling.
    - The numeric summary below lets us look for suspicious ranges and extreme values.
    """)
    return


@app.cell
def _(train_raw):
    train_raw.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Target balance

    We first check whether one target class is rare enough to require special treatment during training or evaluation.
    """)
    return


@app.cell
def _(TARGET, display, pd, plt, train_raw):
    target_counts = train_raw[TARGET].value_counts().sort_index()
    target_rate = train_raw[TARGET].value_counts(normalize=True).sort_index()
    balance = pd.DataFrame({'count': target_counts, 'rate_%': (target_rate * 100).round(1)})
    balance.index = ['0 = completed', '1 = dropped']
    display(balance)
    _ax = target_rate.mul(100).plot.bar(color=['#4c72b0', '#c44e52'], figsize=(5, 3.5))
    _ax.set_xticklabels(['completed (0)', 'dropped (1)'], rotation=0)
    _ax.set_ylabel('share of orders (%)')
    _ax.set_title('Target balance — Dropped_Course')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The classes are reasonably balanced: about 59% completed and 41% dropped
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Exploratory Data Analysis

    We begin with the target and date coverage, then inspect missingness, categorical quality, and numeric relationships.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 Train and test dates

    We plot the monthly drop rate across the _training_ period and overlay where training ends and where the hidden test window ends.
    """)
    return


@app.cell
def _(TARGET, plt, test_raw, train_raw):
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
    _ax = monthly.plot(marker='o', figsize=(12, 4))
    _ax.axhline(
        train_raw[TARGET].mean() * 100, ls='--', color='grey', label='train average'
    )
    _ax.axvline(train_end, ls='--', color='green', label=f'train ends ({train_end.date()})')
    _ax.axvline(test_end, ls=':', color='red', label=f'test ends ({test_end.date()})')
    _ax.set_xlim(train_raw['Course_Start_Date'].min(), test_end)
    _ax.set_ylabel('drop rate (%)')
    _ax.set_title('Drop rate over time — training period and the hidden test horizon')
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Training covers July 2015 through April 2017. The test set begins at the end of that period and continues through August 2017, so the prediction task is temporal: learn from earlier registrations and score a later window.

    The monthly drop rate also changes across the training period. Because a random split would mix earlier and later regimes, we define validation chronologically and later compare the result with a random split.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Missing values

    We compare missingness in train and test, then ask whether _the fact of being missing_ is itself predictive.
    """)
    return


@app.cell
def _(display, pd, test_raw, train_raw):
    missing_compare = pd.DataFrame({
        "train_missing_%": train_raw.isna().mean().mul(100).round(2),
        "test_missing_%": test_raw.isna().mean().mul(100).round(2),
    })
    missing_compare = missing_compare[
        (missing_compare["train_missing_%"] > 0) | (missing_compare["test_missing_%"] > 0)
    ].sort_values("train_missing_%", ascending=False)
    display(missing_compare)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most train/test missingness rates are close. We next check whether the presence of a value is associated with the target.
    """)
    return


@app.cell
def _(TARGET, display, pd, train_raw):
    missingness_cols = [
        'Company_ID',
        'Agent_ID',
        'Registration_Days_Before',
        'Physical_Course_Kits',
        'Daily_Tuition_Cost',
        'Payment_Terms',
    ]
    rows = []
    for _col in missingness_cols:
        stats = (
            train_raw
            .assign(is_missing=train_raw[_col].isna())
            .groupby('is_missing')[TARGET]
            .agg(count='size', drop_rate='mean')
        )
        for is_missing, r in stats.iterrows():
            rows.append({
                'column': _col,
                'is_missing': is_missing,
                'count': int(r['count']),
                'drop_rate_%': round(r['drop_rate'] * 100, 1),
            })
    display(pd.DataFrame(rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rows without a `Company_ID` have a noticeably higher drop rate, and `Agent_ID` presence also separates groups. This motivates explicit presence flags instead of replacing missing identifiers with a typical value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Inspecting categorical values

    Several text columns have far more distinct values than their meanings suggest: hundreds of payment terms, colors, and enrollment types would be surprising. We inspect the raw labels before deciding whether the cardinality is real.
    """)
    return


@app.cell
def _(train_raw):
    TEXT_COLS = list(train_raw.select_dtypes(include=['object']).columns)


    def count_unique_vals(df, col):
        return df[col].nunique()


    def most_common_cats(df, col, n=15):
        return df[col].value_counts(normalize=True).head(n)


    N_COUNT = 8
    for _col in TEXT_COLS:
        top_values = most_common_cats(train_raw, _col, N_COUNT)
        cats = [f'{value!r}: ({share * 100:.1f}%)' for value, share in top_values.items()]
        # N_COUNT = 15
        # for col in TEXT_COLS:
        #     top_values = most_common_cats(train_raw, col)
        cats_str = '\n'.join((' | '.join(cats[i : i + 3]) for i in range(0, len(cats), 3)))
        #     print(f"""
        # {'=' * 80}
        # Column: {col}
        # Unique values: {count_unique_vals(train_raw, col)}
        # Top {len(top_values)} categories:
        # {",   ".join(f"{value}: ({share * 100:.1f}%)" for value, share in top_values.items())}
        # """)
        print(
            f"\n{'=' * 80}\n\nColumn: {_col}\nUnique values: {count_unique_vals(train_raw, _col)}\n\nTop {len(top_values)} categories:\n\n{cats_str}\n"
        )
    return (TEXT_COLS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The raw values explain much of the inflated cardinality. Labels such as `'BLUE'`, `'blue'`, and `'  Blue  '` describe the same category but are stored separately; punctuation and placeholder strings create similar splits in other fields. Before treating these columns as genuinely high-cardinality, we normalize the obvious formatting variants and measure how many levels remain.
    """)
    return


@app.cell
def _(pd):
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


    return COMMON_NANS, COUNTRY_ALIASES, canonicalize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We normalize case, surrounding whitespace, repeated spaces, and injected punctuation. Placeholder labels such as `Unknown` and `?` become missing values rather than new categories. The same deterministic cleaning function will be applied to train and test.
    """)
    return


@app.cell
def _(COMMON_NANS, COUNTRY_ALIASES, TEXT_COLS, canonicalize, pd):
    CAT_COLS = TEXT_COLS + ['Agent_ID', 'Company_ID']


    def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalise every categorical, then map junk placeholders to NaN."""
        df = df.copy()
        for _col in CAT_COLS:
            s = canonicalize(df[_col])
            df[_col] = s.mask(s.isin(COMMON_NANS))
        df['Origin_Country'] = df['Origin_Country'].replace(COUNTRY_ALIASES)
        return df

    return (normalize_cats,)


@app.cell
def _(TEXT_COLS, display, normalize_cats, pd, train_raw):
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
    return (clean_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The before/after table confirms that most of the apparent variety was formatting noise: `Payment_Terms` falls from 236 raw labels to 3 cleaned levels, and `Client_Category` from 505 to 7. Columns that were already consistent remain unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Which categories actually relate to dropping?

    We start with business fields that have only a few cleaned levels, where a direct plot remains readable. Country and identifiers need separate treatment because hundreds of levels would make the same plot misleading.
    """)
    return


@app.cell
def _(TARGET, clean_train, plt):
    def plot_dropout_by_category(df, col, min_count=50, top_n=10, ax=None):
        stats = df.groupby(col, dropna=False)[TARGET].agg(drop_rate='mean', count='size')
        stats = (
            stats[stats['count'] >= min_count]
            .sort_values('count', ascending=False)
            .head(top_n)
            .sort_values('drop_rate')
        )
        labels = [f"{i} (n={int(r['count'])})" for i, r in stats.iterrows()]
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        ax.barh(labels, stats['drop_rate'] * 100, color='#4c72b0')
        overall = df[TARGET].mean() * 100
        ax.axvline(overall, ls='--', color='red', label=f'mean ({overall:.1f}%)')
        ax.set_xlabel('drop rate (%)')
        ax.set_title(f'Drop rate by {col}')
        ax.legend()
        return stats


    _fig, _axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_dropout_by_category(
        clean_train, 'Payment_Terms', min_count=20, top_n=5, ax=_axes[0, 0]
    )
    plot_dropout_by_category(
        clean_train, 'Client_Category', min_count=100, top_n=8, ax=_axes[0, 1]
    )
    plot_dropout_by_category(
        clean_train, 'Submission_Source', min_count=100, top_n=6, ax=_axes[1, 0]
    )
    plot_dropout_by_category(
        clean_train, 'Enrollment_Type', min_count=100, top_n=6, ax=_axes[1, 1]
    )
    plt.tight_layout()
    plt.show()
    return (plot_dropout_by_category,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The clearest surprise is `Payment_Terms`: prepaid, non-refundable registrations drop more often than pay-on-start registrations, the opposite of what we expected. Two possible explanations are that these terms are assigned to riskier deals in advance or that the field is updated later in the registration process. We return to this question after fitting the model.

    The other plots also show useful separation. Direct-website and dedicated-sales registrations drop less often than reseller traffic, organisational enrollment is lower-risk than general admission, and client segments differ. We carry these signals into modelling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A closer look at high-cardinality categories

    `Origin_Country`, `Agent_ID`, and `Company_ID` cannot be judged from an unfiltered chart containing every level. We first require enough rows for a rate to be interpretable, then use Portugal as a compact case because it is both the largest country group and far from the overall drop rate.
    """)
    return


@app.cell
def _(TARGET, clean_train, display, np, pd, plt):
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
    top_by_size = country_stats.sort_values('count', ascending=False).head(country_top_n)
    extreme_by_lift = (
        country_stats[country_stats['count'] >= country_min_n]
        .iloc[
            lambda d: (
                d['lift_pp'].abs().sort_values(ascending=False).index.map(d.index.get_loc)
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


    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_country_dropout(
        top_by_size, f'Drop rate by largest {country_top_n} countries', _axes[0]
    )
    # Pick countries whose drop rate is farthest from the overall mean, after filtering tiny countries.
    plot_country_dropout(
        extreme_by_lift, f'Most unusual country drop rates (n >= {country_min_n})', _axes[1]
    )
    plt.tight_layout()
    plt.show()
    display(
        country_stats
        .sort_values('count', ascending=False)
        .head(country_top_n)[['count', 'drop_rate_pct', 'lift_pp']]
        .round(2)
    )  # ignore countries with too few rows for a stable rate  # sort by distance from the overall drop rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Portugal contains 26,429 registrations and has a 63.8% drop rate, making it both the largest country group and the clearest geographic difference. We use it to investigate whether country overlaps with agents, channels, or other parts of the acquisition process.
    """)
    return


@app.cell
def _(TARGET, clean_train, display, np):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compared with all other countries, Portugal remains clearly different. We next inspect the identifier fields as categories, not numbers, to see whether they show related structure.
    """)
    return


@app.cell
def _(TARGET, clean_train, display, plot_dropout_by_category, plt, train_raw):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_dropout_by_category(clean_train, 'Agent_ID', min_count=150, top_n=12, ax=_axes[0])
    company_presence = train_raw.groupby(train_raw['Company_ID'].notna())[TARGET].agg(
        count='size', drop_rate='mean'
    )
    company_presence.index = ['no company_id', 'has company_id']
    _axes[1].bar(
        company_presence.index,
        company_presence['drop_rate'] * 100,
        color=['#c44e52', '#55a868'],
    )
    _axes[1].set_ylabel('drop rate (%)')
    _axes[1].set_title('Drop rate by Company_ID presence')
    plt.tight_layout()
    plt.show()
    display(company_presence)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Frequent agents have different drop rates, while registrations with a `Company_ID` drop less often (21.2% versus 42.5%). These relationships may overlap with geography, so we perform a small check: does knowing the agent improve country prediction over always guessing the most common country?
    """)
    return


@app.cell
def _(SEED, clean_train, display, pd, train_test_split):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Agent-based prediction raises country accuracy from 0.391 to 0.421, indicating modest overlap between the two fields. Both are included using the compact representation introduced during preparation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 Numeric features: summary, correlation, and suspects

    We now inspect numeric ranges, distributions, and their linear correlations with the target.
    """)
    return


@app.cell
def _(TARGET, display, pd, train_raw):
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
    return (num_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The maximum values reveal several likely data errors: `Students_Count` reaches 9999, and `Practical_Hours` contains both negative values and values up to 10000. We leave the raw values unchanged for this first inspection and decide how to handle them in the outlier section.
    """)
    return


@app.cell
def _(TARGET, num_cols, plt, sns, train_raw):
    corr = train_raw[num_cols + [TARGET]].corr()
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm", center=0
    )
    plt.title("Numeric correlation heatmap (incl. target)")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No raw numeric feature has an extremely strong Pearson correlation with the target. `Registration_Days_Before` and `Pre_Course_Supports_Tickets` stand out most, while inter-feature correlations are generally modest. Because Pearson correlation measures linear association and is sensitive to extremes, we next use binned drop rates to inspect the shape of the strongest relationships.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 Numeric drop-rate profiles

    Binning a couple of the more predictive numeric features shows _how_ risk moves with them (not just whether they correlate linearly).
    """)
    return


@app.cell
def _(TARGET, pd, plt, train_raw):
    def plot_dropout_by_bins(df, col, bins=8, ax=None):
        tmp = df[[col, TARGET]].dropna().copy()
        tmp['bin'] = pd.qcut(tmp[col], q=bins, duplicates='drop')
        stats = tmp.groupby('bin', observed=True)[TARGET].mean().mul(100)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        stats.plot.bar(ax=ax, color='#4c72b0')
        ax.axhline(df[TARGET].mean() * 100, ls='--', color='red', label='mean')
        ax.set_ylabel('drop rate (%)')
        ax.set_title(f'Drop rate by {col} bins')
        ax.legend()
        ax.tick_params(axis='x', labelrotation=45)
        return stats


    _fig, _axes = plt.subplots(1, 2, figsize=(15, 4.5))
    plot_dropout_by_bins(train_raw, 'Registration_Days_Before', bins=8, ax=_axes[0])
    plot_dropout_by_bins(train_raw, 'Pre_Course_Supports_Tickets', bins=6, ax=_axes[1])
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drop rate rises across longer registration lead times, which suggests that plans are more likely to change when courses are booked far in advance. More pre-course support tickets are associated with lower dropping, suggesting that early engagement may reflect stronger commitment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.7 EDA conclusions

    Several observations now guide preparation and modelling:

    - Missing `Company_ID`, support activity, registration channel, enrollment type, and lead time all separate groups with different drop rates. Together, these patterns suggest a broader difference in buyer commitment.
    - `Payment_Terms` is unusually strong and counter-intuitive. We include it and later test how much XGBoost depends on it.
    - Country and agent both contain signal and overlap slightly. Their many levels require a compact encoding instead of a large one-hot expansion.
    - The later test window and changing monthly rates make time-aware validation important. We derive calendar and trend features, then test the time index after selecting a model.
    - Some numeric values are clearly suspicious, while other large values may be legitimate rare cases. We will correct only the values for which we have evidence of an error.

    These conclusions give us a modelling hypothesis: a flexible model may capture the combined effects better than a linear baseline. We test that hypothesis in the model comparison.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Missing-value handling & outlier analysis

    We now turn the EDA findings into reproducible preparation rules. The same fitted rules must be applied to later data, but the exact missing-value treatment can differ by model family.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 Outliers: identify, justify, cap

    We look for values that are physically impossible or absurdly far from the bulk.
    """)
    return


@app.cell
def _(display, num_cols, pd, test_raw, train_raw):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The test set introduces no new forms of corruption, suggesting the same cleaning policy can be safely shared. Comparing the maximum values to the 99th percentile helps identify columns with extreme outliers:
    """)
    return


@app.cell
def _(pd, plt, sns, test_raw, train_raw):
    TAIL_CHECK_COLS = ['Students_Count', 'Practical_Hours', 'Daily_Tuition_Cost']
    tail_long = []
    for split, df in [('train', train_raw), ('test', test_raw)]:
        for _col in TAIL_CHECK_COLS:
            s = df[_col].dropna().rename('value').to_frame()
            s['split'] = split
            s['column'] = _col
            tail_long.append(s)
    tail_long = pd.concat(tail_long, ignore_index=True)
    _fig, _axes = plt.subplots(1, len(TAIL_CHECK_COLS), figsize=(15, 4), sharey=False)
    for _ax, _col in zip(_axes, TAIL_CHECK_COLS):
        _sub = tail_long[tail_long['column'] == _col]
        sns.boxplot(
            data=_sub,
            x='split',
            y='value',
            hue='split',
            order=['train', 'test'],
            palette=['#4c72b0', '#dd8452'],
            showfliers=True,
            ax=_ax,
            legend=False,
        )
        for xpos, split in enumerate(['train', 'test']):
            split_values = _sub.loc[_sub['split'] == split, 'value']
            max_value = split_values.max()
            n_at_max = int((split_values == max_value).sum())
            _ax.annotate(
                f'max={max_value:g}\nn={n_at_max}',
                xy=(xpos, max_value),
                xytext=(0, 7),
                textcoords='offset points',
                fontsize=8,
                ha='center',
            )
        positive = _sub.loc[_sub['value'] > 0, 'value']
        if not positive.empty:
            _ax.set_yscale('log')
            _ax.set_ylim(bottom=max(positive.min() * 0.7, 0.1))
        _ax.set_title(_col)
        _ax.set_xlabel('')
        _ax.set_ylabel('raw value (log scale)')
    _fig.suptitle('Raw tail check for candidate capped columns')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We only alter values that look like data-entry errors, rather than applying a statistical rule to every rare observation. Keeping the rows preserves their other information, while clipping prevents the obvious placeholders from dominating a feature.

    Based on the suspect-column screen and the boxplots, we apply three caps:

    - `Students_Count <= 10`: the values beyond the observed low-count support are repeated `9999` placeholders in both train and test. The cap keeps those rows as large groups without treating 9999 as a real count.
    - `Practical_Hours` in `[0, 12]`: negative values are impossible, and `5000`/`10000` are clear placeholders. A 12-hour upper bound still allows a long practical day and prevents corrupted placeholder values from distorting the feature space.
    - `Daily_Tuition_Cost <= 600`: train has a single `5400` value, while the test maximum is 510. A cap of 600 leaves the observed test range untouched and prevents one corrupted training value from dominating cost calculations.

    Other flagged count columns (`Prev_Course_Dropouts`, `Prev_Course_Attended`, `Registration_Changes`, and test-side `Waiting_List_Days`) have long but plausible tails, so we leave them unchanged and restrict clipping to the three apparent data-entry errors above.
    """)
    return


@app.cell
def _(display, pd, plt, test_raw, train_raw):
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


    cap_rows = []
    for _col, rule in CAP_RULES.items():
        lo, hi = (rule['lower'], rule['upper'])
        train_changed = (
            train_raw[_col].notna() & apply_cap(train_raw[_col], lo, hi).ne(train_raw[_col])
        ).sum()
        test_changed = (
            test_raw[_col].notna() & apply_cap(test_raw[_col], lo, hi).ne(test_raw[_col])
        ).sum()
        action = f'clip to [{lo}, {hi}]' if lo is not None else f'clip to <= {hi}'
        cap_rows.append({
            'column': _col,
            'raw_train_min': train_raw[_col].min(),
            'raw_train_max': train_raw[_col].max(),
            'problem': rule['problem'],
            'action': action,
            'train_rows_affected': int(train_changed),
            'test_rows_affected': int(test_changed),
            'reason': rule['reason'],
        })
    display(pd.DataFrame(cap_rows))
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 6), sharey='row')
    for j, (_col, rule) in enumerate(CAP_RULES.items()):
        before = train_raw[_col].dropna()
        after = apply_cap(train_raw[_col], rule['lower'], rule['upper']).dropna()
        _axes[0, j].hist(before, bins=50, color='#c44e52')
        _axes[0, j].set_yscale('log')
        _axes[0, j].set_title(f'{_col}: raw')
        _axes[0, j].set_xlabel(f'max={before.max():g}')
        _axes[1, j].hist(after, bins=30, color='#55a868')
        _axes[1, j].set_yscale('log')
        _axes[1, j].set_title(f'{_col}: clipped')
        _axes[1, j].set_xlabel(f'max={after.max():g}')
    _axes[0, 0].set_ylabel('count (log)')
    _axes[1, 0].set_ylabel('count (log)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The before/after distributions show that the caps remove isolated invalid tails while
    preserving the bulk of each feature.

    One more data-quality issue appears in the historical counters. Some rows have more
    recorded prior dropouts than prior attended courses, so those fields are not a clean
    numerator/denominator pair.
    """)
    return


@app.cell
def _(train_raw):
    impossible = train_raw[
        train_raw["Prev_Course_Dropouts"] > train_raw["Prev_Course_Attended"]
    ]
    print(f"rows where historical dropouts exceed historical attended: {len(impossible)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We keep both historical counters and combine them in the client-history feature below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Missing-value policy

    One missing-value policy would not suit every model family:

    - Categorical missingness becomes an explicit `"missing"` level on both preprocessing paths. This preserves the possibility that absence itself carries information.
    - `Agent_ID` and `Company_ID` also receive presence flags because EDA showed a clear difference between present and missing groups. Their high cardinality is handled separately in feature engineering.
    - Models that support numeric missing values natively can retain `NaN` and learn how to route it. Models that require a complete numeric matrix receive medians learned from the training partition only.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Feature engineering & dimensionality

    Based on the EDA findings and domain questions, we create features that expose relationships more directly or represent high-cardinality fields more compactly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.1 The engineered features and their rationale

    | Raw signal | Engineered feature(s) | Reason for testing it |
    | --- | --- | --- |
    | `Course_Start_Date` | `start_month`, `start_week`, `start_dow`, `days_since_epoch` | Month/week represent possible yearly seasonality, weekday represents scheduling patterns, and the linear index represents the longer-term shift seen in Section 3.1. |
    | Participant counts | `total_participants`, `prof_share` | Total size and professional share describe group composition more directly than three separate counts. |
    | Practical/theory hours | `total_hours`, `practical_share` | Total duration and hands-on share distinguish courses with the same raw hour count but different structure. |
    | Client history | `prev_drop_rate = dropouts / (attended + 1)` | Combines previous dropouts and attendance into one history signal; the `+1` handles clients with no attended courses. |
    | Tuition cost and hours | `cost_x_days` | Combines price and course length so the model can consider their interaction. |
    | Requested vs assigned lab | `got_requested_lab` | Captures whether the assigned lab configuration matches the original request. |
    | Missing company/agent IDs | `has_company_id`, `has_agent_id` | Preserves the presence differences observed in Section 3.2 even when a raw identifier is removed. |
    | Agent/company/country IDs | frequency encodings and native categories | Retains identity and commonness information while avoiding a wide dummy matrix. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.2 Dimensionality

    The main expansion risk comes from identifiers: `Agent_ID` has 204 cleaned levels and `Origin_Country` has 154. Different model families therefore need different preparation paths.

    Models that require numeric inputs receive rare-level grouping, one-hot encoding, training-median imputation, and scaling. Boosted-tree implementations with native categorical support can work with category labels directly, so the matrix does not need one dummy column per agent or country. We also add one frequency feature per high-cardinality identifier and remove raw `Company_ID`, retaining only its frequency and presence flag.

    During validation, frequency maps are learned from the earlier training partition. For the final submission, we compute frequencies across the available train and test features, without using `Dropped_Course`. This transductive step gives each identifier one consistent frequency at scoring time.
    """)
    return


@app.cell
def _(display, normalize_cats, np, pd, test_raw, train_raw):
    def make_freq_maps(*dfs):
        """Label-free frequency of each ID value across the supplied frames."""
        combined = pd.concat([normalize_cats(d) for d in dfs], ignore_index=True)
        return {
            _col: combined[_col].value_counts(normalize=True)
            for _col in ('Agent_ID', 'Company_ID', 'Origin_Country')
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
        out['kits_per_participant'] = df['Physical_Course_Kits'] / total.replace(0, np.nan)
        out['tickets_per_participant'] = df['Pre_Course_Supports_Tickets'] / total.replace(
            0, np.nan
        )
        out['got_requested_lab'] = (
            df['Requested_Lab_Config'] == df['Assigned_Lab_Config']
        ).astype(float)
        out['has_company_id'] = df['Company_ID'].notna().astype(int)
        out['has_agent_id'] = df['Agent_ID'].notna().astype(int)
        for _col in ('Agent_ID', 'Company_ID', 'Origin_Country'):
            out[f'{_col}_freq'] = (
                df[_col].map(freq_maps[_col]).fillna(0).astype(float)
            )  # group composition & ratios
        for _col in (
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
            out[_col] = df[_col].fillna('missing').astype('category')
        return out


    def align_categories(train_X, *others):
        """Give every frame identical category levels so the boosters agree."""
        for _col in train_X.select_dtypes('category').columns:
            cats = train_X[_col].cat.categories
            for o in others:
                cats = cats.union(o[_col].cat.categories)
            train_X[_col] = train_X[_col].cat.set_categories(cats)
            for o in others:
                o[_col] = o[_col].cat.set_categories(cats)


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
    display(
        X_all[cat_cols].nunique(dropna=False).sort_values(ascending=False)
    )
    return align_categories, build_features, make_freq_maps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tree/native-categorical path contains 42 columns. A naive one-hot expansion of the same fields would create about 435 columns, mostly from agent and country, so this representation avoids 393 sparse dummy columns. The linear and neural baselines use one-hot encoding with rare levels grouped into `other`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Validation methodology

    Before comparing models, we need a validation setup that resembles the later test window.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.1 Adversarial validation — quantifying the drift

    We train a classifier to tell **test rows from train rows** using the features (label removed, raw date and `Client_ID` dropped). If it separates them well above AUC 0.5, the feature distributions have genuinely drifted.
    """)
    return


@app.cell
def _(
    SEED,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    XGBClassifier,
    load_raw,
    pd,
    roc_auc_score,
    train_test_split,
):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The classifier reaches AUC 0.935, so train and test features are distinguishable even after removing the raw date. The strongest differences include tuition cost, client history, registration lead time, waiting time, and several categorical fields. Together with the changing monthly drop rate, this leads us to evaluate models on a later time window.

    ## 6.2 The chronological holdout

    We use `2017-01-01` as the cutoff because it leaves roughly four months for validation, matching the length and future-facing structure of the hidden test window. All model and feature comparisons fit on the earlier rows and evaluate on this later holdout.
    """)
    return


@app.cell
def _(TARGET, align_categories, build_features, make_freq_maps, pd, train_raw):
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
    return Xtr_n, Xtr_t, Xva_n, Xva_t, tr_raw, y_tr, y_va


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. Model experiments & tuning

    The assignment requires at least three models and hyperparameter tuning. We compare one linear model, one neural network, and one boosted-tree model on the same chronological holdout.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.1 The model families

    We tune one important capacity or regularization parameter for each family and compare the selected settings on the same holdout. The strongest family becomes the focus of the next experiments.

    Each model family is paired with its appropriate preprocessing pipeline: bounded one-hot and scaling for the continuous baselines, and native categorical handling for the tree boosters.

    - **Logistic Regression** provides an interpretable linear reference. Its main tuning parameter here is `C`, the inverse regularization strength.
    - **MLP** can learn nonlinear combinations but requires a complete, scaled numeric matrix. We tune its L2 penalty `alpha` while keeping a small two-layer architecture fixed.
    - **XGBoost** builds trees sequentially so later trees correct earlier errors. It can represent thresholds and interactions directly; we tune tree depth and then the learning-rate/tree-count budget.
    """)
    return


@app.cell
def _(OneHotEncoder, SEED, XGBClassifier, pd):
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


    def encode_for_continuous_models(X_tr: pd.DataFrame, X_va: pd.DataFrame, min_count=30):
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

    return (
        encode_for_continuous_models,
        fit_predict_xgb,
        fit_predict_xgb_pair,
        get_xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.2 Focused hyperparameter tuning

    For each required family, we vary one parameter that controls capacity or regularization and select it using chronological validation ROC-AUC, the project metric. Training AUC is shown beside it so we can see when extra capacity improves fit without improving the future holdout.

    For XGBoost, this first sweep varies `max_depth` while holding the boosting budget fixed. A second experiment then tunes the interaction between learning rate and number of trees.
    """)
    return


@app.cell
def _(
    LogisticRegression,
    MLPClassifier,
    SEED,
    StandardScaler,
    Xtr_t,
    Xva_t,
    display,
    encode_for_continuous_models,
    fit_predict_xgb_pair,
    log_loss,
    np,
    pd,
    plt,
    roc_auc_score,
    y_tr,
    y_va,
):
    Xtr_enc, Xva_enc = encode_for_continuous_models(Xtr_t, Xva_t)
    # We scale continuous inputs using StandardScaler to preserve the variance of naturally skewed columns (like Daily_Tuition_Cost) without compressing normal-range observations.
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr_enc)
    Xva_scaled = scaler.transform(Xva_enc)


    def loss_auc(p_tr, p_va):
        """Train/validation metrics helper."""
        return {
            'train_logloss': log_loss(y_tr, p_tr),
            'val_logloss': log_loss(y_va, p_va),
            'train_AUC': roc_auc_score(y_tr, p_tr),
            'val_AUC': roc_auc_score(y_va, p_va),
        }


    tuning_rows = []
    selected_predictions = {}
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr_scaled, y_tr)
        p_tr = m.predict_proba(Xtr_scaled)[:, 1]
        p_va = m.predict_proba(Xva_scaled)[:, 1]
        tuning_rows.append({
            'family': 'Logistic Regression',
            'axis': 'C  (less regularisation →)',
            'x': C,
            **loss_auc(p_tr, p_va),
        })
        if np.isclose(C, 0.001):
            selected_predictions['lr'] = p_va
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
        p_tr = m.predict_proba(Xtr_scaled)[:, 1]
        p_va = m.predict_proba(Xva_scaled)[:, 1]
        tuning_rows.append({
            'family': 'MLP neural network',
            'axis': '1 / alpha  (less regularisation →)',
            'x': 1.0 / alpha,
            **loss_auc(p_tr, p_va),
        })
        if np.isclose(alpha, 0.1):
            selected_predictions['mlp'] = p_va
    for depth in (2, 3, 4, 5, 6, 8, 10):
        _p_tr, _p_va = fit_predict_xgb_pair(
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
            'family': 'Gradient-boosted trees (XGBoost)',
            'axis': 'max_depth  (more capacity →)',
            'x': depth,
            **loss_auc(_p_tr, _p_va),
        })
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
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (_ax, family) in enumerate(zip(_axes, selected_x)):
        d = tuning[tuning['family'] == family].sort_values('x')
        _ax.plot(
            d['x'],
            d['train_AUC'],
            color='#2b5c8f',
            linestyle='--',
            marker='o',
            alpha=0.7,
            label='train AUC',
        )
        _ax.plot(
            d['x'],
            d['val_AUC'],
            color='#1b9e77',
            linestyle='-',
            marker='s',
            linewidth=2,
            label='val AUC',
        )
        star = d[d['selected']].iloc[0]
        _ax.scatter(
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
        _ax.set_title(family, fontsize=12, pad=12, fontweight='bold')
        # Neural net: L2 penalty alpha; plot 1/alpha so "more capacity" points right.
        _ax.set_xlabel(d['axis'].iloc[0], fontsize=10)
        if i == 0:
            _ax.set_ylabel('ROC-AUC Score', fontsize=10)
        if family != 'Gradient-boosted trees (XGBoost)':
            _ax.set_xscale('log')
        _ax.grid(True, linestyle=':', alpha=0.6)
        loc = 'center right' if family == 'Logistic Regression' else 'lower left'
        if family == 'Gradient-boosted trees (XGBoost)':
            loc = 'lower right'
        _ax.legend(loc=loc, fontsize=8, frameon=True, facecolor='white', framealpha=0.9)
    _fig.suptitle(
        'Focused tuning: training vs validation ROC-AUC',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    selected_tuning = tuning.loc[
        tuning['selected'], ['family', 'x', 'train_AUC', 'val_AUC']
    ].copy()
    selected_tuning[['train_AUC', 'val_AUC']] = selected_tuning[
        ['train_AUC', 'val_AUC']
    ].round(4)
    display(selected_tuning)
    pred_lr = selected_predictions['lr']
    pred_mlp = selected_predictions['mlp']
    return pred_lr, pred_mlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Logistic Regression performs best with strong regularization (`C=0.001`); increasing `C` improves training fit slightly but reduces holdout AUC. The MLP performs best at `alpha=0.1`, after which its training/validation gap grows. XGBoost holdout AUC rises through depth 6 and then levels off while training AUC continues upward, so we keep depth 6 as the best trade-off in this sweep.

    The selected settings reach validation AUC 0.8805 for Logistic Regression, 0.8762 for MLP, and 0.9135 for XGBoost. XGBoost's advantage indicates that thresholds, categories, and interactions matter for this problem, so the remaining experiments focus on boosted trees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.3 Improving the gradient model

    After choosing XGBoost, we first tune its boosting budget. We then test whether adding two fixed-configuration boosted-tree implementations improves the ranking further.

    ### (a) Boosting budget: number of trees × learning rate

    The number of trees and the learning rate interact directly. Consistent with our validation choice, we evaluate the tree budget directly in ROC-AUC. We evaluate various tree counts across two learning rate settings:
    """)
    return


@app.cell
def _(
    Xtr_t,
    Xva_t,
    display,
    fit_predict_xgb_pair,
    log_loss,
    pd,
    plt,
    roc_auc_score,
    y_tr,
    y_va,
):
    budget_rows = []
    budget_predictions = {}
    for lr_rate in (0.1, 0.03):
        for n in (50, 100, 200, 400, 700, 1000):
            _p_tr, _p_va = fit_predict_xgb_pair(
                Xtr_t,
                y_tr,
                Xva_t,
                n_estimators=n,
                learning_rate=lr_rate,
                max_depth=6,
                min_child_weight=5,
                reg_lambda=1.0,
            )
            budget_predictions[(lr_rate, n)] = _p_va
            budget_rows.append({
                'learning_rate': lr_rate,
                'n_trees': n,
                'train_logloss': log_loss(y_tr, _p_tr),
                'val_logloss': log_loss(y_va, _p_va),
                'train_AUC': roc_auc_score(y_tr, _p_tr),
                'val_AUC': roc_auc_score(y_va, _p_va),
            })
    budget = pd.DataFrame(budget_rows)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for _ax, (lr_rate, _sub) in zip(_axes, budget.groupby('learning_rate')):
        _sub = _sub.sort_values('n_trees')
        _ax.plot(
            _sub['n_trees'],
            _sub['train_AUC'],
            marker='o',
            ls='--',
            color='#2b5c8f',
            label='train AUC',
        )
        _ax.plot(
            _sub['n_trees'],
            _sub['val_AUC'],
            marker='o',
            color='#d95f02',
            linewidth=2,
            label='val AUC',
        )
        best_row = _sub.loc[_sub['val_AUC'].idxmax()]
        _ax.scatter(
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
        _ax.set_xlabel('number of trees (n_estimators)', fontsize=10)
        if _ax == _axes[0]:
            _ax.set_ylabel('ROC-AUC Score', fontsize=10)
        _ax.set_title(f'Learning Rate = {lr_rate}', fontsize=12, fontweight='bold')
        _ax.grid(True, linestyle=':', alpha=0.6)
        _ax.legend(fontsize=9, loc='lower right')
    _fig.suptitle(
        'Boosting budget: ROC-AUC vs number of trees, by learning rate (star = best AUC)',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    budget_best = budget.loc[
        budget.groupby('learning_rate')['val_AUC'].idxmax(),
        ['learning_rate', 'n_trees', 'train_AUC', 'val_AUC'],
    ].copy()
    budget_best[['train_AUC', 'val_AUC']] = budget_best[
        ['train_AUC', 'val_AUC']
    ].round(4)
    display(budget_best)
    pred_xgb = budget_predictions[(0.03, 700)]
    return (pred_xgb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At learning rate 0.1, validation AUC peaks around 200 trees and then declines while training AUC keeps rising. At 0.03, improvement is slower but the holdout reaches a slightly higher plateau around 700 trees. We choose `learning_rate=0.03` and `n_estimators=700` for XGBoost.

    ### (b) Does adding other boosters help?

    LightGBM and CatBoost build boosted trees differently from XGBoost, so they may rank some registrations differently. We add them with fixed, capacity-aligned settings and test the ensemble itself: does combining their rankings improve the tuned XGBoost result?

    We use rank averaging because ROC-AUC depends on ordering. Each model's predictions are converted to percentile ranks before averaging, so a model's probability scale cannot dominate the blend. The resulting value is a ranking score, not a calibrated cancellation probability.
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    LGBMClassifier,
    SEED,
    Xtr_t,
    Xva_t,
    display,
    get_xgb,
    np,
    pd,
    pred_xgb,
    rankdata,
    roc_auc_score,
    y_tr,
    y_va,
):
    def get_lgbm(**kw):
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


    def get_cat(**kw):
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


    def fit_predict(name, X_tr, y_train, X_val, sample_weight=None):
        """Fit one boosted-tree implementation and return P(drop) on X_val."""
        if name == "cat":
            cat_idx = [
                i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) == "category"
            ]
            X_tr2, X_val2 = X_tr.copy(), X_val.copy()
            for c in X_tr2.columns[cat_idx]:
                X_tr2[c] = X_tr2[c].astype(str)
                X_val2[c] = X_val2[c].astype(str)
            model = get_cat(cat_features=cat_idx)
            model.fit(X_tr2, y_train, sample_weight=sample_weight)
            return model.predict_proba(X_val2)[:, 1]
        if name == "lgbm":
            model = get_lgbm()
            model.fit(
                X_tr,
                y_train,
                sample_weight=sample_weight,
                categorical_feature=X_tr.select_dtypes("category").columns.tolist(),
            )
            return model.predict_proba(X_val)[:, 1]
        model = get_xgb()
        model.fit(X_tr, y_train, sample_weight=sample_weight)
        return model.predict_proba(X_val)[:, 1]


    def rank_avg(predictions):
        """Average percentile ranks; optimized for ordering, not calibration."""
        return np.mean(
            [rankdata(pred) / len(pred) for pred in predictions], axis=0
        )


    pred_t = {
        "lgbm": fit_predict("lgbm", Xtr_t, y_tr, Xva_t),
        "xgb": pred_xgb,
        "cat": fit_predict("cat", Xtr_t, y_tr, Xva_t),
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
    return blend_t, fit_predict, pred_t, rank_avg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All three boosters perform similarly on this holdout. Adding the fixed LightGBM and CatBoost rankings raises AUC from 0.9135 for tuned XGBoost to 0.9156 for the rank-average blend, so we select the blend as the final ranking model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.4 Returning to the time-index hypothesis

    EDA suggested that the long-term time position might help. Because XGBoost performed best in the family comparison, we use it to test `days_since_epoch` directly. We also compare recency weighting and a random split.

    Trees cannot extrapolate a linear trend beyond the observed range, but the index can still separate older and more recent training regimes. Whether that helps is an empirical question answered by the chronological holdout below.
    """)
    return


@app.cell
def _(
    SEED,
    TARGET,
    Xtr_n,
    Xtr_t,
    Xva_n,
    Xva_t,
    align_categories,
    build_features,
    display,
    fit_predict_xgb,
    make_freq_maps,
    np,
    pd,
    pred_xgb,
    roc_auc_score,
    tr_raw,
    train_raw,
    train_test_split,
    y_tr,
    y_va,
):
    pred_xgb_no_time = fit_predict_xgb(Xtr_n, y_tr, Xva_n)

    # rejected idea: down-weight old rows by a 1-year half-life
    age = (tr_raw["Course_Start_Date"].max() - tr_raw["Course_Start_Date"]).dt.days
    w = np.power(0.5, age / 365.0).values
    pred_recency = fit_predict_xgb(Xtr_t, y_tr, Xva_t, sample_weight=w)

    # Diagnostic only: a random split mixes time periods and is optimistic for
    # the future-window task. Its frequency maps still use training rows only.
    tr_r, va_r = train_test_split(
        train_raw, test_size=0.2, random_state=SEED, stratify=train_raw[TARGET]
    )
    fm_r = make_freq_maps(tr_r)
    Xtr_r = build_features(tr_r, fm_r)
    Xva_r = build_features(va_r, fm_r)
    align_categories(Xtr_r, Xva_r)
    pred_random = fit_predict_xgb(Xtr_r, tr_r[TARGET].values, Xva_r)

    ablation = pd.DataFrame({
        "configuration": [
            "XGBoost, no time index",
            "XGBoost, + time index",
            "XGBoost + recency weights",
            "XGBoost, random split (diagnostic only)",
        ],
        "AUC": [
            roc_auc_score(y_va, pred_xgb_no_time),
            roc_auc_score(y_va, pred_xgb),
            roc_auc_score(y_va, pred_recency),
            roc_auc_score(va_r[TARGET].values, pred_random),
        ],
        "validation": ["chrono", "chrono", "chrono", "random"],
    })
    display(ablation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adding the time index improves XGBoost on the chronological holdout, while recency weighting does not improve on that result. The random split mixes periods and reaches a much higher AUC, showing why it gives an optimistic estimate for the later test window.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Model evaluation

    We use the selected rank-average score for ROC-AUC and precision-recall comparison. The confusion matrix and local interpretation use XGBoost probabilities because these analyses require a threshold on one fitted model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.1 ROC & precision–recall curves
    """)
    return


@app.cell
def _(
    auc,
    average_precision_score,
    blend_t,
    plt,
    precision_recall_curve,
    pred_lr,
    pred_mlp,
    pred_t,
    roc_curve,
    y_va,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for pred, name in [
        (pred_lr, 'Logistic Regression'),
        (pred_mlp, 'MLP'),
        (pred_t['lgbm'], 'LightGBM'),
        (pred_t['xgb'], 'XGBoost'),
        (pred_t['cat'], 'CatBoost'),
        (blend_t, 'Rank-average blend (selected)'),
    ]:
        fpr, tpr, _ = roc_curve(y_va, pred)
        _axes[0].plot(
            fpr,
            tpr,
            label=f'{name} (AUC={auc(fpr, tpr):.3f})',
            lw=2.2 if name.startswith('Blend') else 1.2,
        )
        prec, rec, _ = precision_recall_curve(y_va, pred)
        _axes[1].plot(
            rec,
            prec,
            label=f'{name} (AP={average_precision_score(y_va, pred):.3f})',
            lw=2.2 if name.startswith('Blend') else 1.2,
        )
    _axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    _axes[0].set(
        xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC curve'
    )
    _axes[0].legend(loc='lower right', fontsize=8)
    _axes[1].axhline(y_va.mean(), ls='--', color='grey', alpha=0.7)
    _axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–Recall curve')
    _axes[1].legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The blend has the best holdout AUC in this comparison, so it is the selected ranking score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.2 Confusion matrix & threshold metrics

    A confusion matrix requires a threshold, so we use 0.5 as a simple reference point. Nova Academy could later adjust it according to the relative cost of unnecessary follow-up and missed cancellations.
    """)
    return


@app.cell
def _(
    blend_t,
    classification_report,
    confusion_matrix,
    plt,
    pred_t,
    roc_auc_score,
    sns,
    y_va,
):
    xgb_prob = pred_t['xgb']
    y_hat = (xgb_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_va, y_hat)
    _fig, _ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=',d',
        cmap='Blues',
        cbar=False,
        xticklabels=['pred completed', 'pred dropped'],
        yticklabels=['true completed', 'true dropped'],
        ax=_ax,
    )
    _ax.set_title('Confusion matrix — XGBoost @ 0.5')
    plt.tight_layout()
    plt.show()
    print(
        classification_report(y_va, y_hat, target_names=['completed', 'dropped'], digits=3)
    )
    print(f'AUC of selected rank-average score: {roc_auc_score(y_va, blend_t):.4f}')
    print(f'AUC of representative XGBoost: {roc_auc_score(y_va, xgb_prob):.4f}')
    return (xgb_prob,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the dropped class, recall is the share of actual cancellations that XGBoost flags, while precision is the share of its alerts that actually cancel. False positives consume follow-up resources; false negatives leave cancellations unflagged. Accuracy and F1 summarize the chosen threshold but will change if the threshold moves.

    We submit continuous scores because ROC-AUC evaluates the ordering of registrations across all possible thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.3 Registrations near the illustrative threshold

    We inspect how many XGBoost predictions fall near the 0.5 reference threshold.
    """)
    return


@app.cell
def _(plt, sns, xgb_prob):
    plt.figure(figsize=(9, 4.5))
    sns.histplot(xgb_prob, bins=50, kde=True, color="teal")
    plt.axvline(0.5, color="red", ls="--", label="decision boundary")
    plt.axvspan(0.40, 0.60, color="orange", alpha=0.2, label="near-threshold band")
    plt.xlabel("XGBoost predicted P(drop)")
    plt.title("XGBoost prediction distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    near_threshold = ((xgb_prob > 0.40) & (xgb_prob < 0.60)).mean() * 100
    print(f"share of holdout in the 0.40–0.60 band: {near_threshold:.1f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use one registration from this band for the local SHAP explanation below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Interpretation with SHAP

    We use the tuned XGBoost model for detailed interpretation and then compare its SHAP results with the patterns found during EDA.

    We compute TreeSHAP values on a fixed validation sample of up to 10,000 rows to keep the analysis reproducible and the runtime manageable.
    """)
    return


@app.cell
def _(SEED, Xtr_t, Xva_t, get_xgb, np, roc_auc_score, shap, y_tr, y_va):
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
    return X_shap, explainer, shap_model, shap_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.1 Global importance (beeswarm + bar)
    """)
    return


@app.cell
def _(X_shap, display, np, pd, plt, shap, shap_values):
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
    _ax = top.sort_values('mean_abs_shap').plot.barh(
        x='feature', y='mean_abs_shap', legend=False, figsize=(8, 7), color='#4c72b0'
    )
    _ax.set_xlabel('mean |SHAP value|')
    _ax.set_title('Top 20 features by SHAP importance')
    plt.tight_layout()
    plt.show()
    display(top)
    return (importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The strongest XGBoost contributions broadly match the earlier exploration: `Payment_Terms`, `Origin_Country`, the time index, `Agent_ID`, registration lead time, and support-related features appear near the top. Raw country and agent identity contribute more than their frequency encodings, while the engineered ratios add smaller supporting signals.

    ### Checking the suspicious `Payment_Terms` signal

    EDA showed that prepaid, non-refundable registrations drop unexpectedly often, and SHAP now ranks `Payment_Terms` first. To see how strongly the model relies on it, we refit XGBoost without the field and compare chronological AUC.
    """)
    return


@app.cell
def _(
    Xtr_t,
    Xva_t,
    display,
    fit_predict,
    pd,
    pred_t,
    roc_auc_score,
    y_tr,
    y_va,
):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Removing `Payment_Terms` lowers chronological AUC from 0.9135 to 0.9098, so the field helps but is not carrying the model by itself. Its exact recording time is still worth confirming with the data owner.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.2 Direction of the strongest non-payment signal

    `Origin_Country` is the strongest feature after `Payment_Terms`, so we plot the average SHAP contribution of its most common levels. Positive values push XGBoost toward a higher cancellation score; negative values push it toward completion.
    """)
    return


@app.cell
def _(X_shap, importance, pd, plt, shap, shap_values, sns):
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
        _fig, _ax = plt.subplots(figsize=(8, fig_h))
        sns.barplot(
            data=summary.reset_index(), y='level', x='mean_shap', color='#4c72b0', ax=_ax
        )
        _ax.axvline(0, color='black', lw=1, alpha=0.5)
        _ax.set_title(f'Mean SHAP by {feature} level (top {max_categories} + other)')
        _ax.set_xlabel('mean SHAP contribution')
        _ax.set_ylabel(feature)
        plt.tight_layout()
        plt.show()


    try:
        plot_shap_dependence_readable(top_feat)
    except Exception as e:
        print(f'(dependence view skipped for {top_feat}: {e})')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.3 Explaining one near-threshold registration

    We choose one sampled XGBoost prediction near 0.5 and decompose it. The waterfall shows which features pushed this particular score upward and which pushed it downward.
    """)
    return


@app.cell
def _(X_shap, explainer, np, plt, shap, shap_model, shap_values):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this registration, positive and negative contributions nearly balance, producing a score close to the reference threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. Rebuilding and checking the submission

    The stored `data/Group_27_Submission.csv` is the submission that received the leaderboard score. The block below can retrain the three-model blend and compare the rebuilt ranking with that submission.
    """)
    return


@app.cell
def _(
    Path,
    TARGET,
    align_categories,
    build_features,
    display,
    fit_predict,
    make_freq_maps,
    pd,
    rank_avg,
    test_raw,
    train_raw,
):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The comparison checks the rebuilt file's schema and its Spearman rank agreement with the submitted ranking.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11. Conclusions & Executive Summary

    Nova Academy's test registrations occur after the training period, and both the monthly target rate and the adversarial-validation result (AUC 0.935) show temporal distribution shift. Model selection therefore used a four-month chronological holdout.

    Cleaning reduced hundreds of inconsistent text labels to compact category sets. Missingness, payment terms, country, agent, registration timing, and support activity all carried predictive information. Model comparison confirmed that tuned XGBoost outperformed the Logistic Regression and MLP baselines on the future holdout. Adding fixed LightGBM and CatBoost rankings produced a small further gain, from XGBoost AUC 0.9135 to blend AUC 0.9156 on that split.

    The stored rank-average submission received test ROC-AUC **0.889314**, above the required 0.70.

    Further work could include rolling temporal validation, confirming when `Payment_Terms` is recorded, and calibrating XGBoost probabilities for cost-based operational thresholds.
    """)
    return


if __name__ == "__main__":
    app.run()
