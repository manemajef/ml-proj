# Group 27 - Course-Drop Prediction

I**nteractive HTML Report:** [https://manemajef.github.io/ml-proj/](https://manemajef.github.io/ml-proj/)

TAU Intro to Machine Learning final project for Nova Academy. The task is to
predict the probability that a B2B course registration will be cancelled
(`Dropped_Course`).

- **Deadline:** 17.7.26 at 23:59.
- **Assignment source of truth:** [References/instructions.md](References/instructions.md)`.
- **Best scored submission:** [data/Group_27_Submission.csv](data/Group_27_Submission.csv),
  produced by [pipeline.py](pipeline.py), with documented
  leaderboard AUC `0.889314` (1st of 32).

The current notebook work is in [notebook.ipynb](notebook.ipynb) and
[notebook.py](notebook.py). It is the integrated CRISP-DM draft, but it
should still be reviewed before being renamed for submission.

## Current Status

- The scored CSV and clean production-style pipeline exist.
- The main modeling lesson is stable: the hidden test set is a future time window,
  so model decisions are evaluated with a chronological holdout rather than a
  random split.
- The remaining work is mostly notebook quality: make every important decision
  look intentional, justified, and aligned with the assignment.
- The final report PDF is a separate deliverable and is not maintained in this
  repository yet.

## עברית

- קובץ ההגשה המנוקד כבר קיים: [data/Group_27_Submission.csv](data/Group_27_Submission.csv).
- הלוגיקה שהביאה את הציון נמצאת ב-[pipeline.py](pipeline.py).
- המחברת המרכזית לעבודה היא [notebook.ipynb](notebook.ipynb).
- לקריאה מהירה של המחברת עם פלטים, לפתוח את [docs/notebook.md](docs/notebook.md)`,
- מה שנשאר הוא בעיקר חיזוק המחברת: להסביר החלטות, לוודא שאין ניסוחים חזקים מדי,
  ולסגור התאמה לדרישות ב-[References/instructions.md](References/instructions.md).

### שינויים מהעדכון האחרון (הנחיות לעדכון הדוח)

- **ניתוח (Outliers)**: נוספו גרפי קופסה (Boxplots) המשווים את התפלגות המשתנים `Students_Count`, `Practical_Hours` ו-`Daily_Tuition_Cost` בין ה-Train ל-Test.
  - _הנחיה לדוח:_ יש להציג את התרשימים הללו כנימוק ויזואלי לגבולות הקיטום (Capping) שנבחרו (לעומת הניתוח הקודם שלא כלל תמיכה ויזואלית). כמו כן, לגבי מונים היסטוריים שאינם תואמים (Dropouts > Attended), יש לוודא שהדוח מתייחס אליהם כאל מדד עצימות ביטולים (Intensity) ולא כהסתברות הסתברותית חסומה.
- **תובנות EDA וסיפור עסקי (סעיף 3.7 החדש)**: נוסף סיכום מובנה המקשר את המשתנים ל-4 תימות עסקיות (מחויבות הקונה, אנדוגניות של תנאי התשלום, גיאוגרפיה כמתווך, ואי-סטציונריות).
  - _הנחיה לדוח:_ מומלץ מאוד שהסיפור בדוח יתייחס אל 4 התימות הללו (במקום ניתוח מנותק של כל משתנה בנפרד) כדי להעניק היגיון עסקי חזק למחקר.
- **קללת המימדיות**: הובהר השימוש בקידוד קטגורי מובנה (Native Categorical) וקידוד תדרים (Frequency Encoding) ללא זליגת מידע.
  - _הנחיה לדוח:_ קודם לכן המחברת טענה שקידוד קטגורי מובנה מטפל בקללה עבור כל המודלים, בפועל זה עובד רק עבוד רק עבור מודלי עץ, והמודל של גיגרסיה לוגיסטית ורשת ולמידה עמוקה עדיין עשו one hot encoding ויצרו מאות משתני dummies, עכשיו זה טופל.
- **גרפי כוונון היפר-פרמטרים (Hyperparameter Tuning)**: הגרפים עודכנו ומציגים כעת עקומות Bias-Variance (ביצועי Train מול Validation) עבור רגרסיה לוגיסטית (`C`), רשת עצבית MLP (`alpha`), ו-XGBoost (עומק עץ `max_depth` - נבחר עומק 6).
  - _הנחיה לדוח:_ לוודא שהדוח משתמש בplots החדשים.
- **כוונון תקציב הבוסטינג**: נוסף ניתוח וכוונון של מספר העצים (`n_estimators` מ-50 עד 1000) מול קצב הלמידה (`learning_rate`) עבור מודל העצים.
  - _הנחיה לדוח:_ יש לוודא שהדוח מתייחס לטרייד-אוף שבין קצב למידה נמוך (0.03) למספר עצים גבוה (700) לקבלת הכללה מיטבית (Generalization).
- **שילוב מודלים (Ensemble)**: מודל ההגשה הסופי מבוסס כעת על ממוצע דירוגים (Rank-Average Blend) של LightGBM, XGBoost ו-CatBoost.
  - _הנחיה לדוח:_ יש להקפיד בדוח על שימוש במונח 'ממוצע דירוגים' (Rank-Average) לעומת ממוצע הסתברויות רגיל שהיה קודם, שכן הוא מתעלם מסקאלת הכיול וממקסם ישירות את ה-AUC.
- **שינוי מודל ה-SHAP והערכת רגישות**: כל ניתוח ה-SHAP (חשיבות משתנים גלובלית והסבר תצפית בודדת) הועבר למודל ה-**XGBoost** (במקום LightGBM שהיה מקודם), בהתאם להמלצות ההגשה, והורחב למדגם מייצג של 10,000 שורות. בנוסף, בוצע מבחן רגישות ללא `Payment_Terms`.
  - _הנחיה לדוח:_ יש להחליף את כל תרשימי ה-SHAP (beeswarm ומפל המים של תצפית בודדת) לאלו של XGBoost. כמו כן, יש לשלב בדוח את תובנות ניתוח הרגישות המוכיחות שהמודל יציב ושומר על ביצועים גבוהים גם ללא תנאי התשלום.

## Notebook Summary

Open [docs/notebook.md](docs/notebook.md) for a rendered review path,
but refresh it before treating it as current if the notebook changed.

The notebook currently follows this flow:

1. **Business understanding** - frames cancellations as an operational risk problem
   and explains why the submitted output is a continuous risk score evaluated by AUC.
2. **Data loading and first look** - loads train/test data, documents data types,
   missingness, cardinality, target balance, and obvious corrupted values.
3. **EDA: time structure** - shows that test dates start where training dates end,
   turning the task into future-window forecasting rather than same-period
   interpolation.
4. **EDA: missingness** - compares train/test missingness and shows that missing
   identifier fields carry signal, especially `Company_ID`.
5. **EDA: categorical quality** - demonstrates that many categorical columns are
   dirty variants of a small vocabulary, then normalizes casing, whitespace,
   punctuation, junk placeholders, and country aliases.
6. **EDA: categorical signal** - examines payment terms, client category,
   submission source, enrollment type, country, agent, and company presence.
   The main conclusion is that buyer commitment, acquisition context, geography,
   and agent/company identity are intertwined.
7. **EDA: numeric features** - summarizes numeric distributions, correlations,
   and binned drop-rate profiles. The raw numeric signal is weaker and more
   interaction-driven than the categorical/business-context signal.
8. **Missing values and outliers** - clips physically impossible values, keeps
   plausible heavy tails, preserves predictive missingness, and passes numeric
   `NaN` values to tree models where appropriate.
9. **Feature engineering and dimensionality** - builds seasonality, time trend,
   group-composition, history, cost, lab-match, missingness, and frequency
   features. It explains why native categoricals plus frequency encoding avoid a
   large one-hot expansion.
10. **Validation methodology** - uses adversarial validation to show train/test
    drift and defines a chronological holdout matching the future-window task.
11. **Model experiments and tuning** - compares Logistic Regression, MLP, and
    XGBoost as model families, then improves the tree path with boosting-budget
    tuning and a LightGBM/XGBoost/CatBoost rank-average blend.
12. **Evaluation** - uses the rank-average score for AUC/ROC/PR evaluation and
    the mean boosted-tree probability only for threshold diagnostics such as the
    confusion matrix.
13. **SHAP interpretation** - interprets one representative XGBoost model on a
    fixed validation sample, aligning the explanation with the model family
    introduced in the notebook and recommended by the assignment.
14. **Submission builder** - rebuilds the current scoring logic without overwriting
    the scored CSV by default, then compares rebuilt scores with the stored scored
    submission when available.
15. **Executive summary** - summarizes the process, main findings, selected model,
    score, and future work.

## What Still Needs Review

- Ensure each non-obvious notebook decision has a short reason and, where useful,
  a supporting check.
- Verify the XGBoost SHAP change after the notebook is synced/re-executed.
- Keep the tuning narrative honest: focused sweeps and selected operating region,
  not an exhaustive search.
- Keep the final submission wording conservative: the stored scored CSV is the
  leaderboard record unless an exact-match rebuild is verified.
- Sync or re-export the notebook output before using
  [docs/notebook.md](docs/notebook.md) for final review.

## Files

| Path                                                         | Role                                                                  |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| [References/instructions.md](References/instructions.md)     | Assignment requirements and grading source of truth.                  |
| [notebook.ipynb](notebook.ipynb)                             | Current integrated notebook draft.                                    |
| [notebook.py](notebook.py)                                   | Jupytext source for the current notebook draft.                       |
| [docs/notebook.md](docs/notebook.md)                         | Rendered notebook review artifact; may be stale after notebook edits. |
| [pipeline.py](pipeline.py)                                   | Clean scored pipeline used for the best CSV.                          |
| [data/Group_27_Submission.csv](data/Group_27_Submission.csv) | Best scored submission; do not overwrite casually.                    |
| [\_agent/README.md](_agent/README.md)                        | Rules for local planning and audit notes.                             |

Older notebooks and pipelines are kept for history, but `notebook` and
`pipeline.py` are the active paths.

## Requirements Status

Required final delivery is a single `Group_27.zip` containing:

- `Group_27_Submission.csv` - exists as
  `[data/Group_27_Submission.csv](data/Group_27_Submission.csv)`.
- `Group_27_Notebook.ipynb` - not yet assembled under the final submission name;
  `[notebook.ipynb](notebook.ipynb)` is the current draft.
- `Group_27_Report.pdf` - separate report deliverable.

## How To Run

Requires Python 3.13 and `uv`.

```bash
uv sync
```

Dry-run the scored pipeline without writing a CSV:

```bash
uv run python pipeline.py
```

Write a test CSV without touching the scored submission:

```bash
uv run python pipeline.py --write --out data/tmp_submission.csv
```

Run a cheap syntax check for the current notebook source:

```bash
uv run python -m py_compile notebook.py
```

Run the notebook source as a script only when intentionally validating execution:

```bash
MPLBACKEND=Agg uv run python notebook.py
```

## If You Are An Agent

- Read [\_agent/README.md](_agent/README.md) before creating reusable local notes.
- Read [References/instructions.md](References/instructions.md) before making
  grading, sufficiency, or quality judgments.
- Keep notebook work separate from scored-pipeline work unless explicitly asked.
- Do not overwrite [data/Group_27_Submission.csv](data/Group_27_Submission.csv)
  casually.
- Prefer small, cited decision-defense fixes over broad rewrites.
- Do not call `notebook` final or submission-ready until the review against the
  assignment requirements is complete.
