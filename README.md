# Group 27 - Course-Drop Prediction

I**nteractive HTML Report:** [https://manemajef.github.io/ml-proj/](https://manemajef.github.io/ml-proj/)

TAU Intro to Machine Learning final project for Nova Academy. The task is to predict the probability that a B2B course registration will be cancelled (`Dropped_Course`).

- **Deadline:** 17.7.26 at 23:59.
- **Assignment source of truth:** [References/instructions.md](<References/instructions.md>)`.
- **Best scored submission:** [data/Group_27_Submission.csv](<data/Group_27_Submission.csv>),
  produced by [pipeline.py](<pipeline.py>), with documented leaderboard AUC `0.889314` (1st of 32).

The current notebook work is in [notebook.py](<notebook.py>) (`.py` marimo notebook) and a [synced jupyter notebook](<notebook.ipynb>)

> **If `USE_V3=True`:** before exporting `notebook.ipynb`, update the notebook prose to say that ISO week is removed (`41` native / about `434` one-hot columns), XGBoost uses `min_child_weight=10`, `subsample=0.8`, `reg_alpha=0.1`, and `reg_lambda=3.0`, and the chronological blend/without-`Payment_Terms` AUCs are `0.9164`/`0.9093`. The rebuilt filename becomes `data/Group_27_Submission_v3_rebuilt.csv`; keep `0.889314` explicitly labeled as the historical v2 score because v3 has no hidden-test score.

## Road Map

- [ ] **Refine the model's pipeline** - Goal: `AUC > 0.9`, (cant be verified before submission), risks the safe but not impressive `0.8893` current AUC score.
- [ ] Refine ugly plots and graph
- [ ] align [notebook](<notebook.py>) with the [submission-report](<submission-report.md>)

## עברית

- קובץ ההסתברויות שקיבל AUC של `88.889` נמצא כאן: [data/Group_27_Submission.csv](<data/Group_27_Submission.csv>). (הלוגיקה שהביאה את הציון נמצאת ב-[pipeline.py](<pipeline.py>))
- המחברת המרכזית לעבודה היא [notebook.ipynb](<notebook.ipynb>).
- לקריאה מהירה של המחברת עם פלטים, לפתוח את [האתר הזה](https://manemajef.github.io/ml-proj/) או את [docs/notebook.md](<docs/notebook.md>),
- ה-plots יעברו רענון ויהיו ״יפים״ יותר טרם ההגשה.

## Notebook Summary

Open [docs/notebook.md](<docs/notebook.md>) for a rendered review path,
but refresh it before treating it as current if the notebook changed.

The notebook currently follows this flow:

1. **Intro & Business understanding**
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

## Files

| Path                                                         | Role                                                                  |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| [References/instructions.md](<References/instructions.md>)     | Assignment requirements and grading source of truth.                  |
| [notebook.ipynb](<notebook.ipynb>)                             | Current integrated notebook draft.                                    |
| [notebook.py](<notebook.py>)                                   | Jupytext source for the current notebook draft.                       |
| [docs/notebook.md](<docs/notebook.md>)                         | Rendered notebook review artifact; may be stale after notebook edits. |
| [pipeline.py](<pipeline.py>)                                   | Clean scored pipeline used for the best CSV.                          |
| [data/Group_27_Submission.csv](<data/Group_27_Submission.csv>) | Best scored submission; do not overwrite casually.                    |
| [agent/README.md](<_agent/README.md>)                          | Rules for local planning and audit notes.                             |

Older notebooks and pipelines are kept for history, but `notebook` and `pipeline.py` are the active paths.

## How To Run

**Requirements**:

Requires Python 3.13 and `uv`.

**Clone the repo**:

`cd` to desired directory, then run:

```bash
git clone https://github.com/manemajef/ml-proj.git && cd ml-proj
```

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

- Read [agent/README.md](<_agent/README.md>) before creating reusable local notes.
- Read [References/instructions.md](<References/instructions.md>) before making
  grading, sufficiency, or quality judgments.
- Keep notebook work separate from scored-pipeline work unless explicitly asked.
- Do not overwrite [data/Group_27_Submission.csv](<data/Group_27_Submission.csv>)
  casually.
- Prefer small, cited decision-defense fixes over broad rewrites.
- Do not call `notebook` final or submission-ready until the review against the
  assignment requirements is complete.
