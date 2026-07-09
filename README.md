# Group 27 - Course-Drop Prediction

TAU Intro to Machine Learning final project for Nova Academy. The task is to
predict the probability that a B2B course registration will be cancelled
(`Dropped_Course`).

- **Deadline:** 17.7.26 at 23:59.
- **Assignment source of truth:** [`References/instructions.md`](References/instructions.md).
- **Best scored submission:** [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv),
produced by [`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py), with documented
leaderboard AUC `0.889314` (1st of 32).

The current notebook work is in [`notebook_v3.ipynb`](notebook_v3.ipynb) and
[`notebook_v3.py`](notebook_v3.py). It is the integrated CRISP-DM draft, but it
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
- קובץ ההגשה המנוקד כבר קיים: [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv).
- הלוגיקה שהביאה את הציון נמצאת ב-[`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py).
- המחברת המרכזית לעבודה היא [`notebook_v3.ipynb`](notebook_v3.ipynb).
- לקריאה מהירה של המחברת עם פלטים, לפתוח את [`output/notebook_v3.md`](output/notebook_v3.md),
- מה שנשאר הוא בעיקר חיזוק המחברת: להסביר החלטות, לוודא שאין ניסוחים חזקים מדי,
  ולסגור התאמה לדרישות ב-[`References/instructions.md`](References/instructions.md).

## Notebook Summary

Open [`output/notebook_v3.md`](output/notebook_v3.md) for a rendered review path,
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
  [`output/notebook_v3.md`](output/notebook_v3.md) for final review.

## Files

| Path | Role |
| --- | --- |
| [`References/instructions.md`](References/instructions.md) | Assignment requirements and grading source of truth. |
| [`notebook_v3.ipynb`](notebook_v3.ipynb) | Current integrated notebook draft. |
| [`notebook_v3.py`](notebook_v3.py) | Jupytext source for the current notebook draft. |
| [`output/notebook_v3.md`](output/notebook_v3.md) | Rendered notebook review artifact; may be stale after notebook edits. |
| [`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py) | Clean scored pipeline used for the best CSV. |
| [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv) | Best scored submission; do not overwrite casually. |
| [`_agent/README.md`](_agent/README.md) | Rules for local planning and audit notes. |

Older notebooks and pipelines are kept for history, but `notebook_v3` and
`pipeline_v2.py` are the active paths.

## Requirements Status

Required final delivery is a single `Group_27.zip` containing:

- `Group_27_Submission.csv` - exists as
  [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv).
- `Group_27_Notebook.ipynb` - not yet assembled under the final submission name;
  [`notebook_v3.ipynb`](notebook_v3.ipynb) is the current draft.
- `Group_27_Report.pdf` - separate report deliverable.

## How To Run

Requires Python 3.13 and `uv`.

```bash
uv sync
```

Dry-run the scored pipeline without writing a CSV:

```bash
uv run python pipelines/pipeline_v2.py
```

Write a test CSV without touching the scored submission:

```bash
uv run python pipelines/pipeline_v2.py --write --out data/tmp_submission.csv
```

Run a cheap syntax check for the current notebook source:

```bash
uv run python -m py_compile notebook_v3.py
```

Run the notebook source as a script only when intentionally validating execution:

```bash
MPLBACKEND=Agg uv run python notebook_v3.py
```

## If You Are An Agent

- Read [`_agent/README.md`](_agent/README.md) before creating reusable local notes.
- Read [`References/instructions.md`](References/instructions.md) before making
  grading, sufficiency, or quality judgments.
- Keep notebook work separate from scored-pipeline work unless explicitly asked.
- Do not overwrite [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv)
  casually.
- Prefer small, cited decision-defense fixes over broad rewrites.
- Do not call `notebook_v3` final or submission-ready until the review against the
  assignment requirements is complete.
