# ML Course Project — B2B Course-Drop Prediction (TAU)

Predict the probability that a B2B training-course registration is **cancelled**
(`Dropped_Course`). Scored by **AUC**. The submission is a CSV of
`Client_ID, Drop_Probability` for the official test set.

## Status

| Version          | Approach                                       | Leaderboard AUC          |
| ---------------- | ---------------------------------------------- | ------------------------ |
| v1 (midterm)     | Single XGBoost, date dropped, one-hot          | 0.886408                 |
| **v2 (current)** | LGBM+XGB+CatBoost blend, time-aware validation | **0.889314 — 1st of 32** |

The jump came from one insight: **the test set is the future** (test starts
where train ends and runs 4 months on). v1 was tuned on a random split that
scored 0.944 but didn't reflect that; v2 validates on a chronological future
window instead. Full story in [`notebook_v2.py`](<notebook_v2.py>).

<div dir="rtl">

## תקציר (עברית)

**התובנה המרכזית:** קבוצת הטסט היא ה"עתיד" — נתוני האימון נגמרים באפריל 2017 והטסט מתחיל בדיוק שם וממשיך ארבעה חודשים קדימה. לכן חלוקה אקראית לוולידציה נותנת ציון אופטימי מדי, ואנחנו בודקים מודלים על חלון זמן עתידי במקום.

**שני הקבצים החשובים (נוטבוקים רגילים):**

- [`pipeline_v2.ipynb`](<pipeline_v2.ipynb>) — הצינור שמייצר את התחזיות (המודל הסופי).
- [`notebook_v2.ipynb`](<notebook_v2.ipynb>) — כל ההסבר וההיגיון מאחורי ההחלטות.

**איך מריצים:**:
פותחים את ה-Notebook, מוודאים שקבצי הדאטה נמצאים בתיקייה `data/`, ולוחצים Run All. זהו.
**מה חסר** :

יש לנו כבר קובץ הגשה (CSV) עם AUC של 0.889 — הרבה מעל רף המעבר (0.70). מה שחסר הוא בעיקר ה**תיעוד וההסברים** שהפרויקט נמדד עליהם. לפי החלוקה של ההוראות:

- **[קיים] קובץ הפלט (CSV)** — `Group_27_Submission.csv`.
- **[חלקי] חלק א' — EDA והשלמת חסרים (25%)**: יש EDA טיוטתי ב-`notebook_v1`. חסר: הרבה ויזואליזציות מוסברות, ניתוח סטטיסטי לכל משתנה, קורלציות מול משתנה המטרה, והסבר מנומק של השלמת החסרים כולל גרפים של ההתפלגות לפני/אחרי השלמה.
- **[חלקי] חלק ב' — Feature Engineering ו-Outlier Analysis (20%)**: ההנדסה וה-capping קיימים בקוד. חסר: פלוט שמצדיק כל feature חדש, פלוטים שמצדיקים את ה-caps ל-outliers, הסבר מתמטי (נוסחאות) היכן שרלוונטי, והסבר מפורש איך התמודדנו עם "קללת המימדיות" (השתמשנו ב-frequency encoding ובקטגוריות native במקום one-hot מתפוצץ).
- **[חלקי/חסר] חלק ג' + Model Evaluation (35% = 20% מודלים + 15% הערכה)**:
  - **[חסר] Hyperparameter tuning** — צריך tuning שיטתי (לולאה שמנסה פרמטרים על ולידציה **כרונולוגית** ובוחרת את הטוב ביותר). כרגע הפרמטרים כווננו ידנית.
  - **[חלקי] 3+ מודלים** — יש שלושה (LightGBM/XGBoost/CatBoost). כדאי להוסיף גם Random Forest ו-Logistic Regression ממשפחות שונות (כבר קיימים ב-`pipeline_v1`) עם הסבר קצר על כל מודל וההיפר-פרמטרים שלו. כנראה לא נשתמש בהם בהגשה, אבל זה מחזק את חלק ג'.
  - **[חסר] Confusion Matrix + מדדים** — צריך מטריצת בלבול ומדדים נגזרים (precision / recall / F1 / accuracy) עם הסבר מה כל מדד אומר בהקשר של חיזוי ביטולים.
  - **[חסר] SHAP** — חסר לגמרי. נדרש לבחור מודל אחד ולנתח אותו עם SHAP (חשיבות מסבירים ואיך המודל מחליט).
- **[חסר] חלק ה' — סיכום / תקציר מנהלים (5%)**: חסר לגמרי (עד עמוד אחד).

**קבצי הגשה (מגישים כ-ZIP יחיד בשם `Group_27.zip` ל-Moodle):**

- **[קיים]** `Group_27_Submission.csv`
- **[חלקי]** `Group_27_Notebook.ipynb` — צריך נוטבוק הגשה נקי ומאוחד, עם **מספר הקבוצה + שמות המגישים + ת"ז** בראש, תיעוד מלא, והרצת ה-pipeline על הטסט.
- **[חסר]** `Group_27_Report.pdf` — עד 10 עמודים, גופן David 12, רווח 1.5, שוליים רגילים, עם מספר קבוצה + שמות + ת"ז בראש.

> **מועד הגשה סופי: 17.7.26 בשעה 23:59.** חריגה מפורמט ההגשה = הורדה של 5 נקודות.

</div>

## Repository layout

### TLDR:

**The current last pipeline is written in [pipeline_v2.ipynb](<pipeline_v2.ipynb>)**. The reasoning is written in [notebook_v2.ipynb](<notebook_v2.ipynb>)

> [!NOTE] The code assumes data lives in `data/Train_Data.csv`
>
> - If youre data files are at root, copy them into `./data/*` folder

### Code — the `v{N}` convention

Every version has an **exploration** file and a **pipeline** file. Exploration
files carry the reasoning, plots, and dead ends; pipeline files are the clean,
runnable extract that produces a submission (same logic, no plots, keeps the
explanatory prose). Exploration files are standalone — they do **not** import
the pipelines.

| File                               | Role                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------ |
| [`notebook_v1.py`](<notebook_v1.py>) | v1 exploration — EDA + first modelling pass (draft-quality)              |
| [`pipeline_v1.py`](<pipeline_v1.py>) | v1 pipeline — reproduces the midterm submission (0.886)                  |
| [`notebook_v2.py`](<notebook_v2.py>) | v2 exploration — temporal discovery, chrono validation, model comparison |
| [`pipeline_v2.py`](<pipeline_v2.py>) | v2 pipeline — the current best; writes the official submission           |

Each `.py` is a [jupytext](https://jupytext.readthedocs.io/) _percent_ notebook
paired with a `.ipynb` of the same name. **Edit the `.py`** (cleaner diffs), then
`jupytext --sync <file>.py` to refresh the `.ipynb`. Notebooks are better for
humans; `.py` is better for version control and AI assistance.

### Data — `data/`

- `Train_Data.csv`, `Test_Data_No_Target.csv` — the given data.
- `Group_27_Submission.csv` — **the current official submission (v2). Do not
  modify**; it is the file that was scored.
- `Group_27_Submission_v{N}.csv` — older versioned submissions.
- `samples/`, `bak/` — small samples and backups.

### References — `References/`

- `instructions.pdf` / `instructions.md` — **the assignment** (Markdown copy is
  for AI assistance).
- `LiveCodingSession.*` — lecturer's walkthrough of a similar project.
- `crisp-dm-lec.*` — CRISP-DM / data-prep lecture.

### Exports — `output/`

Readable Markdown renders of notebooks (with plots), produced by
`save_output.py`.

## Getting started

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create .venv and install deps (incl. lightgbm, catboost)

# reproduce a submission (writes to a versioned path by default)
uv run python pipeline_v1.py                    # -> data/Group_27_Submission_v1.csv
uv run python pipeline_v2.py --out data/tmp.csv # v2 blend; omit --out to write the official file

# re-run the exploration / experiments
uv run python notebook_v2.py                    # prints the chrono-holdout tables

# refresh a notebook after editing its .py, or export to Markdown
uv run jupytext --sync notebook_v2.py
uv run python save_output.py notebook_v2.ipynb  # -> output/notebook_v2.md
```

> `pipeline_v2.py` with no `--out` overwrites `data/Group_27_Submission.csv`.
> Pass `--out` while iterating so you don't clobber the scored file.

## Conventions for collaborators

- **Versioning:** start a new `v{N}` for a materially different approach. Copy
  `notebook_v{N-1}.py` → `notebook_v{N}.py`, iterate there, then extract the
  winners into `pipeline_v{N}.py`. Don't rewrite old versions — they're the
  baseline history.
- **Validation:** select models on the **chronological** holdout, not a random
  split (see `notebook_v2.py` for why). Random-split AUC is optimistic here.
- **Reproducibility:** everything is seeded (`SEED = 42`); no notebook writes to
  `data/Group_27_Submission.csv` except a deliberate `pipeline_v2.py` run.
- **Ignored files:** `AGENT.md`, `Notes/`, `archive/` are gitignored scratch —
  not part of the shared project.
