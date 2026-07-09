# Group 27 - Course-Drop Prediction

TAU Intro to Machine Learning final project: predict the probability that a Nova
Academy B2B course registration will be cancelled (`Dropped_Course`).

**Deadline:** 17.7.26 at 23:59.  
**Best scored submission:** [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv), produced by
[`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py), with documented leaderboard AUC `0.889314`
(1st of 32).  
**Current notebook/report direction:** [`notebook_v3.ipynb`](notebook_v3.ipynb) is the integrated
draft. It is the right direction to review next, but it is not final or
submission-ready yet.

Open these first:

- [`output/notebook_v3.md`](output/notebook_v3.md) - rendered draft with outputs; fastest review path.
- [`notebook_v3.ipynb`](notebook_v3.ipynb) - current integrated draft notebook.
- [`References/instructions.md`](References/instructions.md) - assignment requirements and grading source of
  truth.

If [`notebook_v3.ipynb`](notebook_v3.ipynb) changes, refresh [`output/notebook_v3.md`](output/notebook_v3.md) before treating
the Markdown as current.

# סטטוס לרון - רשימת עבודה

המטרה כרגע: להשתמש ב-[`notebook_v3.ipynb`](notebook_v3.ipynb) כבסיס למחברת ולדו"ח, בלי לשבור את ההגשה
המנוקדת שכבר קיימת מ-[`pipeline_v2.py`](pipelines/pipeline_v2.py).

## כבר עשינו / די יציב

- יש CSV מנוקד: [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv).
- יש pipeline שמייצר את המודל המנוקד: [`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py).
- הכיוון המרכזי ברור: הטסט הוא חלון זמן עתידי, ולכן ולידציה כרונולוגית עדיפה
  על פיצול אקראי.
- [`notebook_v3.ipynb`](notebook_v3.ipynb) כבר מרכז את רוב הסיפור במקום אחד: EDA, ניקוי, פיצ'רים,
  מודלים, הערכה, SHAP וסיכום.
- Step 3 במחברת תוקן בכיוון הנכון, אבל זה עדיין לא אומר שכל דרישות העבודה
  סגורות.

## חומר שכבר אפשר להכניס לדו"ח

- סיפור הבעיה העסקית: ביטולי קורסים גורמים להפסד תפעולי וכספי, ולכן צריך
  דירוג סיכון/הסתברות ביטול.
- ההחלטה לעבוד עם AUC ועם score רציף, לא רק החלטת 0/1.
- התובנה שהטסט הוא העתיד: התאריכים ב-test מתחילים איפה שה-train נגמר, ולכן
  random 80/20 split נותן AUC אופטימי מדי.
- גם בלי להסתכל על ה-label, train ו-test לא נראים כמו אותה התפלגות: מודל
  adversarial מצליח לזהות אם שורה הגיעה מ-train או test עם AUC בערך `0.935`.
- יש שינוי חזק באחוזי הביטול לאורך זמן בתוך ה-train. זה רמז טוב לכך שהעסק או
  תמהיל הלקוחות השתנו, ולא רק שהמודל "למד רעש".
- הבחירה ב-validation כרונולוגי כבסיס להשוואת מודלים, במקום לבחור לפי split
  אקראי שמערבב עבר ועתיד.
- `Payment_Terms` הוא הממצא הכי חשוד וחזק ב-EDA: הערך
  `prepaid (non-refundable)` מופיע בכ-10.7k שורות ומתקרב ל-100% ביטולים
  (`0.998`). זה מוזר עסקית, כי דווקא קבוצה ששילמה מראש בלי החזר לא אמורה
  כמעט תמיד לבטל. בדו"ח כדאי להציג את זה כממצא חזק שדורש בדיקת leakage/timing,
  לא כהוכחה סופית.
- רוב גדול יחסית של הדאטה מגיע מפורטוגל: `prt` הוא בערך 42% מה-train, ושם
  אחוז הביטול גבוה יחסית (`0.638`). זה יכול להיכנס כממצא EDA על תמהיל מדינות.
- יש פער גדול בין קבוצות עם `Company_ID` ובלי: רק בערך 5% מהשורות כוללות
  `Company_ID`, ושם אחוז הביטול נמוך בהרבה (`0.212` מול `0.425`). לכן
  `has_company_id` הוא פיצ'ר הגיוני.
- גם `Agent_ID` ו-`Origin_Country` חשובים: הם מופיעים גבוה גם ב-EDA וגם ב-SHAP,
  ולכן לא נכון למחוק אותם רק כי הם מזהים/קטגוריים.
- הנתונים הקטגוריים מלוכלכים מאוד: casing, רווחים, סימנים מוזרים ודוגמאות כמו
  `blu#e`. ניקוי טקסט מצמצם קטגוריות בצורה דרמטית, למשל `Payment_Terms`
  מ-236 ערכים גולמיים ל-3, ו-`Origin_Country` מ-721 ל-153.
- הסיפור של מימדיות: הבעיה היא לא כמות הפיצ'רים הנומריים אלא one-hot על
  קטגוריות. עם native categorical/frequency encoding יש בערך 42 פיצ'רים, לעומת
  כ-435 מימדים ב-one-hot נאיבי.
- פיצ'ר הזמן `days_since_epoch` הוא החלטה חשובה: בעץ, כל שורות test שנמצאות
  אחרי תקופת train יכולות ליפול לאזורים/עלים מאוחרים יותר, וזה נותן למודל דרך
  ללמוד את שינוי המשטר בזמן. זה צריך הסבר פשוט בדו"ח, כי זה לא אינטואיטיבי.
- SHAP מחזק את אותו סיפור: הפיצ'רים הבולטים הם `Payment_Terms`,
  `Origin_Country`, `Agent_ID`, `days_since_epoch`, ותדירויות מזהים. זה מתאים
  למה שראינו ב-EDA ולא נראה כמו importance אקראי.
- הכיוון של המודל הנבחר: blend של LightGBM/XGBoost/CatBoost סביב [`pipeline_v2.py`](pipelines/pipeline_v2.py).
- תוצאת ה-leaderboard של הקובץ הקיים: AUC `0.889314`, מקום 1 מתוך 32.

## כנראה לא נספיק / לא כדאי לפתוח בלי סיבה טובה

- מודל חדש לגמרי או שינוי עמוק ב-[`pipeline_v2.py`](pipelines/pipeline_v2.py).
- החלפת אסטרטגיית validation בשלב מאוחר.
- חיפוש tuning גדול ורחב אם המטרה היא לסיים דו"ח ומחברת בזמן.
- כיול הסתברויות מלא. אפשר להזכיר כעבודה עתידית אם צריך.
- בדיקות production מול data owner, למשל אימות מלא של `Payment_Terms`.

## לא להכניס לדו"ח עדיין כי זה עלול להשתנות

- ניסוח שאומר ש-[`notebook_v3.ipynb`](notebook_v3.ipynb) הוא final או submission-ready.
- טענה שכל דרישה מכוסה במלואה לפני שעוברים מול [`References/instructions.md`](References/instructions.md).
- מסקנות חזקות מדי על SHAP: כרגע זה ניתוח של מודל LightGBM מייצג, לא הסבר
  מלא לכל ה-blend.
- ניסוח שאומר ש-`Payment_Terms` הוא בוודאות leakage. יותר נכון להגיד שזה ממצא
  חשוד מאוד שדורש בדיקה של מתי השדה נקבע ביחס לביטול.
- טענה שה-CSV שנבנה מתוך [`notebook_v3.ipynb`](notebook_v3.ipynb) זהה בוודאות לקובץ המנוקד, אלא אם
  בדקנו את זה בפועל באותו רגע.
- ניסוחים שמציגים shortcuts כמו tuning מצומצם או sample ל-SHAP כאילו היו
  ניסוי מלא.

## לאן אנחנו מתקדמים

- קרובים לסגירה: הסיפור המרכזי, המודל המנוקד, validation כרונולוגי, קובץ CSV.
- צריך עוד עבודה: התאמת המחברת לדרישות אחת-אחת, ליטוש ניסוחים, בדיקת
  הוויזואליזציות והסברים, והפקת דו"ח PDF.
- החלטה שעדיין צריך לקבל: האם [`notebook_v3.ipynb`](notebook_v3.ipynb) מספיק טוב אחרי review, או שצריך
  תיקוני מחברת נקודתיים לפני שממירים אותו לשם ההגשה.

## TODO למחברת

- [ ] לעבור על [`notebook_v3.ipynb`](notebook_v3.ipynb) מול [`References/instructions.md`](References/instructions.md) סעיף-סעיף, ולא לסמן דרישה כסגורה רק כי יש לה כותרת במחברת.
- [ ] לוודא שה-Markdown המסונכרן [`output/notebook_v3.md`](output/notebook_v3.md) באמת משקף את הגרסה האחרונה של המחברת לפני שרון עובד ממנו.
- [ ] לבדוק שכל plot חשוב מלווה בהסבר קצר: מה רואים, למה זה משנה, ואיזו החלטה זה תמך.
- [ ] לחזק את סיפור ה-EDA בדו"ח/מחברת: future test window, adversarial validation, drift לאורך זמן, `Payment_Terms`, פורטוגל, `Company_ID`, agents/countries, וניקוי הקטגוריות.
- [ ] לוודא שחלק missing values מסביר לא רק מה מולא, אלא למה המדיניות הזאת הגיונית ל-train ול-test.
- [ ] לוודא שחלק outliers מצדיק מה clipped ומה נשאר כמו שהוא, עם הסבר עסקי ולא רק טכני.
- [ ] לוודא שחלק feature engineering מסביר את הפיצ'רים החדשים ואת `days_since_epoch` בצורה שרון יוכל להעתיק לדו"ח בלי לנחש.
- [ ] לבדוק שחלק dimensionality אומר במפורש למה native categoricals/frequency encoding עדיפים כאן על one-hot נאיבי.
- [ ] לעבור על model comparison/tuning ולוודא שהניסוח לא מוכר את ה-search כרחב יותר ממה שהוא באמת.
- [ ] לוודא שחלק evaluation כולל confusion matrix/threshold metrics עם משמעות עסקית, ולא רק טבלה מספרית.
- [ ] להשאיר את SHAP כניתוח של מודל LightGBM מייצג, לא כהסבר מלא של כל ה-blend.
- [ ] לבדוק את ניסוח `Payment_Terms`: חזק וחשוד, אבל לא לקרוא לו leakage ודאי בלי הוכחת timing.
- [ ] לבדוק האם בניית ה-CSV מתוך המחברת תואמת את [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv); אם לא, לכתוב שהקובץ המנוקד הוא source of truth.
- [ ] להכין גרסת submission בשם הנדרש רק אחרי review: `Group_27_Notebook.ipynb`.
- [ ] אחרי תיקוני מחברת, לרענן את [`output/notebook_v3.md`](output/notebook_v3.md) כדי שה-review הבא לא יקרא פלט ישן.

## לא לגעת בלי כוונה מפורשת

- לא לדרוס את [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv).
- לא לשנות את [`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py) אם עובדים רק על הדו"ח/מחברת.
- לא להפוך קבצי [`_agent/`](_agent/) לקבצי הגשה. אלה קבצי תכנון מקומיים בלבד.

## Project Story

The first submission already passed comfortably, but it was tuned around a
random split. The later work found the important mismatch: the hidden test set is
later in time than the training set. That made the project a forecasting problem
rather than a same-distribution interpolation problem.

The current scored direction keeps the model that worked best on that framing:
clean the dirty categorical data, preserve high-cardinality identity signal
without one-hot explosion, validate on a future window, and submit a rank-average
blend of boosted tree models. [`notebook_v3.ipynb`](notebook_v3.ipynb) is the attempt to turn that modelling
work into a coherent CRISP-DM-style notebook and report story.

Still-open decisions are mostly documentation and sufficiency decisions, not a
need to restart modelling: how much tuning evidence is enough, how strongly to
word SHAP and leakage claims, whether the evaluation section is clear enough,
and what should be included in the final PDF versus left as future work.

## Files

| Path                                                           | Role                                                                                        |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [`output/notebook_v3.md`](output/notebook_v3.md)               | Rendered version of the current integrated draft; open this first for review.               |
| [`notebook_v3.ipynb`](notebook_v3.ipynb)                       | Current integrated draft notebook; review before any submission decision.                   |
| [`notebook_v3.py`](notebook_v3.py)                             | Text source for the draft notebook; do not edit unless intentionally updating the notebook. |
| [`pipelines/pipeline_v2.py`](pipelines/pipeline_v2.py)         | Current scored pipeline/model logic.                                                        |
| [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv) | Current scored CSV submission; do not overwrite casually.                                   |
| [`References/instructions.md`](References/instructions.md)     | Markdown copy of the assignment requirements.                                               |
| [`_agent/`](_agent/)                                           | Local planning/audit workspace only; not a submission folder.                               |

Older notebooks and pipelines ([`notebook_v1.py`](notebook_v1.py), [`notebook_v2.py`](notebook_v2.py), [`pipeline_v1.py`](pipelines/pipeline_v1.py),
[`Project_Ron_V3.ipynb`](Project_Ron_V3.ipynb)) are useful for history and comparison, but they are not
the current landing path.

## Requirements Status

Required final delivery is a single `Group_27.zip` containing:

- `Group_27_Submission.csv` - exists as [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv) and is
  the currently scored submission.
- `Group_27_Notebook.ipynb` - not yet assembled under the final submission name;
  [`notebook_v3.ipynb`](notebook_v3.ipynb) is the current draft to review and adapt.
- `Group_27_Report.pdf` - still needs to be written/exported.

Do not claim the notebook covers every assignment requirement until the
notebook/report review confirms it directly. In particular, review the EDA,
missing-value handling, feature engineering, outlier analysis, dimensionality,
model comparison, tuning, threshold metrics, SHAP interpretation, and executive
summary against [`References/instructions.md`](References/instructions.md).

## How To Run

Requires Python 3.13 and `uv`.

```bash
uv sync
```

Dry-run the currently scored pipeline without writing a CSV:

```bash
uv run python pipelines/pipeline_v2.py
```

Write a test CSV without touching the scored submission:

```bash
uv run python pipelines/pipeline_v2.py --write --out data/tmp_submission.csv
```

Only overwrite [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv) deliberately:

```bash
uv run python pipelines/pipeline_v2.py --write
```

Run the current integrated draft notebook as a script only when you intentionally
want to validate or refresh its execution:

```bash
uv run python notebook_v3.py
```

## If You Are An Agent

- Read [`_agent/README.md`](_agent/README.md) before creating or editing local planning/audit files.
- Treat [`References/instructions.md`](References/instructions.md) as the grading source of truth.
- Keep [`_agent/`](_agent/) outputs out of the submission package.
- Do not edit [`notebook_v3.py`](notebook_v3.py) or [`notebook_v3.ipynb`](notebook_v3.ipynb) for README-only tasks.
- Keep [`pipeline_v2.py`](pipelines/pipeline_v2.py) / [`data/Group_27_Submission.csv`](data/Group_27_Submission.csv) separate from
  [`notebook_v3.ipynb`](notebook_v3.ipynb): v2 is the currently scored submission, while v3 is the current
  integrated draft.
- Prefer honest status language. Do not call [`notebook_v3.ipynb`](notebook_v3.ipynb) final,
  submission-ready, or fully requirement-complete unless the notebook/report
  work proves that.
