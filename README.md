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
window instead. Full story in [`notebook_v2.py`](notebook_v2.py).

## Repository layout

### TLDR:

**The current last pipeline is written in [pipeline_v2.ipynb](pipeline_v2.ipynb)**. The reasoning is written in [notebook_v2.ipynb](notebook_v2.ipynb)

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
| [`notebook_v1.py`](notebook_v1.py) | v1 exploration — EDA + first modelling pass (draft-quality)              |
| [`pipeline_v1.py`](pipeline_v1.py) | v1 pipeline — reproduces the midterm submission (0.886)                  |
| [`notebook_v2.py`](notebook_v2.py) | v2 exploration — temporal discovery, chrono validation, model comparison |
| [`pipeline_v2.py`](pipeline_v2.py) | v2 pipeline — the current best; writes the official submission           |

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
