# Local Agent Workspace

This folder is for persistent local agent outputs that should usually stay out of Git.

Agents do not automatically know these rules just because the folder exists. If a task asks for an audit, plan, prompts, review notes, or a handoff file, read this file first and follow it.

## What Belongs Here

Create a file here for:

- audits
- plans
- prompts for later agent threads
- decisions
- review findings
- handoff notes
- command-output summaries that are evidence for a decision

Do not create a file here for:

- quick one-off answers
- generic explanations
- raw command output unless it is useful evidence
- files that should be part of the actual submitted project

## Naming

Use flat, dated Markdown files:

```text
YYYY-MM-DD-short-topic.md
```

Examples:

```text
2026-07-07-notebook-v3-code-audit.md
2026-07-07-notebook-v3-markdown-audit.md
2026-07-07-readme-rewrite-plan.md
```

Keep names short, lowercase, and hyphen-separated.

## index.md

Keep `_agent/index.md` as the local map of important files. It is ignored by Git.

Each entry should be one short line:

```text
- 2026-07-07-notebook-v3-code-audit.md — analysis/code-quality audit before notebook fixes.
```

Do not turn `index.md` into a full report. It is just a map.

## Shared Prompt Templates

For any task that evaluates notebook quality, assignment coverage, grading risk, analysis sufficiency, or academic honesty, prepend:

```text
_agent/base-grading-agent-prompt.md
```

Do not duplicate that template inside every local plan unless the task needs extra constraints. Keep the grading/evaluation rules canonical there to avoid drift.

## Thread Rules

- If an output will be reused by another agent/thread, write it as a file here.
- If an output is just a direct answer, leave it in chat.
- Prefer one flat folder. Create subfolders only when one topic has more than about 20 files.
- Do not commit files from this folder except `_agent/README.md` and explicitly unignored shared prompt templates.
- If editing project files, do not hide the actual diff inside `_agent/`; summarize it there only when useful.

## Git Rules

The project `.gitignore` should keep this pattern:

```gitignore
_agent/*
!_agent/
!_agent/README.md
!_agent/base-grading-agent-prompt.md
```

That means this README and shared prompt templates can be committed, while local agent outputs stay ignored.
