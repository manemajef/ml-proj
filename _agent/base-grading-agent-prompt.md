# Base Grading Agent Prompt

Use this at the top of any prompt where an agent evaluates notebook quality, assignment coverage, grading risk, analysis sufficiency, or whether a claim is academically justified.

```text
Before making grading, quality, or assignment-sufficiency judgments, read References/instructions.md.

This is a machine-learning course project that affects the final grade. References/instructions.md is the source of truth for the graders' requirements. Do not infer requirements only from the README, notebook prose, prior agent notes, or file names.

When evaluating the work:
- distinguish assignment requirements from nice-to-have improvements
- distinguish code/execution problems from markdown/disclosure problems
- do not treat runtime shortcuts as bad by default; decide whether they are acceptable if clearly disclosed
- cite concrete file lines for claims
- avoid making the project sound final or submission-ready unless the evidence supports that
```

If the task creates a reusable output file, also include:

```text
Before writing any local audit/plan/prompt files, read _agent/README.md and follow its naming/index rules.
If you create a reusable output file, save it under _agent/ using YYYY-MM-DD-short-topic.md and update _agent/index.md with a one-line entry.
```
