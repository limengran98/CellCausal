# Notebook Workflow

## Purpose

`notebook-workflow` is the repo-native router for the notebook skill family. It treats
notebook work as an execution surface over shared artifacts and run-state, rather than
as the core ontology of the whole system.

## When To Trigger

Use this skill when the user asks to:

- generate a notebook experiment design
- execute a notebook
- review notebook structure / scientific quality
- refresh evidence before review
- externally autofix a failed notebook run
- run a small multi-step notebook loop such as `review -> execute`

Typical requests:

- `帮我生成一个 notebook 实验设计`
- `执行这个 notebook`
- `review 一下这个 notebook 的结构和科学性`
- `补充生物学证据重新 review，再执行`

## Inputs

- Natural-language query
- Optional recent notebook artifact
- Optional recent notebook run result

## Outputs

Depending on the requested action, current Python runtime may return:

- `generate`
- `retrieval_refresh`
- `review`
- `execute`
- `autofix`
- `multi_step`

Artifacts may include:

- `notebook`
- `evidence_refresh`
- `review_report`
- `notebook_run`

## Current Implementation Status

- Implemented as a Python class skill in `cellscientist/skills/notebook_workflow.py`
- Delegates to repo-native subskills:
  - `notebook-generate`
  - `notebook-retrieval-refresh`
  - `notebook-review`
  - `notebook-execute`
  - `notebook-autofix`
- Reuses legacy bridges where needed for generate / execute / review / repair
- Supports minimal ordered multi-step execution over shared state

## Boundaries

- This is a workflow router, not a generic notebook platform
- It remains an execution-surface skill family, not the architectural center of the system
- Legacy notebook logic is still bridged under the hood; this package does not replace legacy runtime
