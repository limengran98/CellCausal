# CellCausal

CellCausal is a repo-native life science agent runtime for structured analysis, experiment planning, and notebook-based validation.

The project has two main layers:

- Native scientific skills for tasks like drug analysis and enzyme mining
- A notebook execution surface for generation, review, and validation when a notebook is the right artifact

The current direction is skill-first. Notebook support remains available, but it is not the center of the system.

## What This Repo Includes

- A lightweight runtime with planner, registry, session state, artifacts, and manifests
- Native skills such as `drug-analysis` and `enzyme-mining`
- A notebook workflow for generation, review, execute, and autofix
- A small eval harness with golden cases
- Research record scaffolding for run and eval metadata
- Repo-local references used to integrate external scientific workflows and model code

## Quick Start

Run an interactive query:

```bash
python run_interactive.py --query "分析一下 metformin 的靶点、机制和安全性"
```

Run enzyme mining:

```bash
python run_interactive.py --query "挖一下和脂代谢相关的候选酶，并生成一个可验证的 notebook 框架"
```

Run the lightweight eval suite:

```bash
python evals/run_eval.py
```

The legacy full pipeline is still available:

```bash
python run_pipeline.py
```

## Project Layout

```text
cellscientist/   Core runtime, skills, tools, legacy bridges, and evidence models
configs/         Runtime and legacy configuration files
docs/            Design notes and integration plans
evals/           Golden cases, rubric, fixtures, and eval runner
prompts/         Prompt templates and reference recipe fragments
records/         Research record templates and generated manifests
references/      Repo-local notebooks, model code, and integration references
skills/          Directory-level skill metadata such as SKILL.md
tests/           Unit and smoke tests
```

## Main Runtime Areas

- `cellscientist/runtime/`
  Planner, orchestrator, manifests, handoff helpers, and state tracking
- `cellscientist/registry/`
  Skill registration and metadata catalog
- `cellscientist/skills/`
  User-facing native skills and notebook workflow surfaces
- `cellscientist/tools/`
  Reusable task-specific helpers, adapters, and bridges
- `cellscientist/legacy/`
  Compatibility wrappers around older notebook and pipeline code

## Current Skill Surface

- `drug-analysis`
  Structured drug-centric analysis for drug names and SMILES
- `enzyme-mining`
  Candidate mining, filtering, ranking readiness, and notebook-ready scaffold output
- `notebook-workflow`
  Notebook generation, review, execute, autofix, and retrieval-augmented loops

## Outputs and Local State

- `results/`
  Local run artifacts such as notebook outputs, drug-analysis outputs, and enzyme-mining exports
- `records/`
  Run manifests, eval manifests, notebook indexes, and paper-oriented record templates

These directories are local working outputs, not the source of truth for the codebase.

## Notes

- Native scientific skills are the mainline path
- Notebook support is an execution and validation surface
- Legacy BBBC036 and older notebook flows are kept as compatibility paths, not as the long-term architecture center
- Eval and record subsystems are included so runs can be inspected and compared without external platforms
