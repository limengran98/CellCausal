# Research Records

This directory is the minimal research-record layer for CellCausal.

It is meant to capture reproducibility and paper-prep metadata early, without
turning the repo into a full ELN or manuscript generator.

## Layout

- `system/`
  Run-level manifests for interactive or scripted runtime executions.
- `tasks/`
  Task-level notes, frozen case definitions, or future task manifests.
- `data/`
  Dataset release targets, accession placeholders, and data-version notes.
- `evals/`
  Eval manifests that summarize case files, suite breakdowns, and runtime metadata.
- `notebooks/`
  Lightweight indexes for notebook runs, reviews, errors, and repairs.
- `ablations/`
  Reserved for future ablation manifests and comparison summaries.
- `paper/`
  Paper-ready templates for data/code availability and reporting-summary material.

## Current Rule

The record layer stores structured manifests and lightweight indexes.
It should not copy large result artifacts by default.

For notebook runs, prefer:

- pointers to `trial_dir`
- `notebook_path`
- `review_report_path`
- `error_log_path`
- `patched_notebook_path`

rather than duplicating large notebooks or result folders.
