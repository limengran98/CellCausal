# Drug Analysis

## Purpose

`drug-analysis` is a repo-native life-science skill for evidence-driven drug analysis.
It accepts either a drug name or a SMILES string, normalizes the input entity, gathers
available local/BioKB evidence, and returns a structured analysis result without routing
through notebook generation.

## When To Trigger

Use this skill when the user asks for:

- drug mechanism analysis
- target / indication / safety analysis
- SMILES-based drug analysis
- a structured view of what is known, uncertain, and worth checking next for a drug-like entity

Typical requests:

- `分析一下 metformin 的靶点、机制和安全性`
- `根据这个 SMILES 做药物分析: CN(C)C(=N)NC(=N)N`
- `查一下 imatinib 的机制和适应症`

## Inputs

- Natural-language query
- Drug name or alias
- SMILES string

## Outputs

Current Python runtime returns a structured result with at least:

- `task`
- `input_type`
- `normalized_entity`
- `summary`
- `mechanism`
- `targets`
- `indications`
- `safety`
- `evidence`
- `next_questions`

Artifacts:

- `drug_analysis` artifact
- workspace under `results/drug_analysis/<entity>/`

## Current Implementation Status

- Implemented as a Python class skill in `cellscientist/skills/drug_analysis.py`
- Uses repo-native runtime routing and metadata
- Reuses:
  - local drug lookup data
  - SMILES normalization helpers
  - BioKB semantic table generation
  - structured evidence objects
- Does not depend on notebook generation or legacy notebook execution

## Boundaries

- This is a native analysis skill, not a notebook wrapper
- This is heavier than `drug-info`, which remains the lightweight info-card skill
- This skill currently favors structured evidence synthesis over broad multi-source RAG
