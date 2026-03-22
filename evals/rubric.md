# Eval Rubric

This rubric is intentionally lightweight. It is meant for structured human review
of the current system, not for claiming benchmark-grade performance.

## A. Drug-analysis Rubric

### 1. Input parsing correctness

- `0`: Misclassifies the query, crashes, or fails to distinguish drug name vs SMILES.
- `3`: Correctly routes most cases and usually identifies the right input type, but normalization is incomplete or noisy.
- `5`: Correctly routes and normalizes the entity with a clear `input_type`, usable normalized entity record, and no obvious confusion.

### 2. Structure completeness

- `0`: Key sections are missing or the output is mostly unstructured text.
- `3`: Has the major sections but some are shallow, empty, or inconsistently populated.
- `5`: Consistently returns a structured result with summary, mechanism, targets, indications, safety, evidence, and next questions.

### 3. Key knowledge coverage

- `0`: Misses the central mechanism/target/safety facts for the entity.
- `3`: Covers some core facts but leaves major obvious gaps or weakly distinguishes mechanism vs indication vs safety.
- `5`: Covers the most important mechanism, target, indication, and safety themes appropriate for the query.

### 4. Obvious error control

- `0`: Contains clear contradictions, hallucinated sections, or obvious entity confusion.
- `3`: Mostly reasonable but still has noticeable overclaims, mismatched entity names, or weak uncertainty handling.
- `5`: Avoids obvious mistakes, preserves uncertainty where needed, and does not overstate unsupported conclusions.

### 5. Evidence usefulness

- `0`: Evidence is missing, irrelevant, or unusable.
- `3`: Evidence exists but is thin, weakly connected to the claims, or hard to act on.
- `5`: Evidence is relevant, interpretable, and clearly helps explain why the conclusion was produced.

## B. Notebook Workflow Rubric

### 1. Routing correctness

- `0`: Routes to the wrong family, crashes, or misses the notebook intent entirely.
- `3`: Usually routes to notebook workflow but misses some natural-language variants.
- `5`: Reliably maps natural notebook requests to the right workflow entry and subskills.

### 2. requested_actions correctness

- `0`: Action plan is missing or materially wrong.
- `3`: Captures the main action but misses follow-up steps or sequence details.
- `5`: Correctly extracts the intended ordered action sequence, especially for composite queries.

### 3. Artifact/result completeness

- `0`: Output is mostly placeholder or lacks usable structured fields/artifact context.
- `3`: Returns partial structured state, but important artifact or run-result context is missing.
- `5`: Returns clear structured results with action/status, traceable artifacts, and enough context for follow-up steps.

### 4. Diagnostic quality on failure

- `0`: Fails opaquely or crashes.
- `3`: Failure is visible but missing clear cause, context, or next-step hints.
- `5`: Failures are controlled, structured, and make it clear what was attempted, what blocked, and what can happen next.

### 5. Loop usefulness

- `0`: Composite notebook loops do not work or collapse into one generic step.
- `3`: Multi-step loops execute partially, but evidence refresh / review / execute handoff is weak.
- `5`: The workflow meaningfully supports retrieval-refresh -> review -> execute with reusable context and understandable outputs.
