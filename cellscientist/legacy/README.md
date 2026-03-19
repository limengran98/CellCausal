# Legacy

`legacy/` is reserved for old workflow bridges, compatibility shims, and logic scheduled for deprecation.

Rules for this directory:

- Put migration adapters here when the new runtime must temporarily call into old behavior.
- Keep deprecated logic isolated here instead of spreading it through the new spine.
- Do not add new product features to `legacy/`.
- New runtime work should go into `runtime/`, `registry/`, `skills/`, `tools/`, or `evidence/` instead.
