# Skill Directory Transition

This note explains how the current Python class-based skill runtime relates to the next
directory-level skill shape.

## 1. Current Relationship

Right now, the runtime source of truth is still the Python skill class:

- routing lives in `runtime/` and `registry/`
- execution lives in `cellscientist/skills/*.py`
- state, artifacts, and evidence live in the runtime spine

The new top-level `skills/<skill-name>/SKILL.md` directories are not loaders yet. They
are compatibility scaffolds that make each skill easier to describe, package, discover,
and later evolve into a richer skill bundle.

In other words:

- Python class skill = executable runtime unit
- directory-level skill package = descriptive/package layer attached to that unit

## 2. Why Runtime Stays Repo-Native For Now

The repo-native runtime should stay primary in the near term because it already provides:

- planner-first routing
- structured state and artifact flow
- notebook-family bridging where legacy is still required
- native skill growth for paths like `drug-analysis`

Replacing that with a full `SKILL.md` loader now would create more surface area than value.
The current goal is compatibility, not runtime replacement.

This follows the useful parts of the reference projects without copying their full stack:

- DrugClaw:
  skill-centric organization and task-native packaging
- awesome-claude-skills:
  folder-level skill metadata and reusable documentation
- Dr. Claw:
  discoverable project skill library concepts
- Superpowers:
  planner-first execution discipline and reusable workflow thinking

## 3. What The New Directory Layer Adds Today

The new layer currently adds three small things:

1. A stable package path for selected skills:
   - `skills/drug-analysis/`
   - `skills/notebook-workflow/`

2. A human-readable contract in `SKILL.md`:
   - what the skill is for
   - when to trigger it
   - what it takes in
   - what it returns
   - what state it is actually in today

3. A minimal metadata bridge in the Python runtime:
   - `BaseSkill.skill_package_metadata()`
   - `BaseSkill.skill_metadata()["skill_package"]`
   - `SkillRegistry.skill_catalog()`

This means the registry can already expose:

- skill name
- runtime metadata
- whether a directory package exists
- where its `SKILL.md` lives

without making execution depend on those files.

## 4. What This Is Not Yet

This is not yet:

- a `SKILL.md` runtime loader
- a template/script auto-discovery system
- an external skill import mechanism
- a replacement for Python class execution

That is intentional. The runtime remains class-based; the directory layer is just the
first compatibility seam.

## 5. Gradual Transition Path

The safe transition path is:

1. Keep Python class skills as the executable core
2. Give important skills directory packages with `SKILL.md`
3. Let registry/catalog surfaces expose package paths and docs
4. Add optional package-local resources later:
   - `templates/`
   - `scripts/`
   - `resources/`
   - `examples/`
5. Only after several native skills exist should the repo consider a real package loader

## 6. Immediate Architectural Rule

For the next stage, the rule should be:

- new important skills should have both:
  - a Python class implementation
  - a small directory package with `SKILL.md`

But:

- runtime execution should still resolve through the Python registry
- `SKILL.md` should document and package skills, not become the execution truth yet

## 7. Main Conclusion

CellCausal should move toward directory-level skill packages incrementally, not by
replacing the current repo-native runtime in one jump.

For now:

- repo-native Python runtime remains primary
- directory-level skills are the compatibility layer
- this is enough to start resembling DrugClaw / awesome-claude-skills style skill shapes
  without importing their full runtime model
