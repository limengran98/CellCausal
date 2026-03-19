# Skill Rearchitecture Notes

Scope: this note is based only on the public README content and root-level repository structure that were readable during review. It is a design-extraction document, not an implementation plan to be copied verbatim.

References reviewed:

- DrugClaw: <https://github.com/QSong-github/DrugClaw>
- awesome-claude-skills: <https://github.com/ComposioHQ/awesome-claude-skills>
- dr-claw: <https://github.com/OpenLAIR/dr-claw>
- superpowers: <https://github.com/obra/superpowers>

Notes on source availability:

- For DrugClaw, the raw README was readable, but the GitHub landing page was not reliably accessible in this environment. Conclusions for DrugClaw therefore rely on README-described structure and usage concepts.
- For the other three projects, the README and root-level GitHub repository page were readable enough to extract high-level architecture signals.

## 1. What to Borrow

### DrugClaw

Most valuable ideas to borrow:

- Drug-native task framing. DrugClaw is organized around concrete drug questions such as target identification, adverse drug reactions, drug-drug interaction, repurposing, labeling, and pharmacogenomics rather than around a generic "assistant" abstraction.
  CellCausal implication: user-facing entrypoints should become domain-shaped skills such as `drug_info`, `drug_target_profile`, `drug_safety_profile`, `pathway_context`, and `cell_state_interpretation`, instead of pushing all biomedical queries into a catch-all agent.
- Resource-native querying. DrugClaw's README emphasizes that each resource has its own semantics and that the system prepares source-aware instructions and examples before querying.
  CellCausal implication: we should keep the separation "skills own task policy, tools own source execution." ChEMBL-like lookups, notebook parsing, local dataset reading, and literature retrieval should not be prematurely flattened into one fake universal interface.
- Evidence-oriented answers. DrugClaw is not just doing retrieval; it is trying to assemble grounded answers across heterogeneous biomedical resources.
  CellCausal implication: the new architecture should require structured evidence objects for knowledge-heavy results, so evidence is a data model, not a block of prose hidden inside final text.
- Optional operating modes. DrugClaw exposes multiple modes such as simpler retrieval-only paths versus more structured reasoning paths.
  CellCausal implication: if we later support `simple`, `evidence_heavy`, or `legacy_bridge` behavior, those should be planner-selected execution modes, not separate runtime stacks.
- Explicit separation of skills, examples, and resource metadata.
  CellCausal implication: our own boundary between `skills/`, `tools/`, `evidence/`, and `legacy/` is the right direction and should be strengthened rather than blurred.

### awesome-claude-skills

Most valuable ideas to borrow:

- Skills as small, focused, reusable units. The repository is organized as a curated catalog of narrow skills with clear purpose.
  CellCausal implication: each skill should solve one stable job well. We should resist giant "biomed_super_skill" blobs that mix routing, retrieval, synthesis, and execution into one prompt surface.
- Strong skill packaging discipline. The ecosystem around Claude Skills assumes a recognizable shape: name, description, when to use, examples, and supporting assets.
  CellCausal implication: repo-native skills should evolve toward a light manifest plus human-readable instructions, even if we do not yet adopt an external marketplace format.
- Taxonomy matters. The project groups skills into categories such as document processing, development tools, data and analysis, collaboration, and systems work.
  CellCausal implication: our skill catalog should be intentionally grouped into domain skills, method skills, and meta skills instead of remaining a flat list of ad hoc modules.
- Portability is useful, but it comes after clarity.
  CellCausal implication: we should write skills in a way that is self-describing enough to port later, while keeping the first-class target as the repository's own runtime.
- Plugin and app action ecosystems show how skills can accumulate operational reach.
  CellCausal implication: the useful lesson is not "integrate SaaS actions now"; it is that skills become maintainable when their dependencies and activation conditions are explicit.

### dr-claw

Most valuable ideas to borrow:

- Planner-generated research brief plus task list. dr-claw explicitly turns an initial user conversation into a structured research brief and a task pipeline.
  CellCausal implication: planner-first is not just a nice idea; it should be the execution backbone. The planner should normalize intent before any heavy action happens.
- Task list as execution truth. The README describes generated tasks as the unit of execution and review.
  CellCausal implication: CellCausal should let the planner define the smallest useful task path for the current request, instead of encoding one universal stage machine in the runtime.
- Inspectable artifacts and progress. dr-claw's research workflow exposes brief, tasks, progress, artifacts, and execution state.
  CellCausal implication: `SessionState` should stay inspectable and artifact-aware. Skill trace, evidence ids, notes, and artifacts are not debugging leftovers; they are core runtime outputs.
- Skills as a subsystem inside a larger product. dr-claw has skills, but it also has server, shared code, docs, and UI.
  CellCausal implication: `skills/` should remain a reusable library that is not coupled to any future UI, notebook frontend, or CLI shell.
- Backend abstraction. dr-claw can switch across multiple agent backends.
  CellCausal implication: compatibility layers belong at the edges. Planner, evidence models, and core skill contracts should not depend on a specific agent runtime.

### superpowers

Most valuable ideas to borrow:

- Planning before acting. superpowers makes planning, scoping, and explicit workflow selection a first-class behavior.
  CellCausal implication: the runtime should continue to treat planning as mandatory, even for lightweight requests. We do not need long plans every time, but we do need planner-normalized intent before execution.
- Skills as operational policy, not just text snippets. superpowers uses skills to define how debugging, testing, planning, and review should happen.
  CellCausal implication: CellCausal should also have meta skills that enforce reviewable behavior such as evidence audit, contradiction checking, output schema validation, and legacy-boundary enforcement.
- Automatic skill triggering. superpowers treats relevant skills as something the agent checks before every task.
  CellCausal implication: once intent is known, registry-based automatic skill resolution should be the default path rather than manual switching.
- Supporting conventions around the skill library. The repository includes docs, tests, commands, and install flows around skills.
  CellCausal implication: a durable skill architecture needs more than Python files. Over time we should standardize minimal manifests, docs, examples, and tests around repo-native skills.
- Explicit philosophy such as "evidence over claims" and "systematic over ad hoc."
  CellCausal implication: this is highly aligned with a scientific runtime. The system should prefer typed evidence and explicit verification over impressive but opaque agent traces.

## 2. What NOT to Copy

The reference projects are useful, but several patterns should not be imported directly into CellCausal V2.

- Do not copy heavy multi-agent orchestration as the default execution truth. Some reference projects lean into broad multi-agent or multi-backend orchestration; CellCausal should start from one planner, one orchestrator, one selected skill, and explicit state transitions.
- Do not introduce a complex graph runtime too early. DrugClaw-style graph reasoning can be valuable later, but it should remain an optional mode or skill, not the default substrate under every request.
- Do not hard-code fixed stages as the bottom-layer truth. A planner may decide on a mini pipeline, but the runtime should not assume that every task must pass through a universal `survey -> design -> execute -> review` chain.
- Do not copy UI-driven architecture. dr-claw's dashboard, chat surface, and Research Lab are product-layer ideas, not immediate runtime requirements for CellCausal.
- Do not import plugin-marketplace or SaaS-action mechanics. awesome-claude-skills includes app automation and plugin concerns that are not aligned with CellCausal's immediate goal of repo-native scientific skills.
- Do not import software-engineering workflow dogma as scientific workflow truth. superpowers is strong on git worktrees, TDD, branch lifecycle, and code review process; those ideas are useful only selectively and should not become CellCausal's universal execution contract.
- Do not flatten skills, tools, retrievers, and adapters into one concept. A skill should not become interchangeable with a database client, notebook executor, or CLI wrapper.
- Do not make external compatibility the first milestone. Cross-platform or marketplace packaging can come later; the first target should be a clear, contributor-friendly, repo-native skill architecture.
- Do not let legacy bridges become the new center of gravity. If we wrap old pipeline behavior, that bridge belongs in `legacy/`, not in the new runtime or core skill contracts.

## 3. Proposed CellCausal V2 Design Principles

1. Planner first. Every execution path begins with normalized intent before retrieval, notebook execution, or synthesis begins.
2. Skill-native architecture. The orchestrator resolves skills; it does not accumulate domain-specific branching logic.
3. Tool and retriever separation. Tools perform low-level execution or source access. Skills define task policy, combination logic, and output shaping.
4. Evidence as structured objects. Evidence should live in typed models with claim, citation, provenance, confidence, and metadata instead of being buried in prose.
5. Artifact-first outputs. Skills should return typed artifacts that can be inspected, persisted, reviewed, or passed downstream.
6. Domain-native interfaces over fake universal wrappers. Different sources may keep different parameters and internal shapes as long as skills normalize the final result contract.
7. Legacy isolation. Old pipeline steps and compatibility bridges stay in `legacy/` and do not define the new runtime's architecture.
8. Repo-native skills first, external compatibility later. The first durable format is "readable, testable, and runnable in this repository."
9. Thin runtime, explicit state. `runtime/` should own session state, planning, routing, and top-level execution trace, but not hide complex behavior in magic background systems.
10. Modes are planner hints, not architecture forks. If we later add `simple`, `evidence_heavy`, or `legacy_bridge` modes, they should be selected inside one runtime model.

## 4. Proposed Skill Taxonomy

The first useful taxonomy for CellCausal should be based on the job to be done, not on implementation detail.

### Domain skills

These are user-facing, biology- or medicine-shaped skills that understand which evidence targets matter for a given question.

Candidate skills:

- `drug_info`
- `drug_target_profile`
- `drug_safety_profile`
- `drug_indication_landscape`
- `compound_mechanism_context`
- `pathway_context`
- `cell_state_interpretation`
- `gene_perturbation_context`
- `disease_context`
- `phenotype_summary`

Why this class matters:

- It matches how users naturally ask questions.
- It keeps CellCausal domain-native instead of agent-native.
- It gives us a clean place to define expected artifact and evidence output for each scientific question family.

### Method skills

These are reusable research-method skills that multiple domain skills can call.

Candidate skills:

- `entity_grounding`
- `literature_retrieval`
- `database_retrieval`
- `evidence_synthesis`
- `evidence_deduplication`
- `contradiction_check`
- `table_normalization`
- `notebook_inspection`
- `result_sanity_check`
- `artifact_packaging`

Why this class matters:

- It avoids duplicating retrieval and synthesis logic across domain skills.
- It creates reusable quality-improving behavior without pushing everything down into low-level tools.
- It provides a natural place to introduce better evidence handling before adding more user-facing skill types.

### Meta skills

These govern orchestration quality, review discipline, and runtime safety.

Candidate skills:

- `intent_triage`
- `skill_router_review`
- `evidence_audit`
- `output_schema_check`
- `legacy_bridge`
- `legacy_scope_guard`
- `session_summarizer`
- `change_review`
- `skill_scaffold_review`
- `codex_task_guard`

Why this class matters:

- It gives reliability behavior a home without bloating the orchestrator.
- It lets us borrow good operational discipline from superpowers without importing its whole software workflow.
- It creates a path to enforce architecture boundaries through skills and checks rather than one-off comments.

## 5. Mapping to Current Refactor Plan

This section maps the reference-project lessons onto the V1 skeleton that already exists in CellCausal.

### `runtime/`

Borrowed signal:

- dr-claw: planner-generated brief and task flow
- superpowers: planning before action
- DrugClaw: planner-selectable operating modes

Mapping to CellCausal:

- `runtime/` should remain the home of `SessionState`, planning, orchestration, and execution trace.
- The planner should stay lightweight but should become the canonical place to classify intent, constraints, and eventually execution mode.
- `runtime/` should not absorb domain knowledge, source-specific retrieval logic, or notebook-compatibility code.
- If we later support multi-step task execution, the task list should be a planner output, not a hard-coded stage engine.

### `registry/`

Borrowed signal:

- awesome-claude-skills: metadata-driven skill discovery
- superpowers: automatic skill activation
- DrugClaw: visible separation across skill families

Mapping to CellCausal:

- `registry/` should remain the stable routing surface from intent to skill.
- The next extension should be richer skill metadata such as tags, supported entities, evidence requirements, and prerequisites.
- Registry logic should decide "which skill is a fit," not "how to execute a source query."
- This keeps routing centralized and avoids `if/elif` branching leaking across unrelated modules.

### `skills/`

Borrowed signal:

- DrugClaw: domain-native skill framing
- awesome-claude-skills: focused reusable skill units
- superpowers: meta skills as policy
- dr-claw: skills as a subsystem within a broader workflow

Mapping to CellCausal:

- `skills/` should grow into domain, method, and meta families rather than one flat folder of unrelated modules.
- Each skill should have a clear responsibility, stable input/output shape, expected artifact type, and evidence behavior.
- Skills should remain the place where low-level tool results are turned into task-shaped outputs.
- `skills/` should not become a hidden dumping ground for planner logic or legacy execution code.

### `tools/`

Borrowed signal:

- DrugClaw: source-aware querying
- awesome-claude-skills: explicit prerequisites and dependencies

Mapping to CellCausal:

- `tools/` should contain low-level wrappers, executors, parsers, and retrievers.
- A tool should answer "how do I talk to this source or executor?" rather than "what task am I solving for the user?"
- Tools may expose heterogeneous parameters internally; skills are responsible for normalizing task output.
- This is where thin wrappers over ChEMBL-like sources, notebook inspection, or local dataset access should live.

### `evidence/`

Borrowed signal:

- DrugClaw: evidence-grounded synthesis
- superpowers: verification over unchecked claims

Mapping to CellCausal:

- `evidence/` should be treated as a first-class package, not a utility afterthought.
- It should define the reusable objects that let domain skills, method skills, and review logic speak the same evidence language.
- If a skill makes knowledge claims, it should either emit structured evidence or explicitly declare why evidence is absent.
- This package is the main protection against collapsing back into plain-text agent answers.

### `legacy/`

Borrowed signal:

- DrugClaw: explicit legacy/script edges
- dr-claw and superpowers: backend-specific integration belongs at the edges

Mapping to CellCausal:

- `legacy/` should contain adapters, compatibility shims, and old-pipeline bridges only.
- It should be the only place allowed to know too much about historical pipeline structure.
- New capabilities should not land here unless the task is explicitly about compatibility.
- The success condition for V2 is not deleting legacy immediately; it is isolating legacy from the new architecture.

## Recommended constraints for future Codex tasks

1. Default to adding new capability in `skills/`, `tools/`, `evidence/`, or `registry/`, not in `run_pipeline.py`.
2. Do not modify `core/agents.py` unless the task is explicitly about legacy compatibility or old-pipeline maintenance.
3. Prefer thin wrappers over existing code instead of transplanting whole legacy modules into the V2 runtime.
4. Any knowledge-heavy output should emit structured evidence objects or explicitly record why evidence is unavailable.
5. Keep tools low-level and side-effect-bounded; keep task policy inside skills.
6. Preserve planner-first routing. Do not smuggle domain logic into the orchestrator as ad hoc branching.
7. Treat fixed phases as optional planner products, not universal runtime truth.
8. Prefer repo-native skill contracts and documentation before considering external plugin or marketplace compatibility.
9. Avoid introducing heavyweight orchestration frameworks, graph runtimes, or new infrastructure dependencies without explicit approval.
10. Add or update focused tests at the skill/tool boundary when behavior changes, even if full end-to-end coverage comes later.

