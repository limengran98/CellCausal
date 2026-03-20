# Branch A Exit Plan

Scope: this note is a stop-condition document for Branch A. It is not a new refactor roadmap. Its job is to mark what is now stable enough, what remains intentionally legacy-bound, and where the next mainline work should go.

Design stance used in this document:

- Borrow from DrugClaw: domain-native skills and source-native retrieval, not a generic biomedical mega-agent.
- Borrow from awesome-claude-skills: lightweight skill metadata and skill-directory discipline, but keep the runtime repo-native for now.
- Borrow from dr-claw: artifacts, workspaces, and inspectable project state, but do not treat fixed phase flows as the permanent execution truth.
- Borrow from Superpowers: planner-first and action decomposition before execution, but do not expand Branch A into a general-purpose meta-agent program.

## A. Current State Snapshot

### What the new runtime already has

- `runtime/` now provides a lightweight but real execution spine:
  - planner-based intent parsing
  - session state
  - artifact and notebook/run-state tracking
  - structured result packaging
- `registry/` now resolves repo-native skills and can return structured clarification suggestions instead of crashing on unknown requests.
- `skills/` now has the first real skill family split:
  - `drug-info`
  - `notebook-workflow`
  - notebook subskills for `generate`, `execute`, `review`, and `autofix`
- planner output is no longer just `task_type`; it also carries `requested_actions` and secondary hints, which is enough for minimal multi-step orchestration.
- evidence already exists as a typed concept for the drug path, which is the first sign that new native skills can grow outside notebook-centric execution.

### What the notebook family currently is

- The notebook family is now a router plus thin wrappers, not a single placeholder skill.
- `notebook-workflow` can:
  - route single actions
  - execute minimal ordered multi-step action lists
  - reuse recent notebook artifacts and run results
- `notebook-generate` has:
  - legacy bridge wiring
  - provider failover diagnostics
  - degraded notebook fallback
  - structured notebook artifact output
- `notebook-execute` has:
  - legacy execution bridge wiring
  - structured run-result output
- `notebook-review` and `notebook-autofix` now exist as post-processing surfaces:
  - review consumes notebook/run context and can bridge into legacy analysis/report generation
  - autofix is explicitly an external repair wrapper, not a duplicate of execute

### What still strongly depends on legacy

- Notebook generation still depends on the legacy `phase_generate` stack in [prompt_orchestrator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_orchestrator.py).
- Notebook execution still depends on the legacy `phase_execute` and `run_notebook_with_autofix` path.
- Notebook review still depends on legacy `phase_analyze`, `review_workflow.py`, and `experiment_report.py`.
- Prompt-driven notebook generation still depends on the monolithic contracts in:
  - [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml)
  - [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml)
- Legacy config inheritance and stage setup still drive most notebook behavior through:
  - [pipeline_config.json](/home/lmr/CellCausal-main/configs/pipeline_config.json)
  - [experiment_config.json](/home/lmr/CellCausal-main/configs/experiment_config.json)
  - [review_config.json](/home/lmr/CellCausal-main/configs/review_config.json)

### What is already good enough for native skill growth

- planner and registry are now separate from the legacy phase machine
- notebook is no longer the only entrypoint concept
- artifacts, evidence, and run-state now exist as reusable runtime objects
- unknown requests no longer crash the interface
- a new skill can now be added without editing legacy core files, as long as it fits the repo-native `skills/ + tools/ + runtime/` contract

Bottom line:

- Branch A has successfully turned notebook execution into a bridged skill family.
- Branch A has not made the old experiment stack generic, and it should stop trying to.

## B. Hardcoded Coupling Inventory

### 1. Data-format coupling

- High: [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml) hard-codes a `combined` HDF5 group and mandatory datasets such as `morphology_pre`, `morphology_post`, `smiles`, `dose`, and `split_id`.
- High: [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml) assumes five-fold `split_id` cross-validation and explicitly forbids random splitting.
- High: [pipeline_config.json](/home/lmr/CellCausal-main/configs/pipeline_config.json) defaults to `dataset_name = BBBC036` and `split_name = smiles`, which makes current dataset discovery BBBC036-first by default.
- High: [execution_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/execution_workflow.py) and pipeline utils resolve a single “Stage 1 H5” path and expose it as `STAGE1_H5_PATH`, which is a specific adapter seam masquerading as general runtime.
- Medium: `bio_kb` defaults in [pipeline_config.json](/home/lmr/CellCausal-main/configs/pipeline_config.json) point at a BBBC036 SMILES H5 path and assume chemical perturbation access is the default data context.
- Medium: SMILES resolvers and H5 column logic in `cellscientist/core/smiles_resolver.py` and `cellscientist/core/bio_kb/smiles_resolver.py` are useful, but still tuned around a perturbation H5 worldview.
- Medium: generated notebook code and review prompts assume morphology and chemistry are always joint modalities.

### 2. Task coupling

- High: [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml) defines the task as cellular drug perturbation / Cell Painting modeling, not generic notebook generation.
- High: [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml) assumes the objective is iterative model optimization under a benchmark metric regime, not general notebook review.
- High: [experiment_config.json](/home/lmr/CellCausal-main/configs/experiment_config.json) and [review_config.json](/home/lmr/CellCausal-main/configs/review_config.json) hard-code a benchmark-style loop around `PCC`, thresholds, iterations, and protected cells.
- High: the review path assumes “improve the model notebook” rather than “inspect or explain a scientific workflow.”
- Medium: the notebook family still inherits an implied “generate -> execute -> analyze/review” worldview from legacy phase naming and filesystem layout.
- Medium: the current review optimization hierarchy is fixed to architecture / fusion / loss, which is specific to one modeling recipe, not a generic scientific runtime.
- Low: current planner notebook heuristics still infer notebook family from experiment-design and execution phrases that are largely shaped by the existing perturbation modeling workflow.

### 3. Prompt coupling

- High: [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml) is a monolithic recipe prompt that mixes runtime contract, data schema, modeling protocol, evaluation rules, and output interface in one file.
- High: [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml) locks the model interface to `(features, dose, smiles_emb, batch_indices)` and expects notebook cell-level mutation under that interface.
- High: `prompt_generator.generate_notebook_content()` in [prompt_generator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_generator.py) expects notebook JSON with `cells` and `hypergraph`, which is a prompt-specific output protocol, not a general runtime contract.
- Medium: prompt content assumes innovation-first modeling, baselines, DEG metrics, and biology-aware architectural novelty as one inseparable template package.
- Medium: external knowledge injection in `prompt_generator.py` and `review_workflow.py` is currently bound to design/review prompt assembly rather than abstracted as reusable evidence synthesis fragments.
- Medium: prompt files encode output artifact names like `metrics.json`, `analysis_summary.json`, and `experiment_report.md`, which should be recipe-level conventions, not runtime truths.

### 4. Runtime coupling

- High: notebook bridges in `cellscientist/legacy/` still depend directly on legacy entries in:
  - [prompt_orchestrator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_orchestrator.py)
  - [execution_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/execution_workflow.py)
  - [review_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/review_workflow.py)
- High: notebook remains the implied default artifact for “serious scientific work” in the old stack, even though the new runtime no longer requires that assumption.
- Medium: result directories like `generate_execution/`, `review_feedback/`, `prompt_run_*`, and `workspace_*` are still operational truths inside legacy discovery logic.
- Medium: legacy execution and review wrappers still use the old phase naming model (`generate`, `execute`, `analyze`, `review_feedback`) as if that phase graph were the universal system ontology.
- Medium: the runtime currently has a clean notebook family, but not yet a fully separate notion of recipe package versus execution surface.
- Low: notebook-family suggestions still dominate current clarification behavior because the legacy surface remains the most mature execution path.

## C. Generic Runtime vs Recipe Boundary

This is the key Branch A conclusion.

### Should belong to generic runtime

- planner and intent parsing:
  - [planner.py](/home/lmr/CellCausal-main/cellscientist/runtime/planner.py)
- session state, artifacts, notebook run-state, evidence ids:
  - [state.py](/home/lmr/CellCausal-main/cellscientist/runtime/state.py)
  - [notebook_models.py](/home/lmr/CellCausal-main/cellscientist/runtime/notebook_models.py)
- skill resolution and skill metadata:
  - [skill_registry.py](/home/lmr/CellCausal-main/cellscientist/registry/skill_registry.py)
  - [base.py](/home/lmr/CellCausal-main/cellscientist/skills/base.py)
- repo-native skill surfaces:
  - [drug_info.py](/home/lmr/CellCausal-main/cellscientist/skills/drug_info.py)
  - [notebook_workflow.py](/home/lmr/CellCausal-main/cellscientist/skills/notebook_workflow.py)
- structured result packaging and clarification fallback:
  - [orchestrator_v2.py](/home/lmr/CellCausal-main/cellscientist/runtime/orchestrator_v2.py)
  - [run_interactive.py](/home/lmr/CellCausal-main/run_interactive.py)
- thin notebook bridge wrappers as an execution surface:
  - [notebook_bridge.py](/home/lmr/CellCausal-main/cellscientist/legacy/notebook_bridge.py)
  - [notebook_review_bridge.py](/home/lmr/CellCausal-main/cellscientist/legacy/notebook_review_bridge.py)
  - [notebook_repair_bridge.py](/home/lmr/CellCausal-main/cellscientist/legacy/notebook_repair_bridge.py)
- generic evidence and tool patterns:
  - `cellscientist/evidence/`
  - `cellscientist/tools/`

### Should belong to recipe / adapter / template

- dataset-specific H5 assumptions and field expectations:
  - BBBC036
  - `combined`
  - `morphology_pre`
  - `morphology_post`
  - `smiles`
  - `dose`
  - `split_id`
- perturbation-modeling task defaults:
  - Cell Painting perturbation prediction
  - benchmark loops
  - DEG metrics
  - PCC-first winner selection
- task-specific experiment presets:
  - [experiment_config.json](/home/lmr/CellCausal-main/configs/experiment_config.json)
  - [review_config.json](/home/lmr/CellCausal-main/configs/review_config.json)
- domain prompt fragments and notebook generation contracts:
  - [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml)
  - [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml)
  - [autofix.yml](/home/lmr/CellCausal-main/prompts/autofix.yml)
  - [experiment_report.yaml](/home/lmr/CellCausal-main/prompts/experiment_report.yaml)
- data adapters:
  - H5 path resolution
  - SMILES extraction
  - Cell Painting feature reading
  - any future non-H5 loaders
- named recipe packages:
  - “BBBC036 Cell Painting perturbation modeling”
  - future “drug-analysis”
  - future “enzyme-mining”

### Design boundary rules

1. `runtime/` should decide intent, action order, state flow, and fallback behavior. It should not decide BBBC036 schema or perturbation metrics.
2. `registry/` should know skill metadata and suggestions. It should not know H5 columns, fold logic, or model signatures.
3. `skills/` should express user-facing jobs and compose tools/artifacts. They should not absorb full benchmark recipes.
4. notebook is an execution surface, not the core ontology of the system.
5. H5 parsing, SMILES extraction, and BBBC036 feature assumptions are adapters, not runtime.
6. DEG metrics, PCC thresholds, architecture/fusion/loss optimization hierarchy, and Cell Painting modeling assumptions are recipe concerns, not generic notebook logic.
7. prompt files that encode modeling contracts or dataset structure are recipe/template assets, not generic skill behavior.
8. legacy phase code can remain callable through bridges, but those bridges must not define future-native skill architecture.
9. a new native skill should be addable without importing BBBC036 prompt contracts or perturbation notebook generation.

## D. What Must Be Frozen in Legacy

These parts are not worth pulling further into the generic layer during Branch A.

- The old phase machine in [prompt_orchestrator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_orchestrator.py):
  - `phase_generate`
  - `phase_execute`
  - `phase_analyze`
  - `run_full_pipeline`
- The legacy stage setup logic in [execution_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/execution_workflow.py), especially its Stage-1 H5 / idea file setup assumptions.
- The iterative optimization engine in [review_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/review_workflow.py), including:
  - optimization history
  - task graph updates
  - protected cell policy
  - iteration loops
- The notebook JSON + hypergraph prompt protocol in [prompt_generator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_generator.py).
- The benchmark-style review configuration in [review_config.json](/home/lmr/CellCausal-main/configs/review_config.json).
- The monolithic perturbation-modeling prompt contracts in:
  - [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml)
  - [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml)

What “freeze in legacy” means here:

- keep bridging them
- keep using them where they already work
- stop promoting them as generic architectural centerpieces
- do not spend more Branch A time abstracting them unless a later native skill is directly blocked

## E. Minimal Decoupling Plan

Only three small actions are justified after this document.

### 1. Add an explicit data-adapter seam

- Purpose:
  - separate generic runtime from BBBC036/H5 assumptions
- Modules involved:
  - [execution_workflow.py](/home/lmr/CellCausal-main/cellscientist/core/execution_workflow.py)
  - `cellscientist/core/smiles_resolver.py`
  - `cellscientist/pipeline/utils.py`
  - future adapter package or adapter manifest
- Expected gain:
  - native skills stop inheriting “Stage 1 H5” as a hidden universal dependency
- Why only this small:
  - we do not need a new data framework yet; we only need one explicit seam where BBBC036-specific loading can live without contaminating future skills

### 2. Add an explicit recipe seam for perturbation modeling

- Purpose:
  - make “BBBC036 Cell Painting perturbation modeling” a named recipe rather than the implicit default behavior of notebook generation
- Modules involved:
  - [pipeline_config.json](/home/lmr/CellCausal-main/configs/pipeline_config.json)
  - [experiment_config.json](/home/lmr/CellCausal-main/configs/experiment_config.json)
  - [review_config.json](/home/lmr/CellCausal-main/configs/review_config.json)
  - notebook bridges that load these configs
- Expected gain:
  - future skills can reuse runtime surfaces without inheriting one benchmark’s optimization worldview
- Why only this small:
  - we do not need a full recipe engine yet; we only need a clean label and boundary so the current recipe stops pretending to be the system default

### 3. Add a prompt/template seam

- Purpose:
  - split monolithic prompt contracts into “execution-surface template” versus “domain recipe fragment”
- Modules involved:
  - [pipeline_prompt.yaml](/home/lmr/CellCausal-main/prompts/pipeline_prompt.yaml)
  - [review_optimize.yaml](/home/lmr/CellCausal-main/prompts/review_optimize.yaml)
  - [prompt_generator.py](/home/lmr/CellCausal-main/cellscientist/core/prompt_generator.py)
- Expected gain:
  - notebook generation can remain bridged without forcing every future skill to inherit Cell Painting modeling assumptions
- Why only this small:
  - the goal is not prompt redesign; it is only to expose a seam between reusable notebook-shell concerns and recipe-specific modeling instructions

## F. Exit Criteria for Branch A

Branch A should stop here when all of the following are true.

1. notebook has been demoted to an execution surface.
   - notebook generation/execution/review/autofix exist as a skill family
   - notebook is no longer the only system entrypoint
   - notebook is not treated as the architecture’s core abstraction

2. runtime, recipe, and adapter boundaries are clear enough to guide new work.
   - planner, registry, state, artifact, and evidence live in the new runtime spine
   - BBBC036/H5/Cell Painting specifics are recognized as recipe or adapter concerns
   - old phase logic is explicitly legacy-bound

3. new native skills no longer need to be justified through the old notebook prompt chain.
   - a new skill can be added under `skills/` with repo-native routing
   - it does not need to inherit `pipeline_prompt.yaml`
   - it does not need to inherit BBBC036 assumptions

4. the notebook family is “good enough” rather than “fully generalized.”
   - it can route, bridge, and degrade safely
   - it no longer crashes on natural notebook queries or composite follow-ups
   - further notebook-centric abstraction work is no longer the highest-value path

5. mainline focus should immediately switch to native life-science skills.
   - `drug-analysis`
   - `enzyme-mining`

This is the stop condition:

- If the next proposed work item is still “make notebook/legacy prompt execution more generic” rather than “ship a native domain skill,” Branch A has already run too long.

## G. Immediate Next Step After Branch A

### 1. `drug-analysis` MVP

- Why first:
  - it directly validates the new runtime on a domain-native skill that does not need notebook generation as its core surface
  - it exercises planner, skill metadata, tools, evidence, and synthesis in the direction Branch A was supposed to unlock
  - it proves the system can answer life-science questions without routing everything through the perturbation notebook stack

### 2. `enzyme-mining` MVP

- Why second:
  - it is a strong test that the runtime can support a very different scientific task shape
  - it naturally pressures the future data-adapter and recipe seams without requiring notebook-first design
  - it forces source-native retrieval and evidence handling to mature beyond the current perturbation benchmark worldview

### 3. Evidence/report synthesis skill

- Why third:
  - both `drug-analysis` and `enzyme-mining` will need a consistent way to turn evidence objects into structured outputs and reports
  - this is more reusable than continuing to optimize notebook prompts
  - it strengthens the shared scientific runtime rather than deepening notebook-specific abstractions

## Final Branch A Conclusion

Branch A should be considered complete after this document.

What Branch A achieved:

- a functioning new runtime skeleton
- a real notebook skill family
- structured artifact and run-state handling
- basic evidence-aware native skill growth
- safer planner and fallback behavior

What Branch A should not continue doing:

- turning the old BBBC036 / Cell Painting perturbation workflow into the universal runtime
- abstracting legacy notebook prompts as if they were the system’s permanent core language
- optimizing notebook-centric architecture before shipping native scientific skills

After this point, notebook should remain available as an execution surface and legacy compatibility path, but it should not remain the center of architectural attention.
