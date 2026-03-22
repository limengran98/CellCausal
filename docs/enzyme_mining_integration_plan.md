# Enzyme-Mining Integration Plan

Scope: this note is an integration plan for the existing enzyme-mining assets already present in this repo. It is not a runtime rewrite. Its goal is to translate the user's notebooks and bundled model code into the current `skills/ + tools/ + artifacts + records/` architecture with the smallest credible bridge.

## 1. Files Actually Reviewed

The following repo-local files were directly inspected for this plan:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb)
- [JGI.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/JGI.ipynb)
- [EBI.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/EBI.ipynb)
- [README.md](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/README.md)
- [predict.py](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/predict.py)
- [run_catapro.sh](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/run_catapro.sh)
- [generate_features.py](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/generate_features.py)
- [sample_inp.csv](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/samples/sample_inp.csv)
- [catapro_test-pred.csv](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/catapro_test-pred.csv)
- [output_sequences.zip](/home/lmr/CellCausal-main/references/enzyme_mining/output_sequences.zip)

No requested path was missing in this repo snapshot.

## 2. Existing User Workflow Extracted From Notebooks

### A. Multi-source sequence acquisition

The existing workflow is not one abstract "enzyme-mining" prompt. It is a concrete sequence curation pipeline.

#### EBI / UniProt / UniRef retrieval

Observed in:

- [EBI.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/EBI.ipynb)
- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 9
- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 10

Actual steps present:

- Query EBI search with a text phrase such as `"dihydroxy acid dehydratase"`.
- Enumerate source databases:
  - `UniRef100`
  - `UniRef90`
  - `UniRef50`
  - `UniProtKB`
  - plus `EPO`, `JPO`, `USPTO` in the same retrieval loop
- Cache ID lists per source.
- Download FASTA entries in parallel with retries.
- Save per-ID FASTA files under source-specific folders.
- Merge downloaded FASTA into source-level merged FASTA outputs.
- Retry previously failed IDs from `*_failed_ids.txt`.

Interpretation:

- This is already a real source adapter pattern, even though it lives inside notebooks.
- The notebook logic separates:
  - source search
  - ID cache
  - FASTA download
  - retry bookkeeping

#### JGI retrieval

Observed in:

- [JGI.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/JGI.ipynb)
- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cells 12-13

Actual steps present:

- Read JGI gene IDs from `id.txt`.
- Fetch FASTA from JGI IMG gene detail page using `gene_oid`.
- Parse returned HTML with `BeautifulSoup`.
- Extract FASTA from `<pre>` block.
- Write to `all.fasta`.
- Support resume/append mode from a chosen restart gene ID.

Interpretation:

- JGI is currently a separate source adapter with its own access pattern.
- Unlike EBI, it is not an ID search pipeline; it is an ID-to-FASTA fetch pipeline.

#### NCBI step

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 7
- [output_sequences.zip](/home/lmr/CellCausal-main/references/enzyme_mining/output_sequences.zip)

Important clarification:

- The notebook heading says `ncbi 提取序列代码`, but the actual code is a batch `CD-search / DomainHits` workflow against NCBI Conserved Domain Search, not a simple sequence download script.
- `output_sequences.zip` contains 239 split FASTA files such as `split_1.fasta`, `split_2.fasta`, etc.
- These split FASTA files are used as batch input to NCBI's web CD-search through Playwright automation.

Actual steps present:

- Unzip `output_sequences.zip` into `input_fasta/`.
- Submit FASTA batches to `https://www.ncbi.nlm.nih.gov/Structure/bwrpsb/bwrpsb.cgi`.
- Wait for `"Search completed successfully"`.
- Download/collect `*_DomainHits_rep.txt`-style results.
- Track failed batches and log execution.

Interpretation:

- In the current user workflow, NCBI is a domain-screening stage, not the primary sequence-source stage.
- This matters because in the future architecture NCBI should likely be modeled as a domain-annotation adapter, not a generic source retriever.

### B. FASTA merge

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 1

Actual step:

- Merge all FASTA files from a folder into one `merged_all*.fasta`.

Role in workflow:

- This is the initial consolidation point after multi-source acquisition.
- It creates the first global candidate pool.

### C. Exact-sequence deduplication

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 3
- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 17

Actual step:

- Use amino-acid sequence exact identity as the dedupe criterion.
- Keep only the first occurrence for duplicated sequences.

Role in workflow:

- First dedupe creates `文件1+`.
- Final dedupe runs again after intersection filtering to ensure the final set is globally nonredundant.

This is worth preserving almost unchanged.

### D. Domain filtering (`DomainHits`)

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 5
- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 20

Actual step:

- Repair and merge `*_DomainHits_rep.txt`.
- Expect 11 tab-separated columns.
- Build:
  - `merged_full.csv`
  - `merged_filtered.csv`
  - `merged_results.txt`
- Filter out `Incomplete = N/C`.
- Keep hits whose `Short name` matches domain keywords:
  - `GMC_oxred_C`
  - `GMC_oxred_N`
  - `BBE`
  - `FAD`

Role in workflow:

- This is the real biochemical narrowing step in the user's pipeline.
- It converts raw sequence acquisition into a domain-constrained candidate set.

### E. Intersection retention

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 15

Actual step:

- Read a `summary.txt` containing retained headers.
- Extract target IDs from lines containing `>`.
- Filter `文件1+` FASTA so only IDs present in `summary.txt` remain.
- Write `filtered_by_summary.fasta`.

Role in workflow:

- This is the set-intersection stage between:
  - the broad merged/deduped sequence pool
  - the domain-screened shortlist

### F. Final nonredundant set

Observed in:

- [挖酶.ipynb](/home/lmr/CellCausal-main/references/enzyme_mining/挖酶.ipynb) cell 17

Actual step:

- Run exact-sequence dedupe again on `filtered_by_summary.fasta`.
- Output `final_unique_sequences.fasta`.

Role in workflow:

- This is the actual candidate set that should feed downstream ranking or experimental planning.

## 3. Candidate Mining Architecture Under Current Runtime

The current runtime already has the right high-level shape:

- native skill orchestration in `skills/`
- low-level logic in `tools/`
- structured outputs in `artifacts`
- execution handoff through notebook surface only when needed

The enzyme-mining assets should map into that architecture as follows.

### A. Source adapters

These should become source-specific adapters, not be buried inside the skill itself.

Recommended adapter layer:

- `ebi_search_adapter`
  - text query -> source-specific IDs -> FASTA retrieval
  - covers `UniProtKB`, `UniRef100`, `UniRef90`, `UniRef50`
- `jgi_gene_adapter`
  - gene IDs -> FASTA retrieval via JGI HTML page parsing
- `ncbi_domain_adapter`
  - FASTA batch -> DomainHits result set
  - this is annotation/screening, not primary sequence retrieval

Why adapters:

- each source has different request semantics
- retry logic differs by source
- auth/rate-limit/resume behavior differs by source
- future failures should be attributable to a named source layer

### B. Processing tools

These should become reusable tools, because they are transformation steps rather than user-facing orchestration.

Recommended processing tools:

- `fasta_merge_tool`
  - merge many FASTA files into one pool
- `sequence_dedupe_tool`
  - exact-sequence dedupe with provenance preservation
- `domainhits_repair_merge_tool`
  - repair NCBI `DomainHits_rep.txt` and merge to structured table
- `domain_filter_tool`
  - keyword/domain filtering, incomplete-hit filtering
- `candidate_intersection_tool`
  - intersect shortlist IDs against the broad nonredundant FASTA pool
- `fasta_export_tool`
  - write final candidate FASTA and summary table

These belong in `tools/`, not in `skills/`.

### C. Enzyme-mining skill orchestration

The `enzyme-mining` skill should not manually implement source scraping loops line by line.

It should orchestrate:

1. query focus normalization
2. source selection summary
3. candidate acquisition status
4. dedupe/domain/intersection processing status
5. evidence synthesis and next questions
6. optional notebook-ready handoff scaffold

In other words, the skill should answer:

- what was searched
- what survived filtering
- why those candidates are being prioritized
- what can be validated next

It should not itself become:

- a huge FASTA ETL notebook
- a browser automation script
- a model-serving loop

### D. Recommended artifacts

The following outputs should become explicit artifacts:

- `candidate_source_summary`
  - which sources were queried or summarized
  - counts and retrieval status
- `candidate_fasta_pool`
  - merged or deduped FASTA manifest
- `domain_filter_table`
  - structured domain hits and filtering output
- `candidate_set`
  - final nonredundant candidate enzyme table/FASTA manifest
- `ranking_result`
  - optional CataPro or other model ranking output
- `experiment_scaffold`
  - notebook-ready validation handoff

This is the right place to preserve the user's original workflow without forcing everything through notebook execution.

## 4. CataPro Integration Path

### A. What CataPro actually expects

Observed in:

- [README.md](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/README.md)
- [predict.py](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/predict.py)
- [sample_inp.csv](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/samples/sample_inp.csv)

Input format:

- CSV table with at least:
  - `Enzyme_id`
  - `type`
  - `sequence`
  - `smiles`

The bundled sample file confirms this shape.

Internal feature path:

- sequence -> ProtT5 embedding
- smiles -> MolT5 embedding
- smiles -> MACCS fingerprint
- concatenated features -> 10-fold ensemble models

Output format:

Observed in:

- [catapro_test-pred.csv](/home/lmr/CellCausal-main/references/enzyme_mining/CataPro-master/CataPro-master/inference/catapro_test-pred.csv)

Columns:

- `fasta_id`
- `smiles`
- `pred_log10[kcat(s^-1)]`
- `pred_log10[Km(mM)]`
- `pred_log10[kcat/Km(s^-1mM^-1)]`

### B. How to convert CellCausal candidate sets into ranking input

The bridge is conceptually simple:

1. take final candidate enzyme records
   - `enzyme_id`
   - `sequence`
2. pair them with one or more substrate SMILES
3. write a ranking input CSV in CataPro format:
   - `Enzyme_id`
   - `type = wild` by default unless user says mutant
   - `sequence`
   - `smiles`
4. run `predict.py`
5. collect ranking CSV as structured artifact

This means CataPro should be modeled as a ranking bridge, not as the whole enzyme-mining skill.

### C. When CataPro can be run for real

Real execution is plausible only when all of the following are true:

- local runtime has:
  - `torch`
  - `transformers`
  - `numpy`
  - `pandas`
  - `rdkit`
- the bundled model assets are present and readable
- the inference code is importable or callable from a thin wrapper
- input CSV is built correctly from candidate sequences and substrate SMILES
- compute device assumptions are satisfied
  - README uses `cuda:0`
  - but CPU fallback feasibility should be checked explicitly before relying on it

### D. When CataPro should stay a controlled bridge

It should remain a controlled bridge if any of the following hold:

- required model weights are incomplete or incompatible
- environment packages do not match inference requirements
- runtime cost is too high for a default skill path
- substrate SMILES are missing
- candidate sequence set is not yet in a structured table form

In the current runtime, the safe first step is:

- ranker input preparation
- environment/status check
- explicit `ranking_ready` or `ranking_blocked` result

Not:

- silent auto-launch of heavyweight inference

## 5. Proposed MVP Implementation Scope

The first enzyme-mining integration MVP should stay narrow.

### Do now

1. Candidate source summary

- represent which source families matter:
  - EBI / UniProt / UniRef
  - JGI
  - NCBI domain screen
- record whether input came from:
  - existing files
  - local bundled outputs
  - user-provided FASTA/CSV

2. Dedupe and domain filtering bridge

- expose thin wrappers around the exact-sequence dedupe and domain-filter logic
- accept local files as inputs
- return structured counts and output paths

3. Ranking status / model bridge

- build a `catapro_ranking_bridge`
- it should first support:
  - input CSV generation
  - environment/model availability checks
  - controlled invocation status
- it may optionally run real inference if all local conditions are satisfied

4. Notebook-ready scaffold

- keep the current `experiment_scaffold` output pattern
- enrich it with:
  - candidate set summary
  - ranking-ready substrate assumptions
  - validation notebook sections for enzyme ranking and filtering audit

### Do not do yet

- fully automated live crawling of all external sequence sources
- always-on browser automation against NCBI/JGI
- large-scale database synchronization
- background downloading of large model assets
- collapsing the whole workflow into a notebook monolith
- making CataPro a mandatory dependency for every enzyme-mining request

## 6. Mapping to Current CellCausal Architecture

### A. `skills/`

`enzyme-mining` should remain the main user-facing skill.

Responsibilities:

- normalize the biological focus
- decide whether the user is asking for:
  - candidate discovery
  - shortlist refinement
  - ranking
  - notebook-ready validation handoff
- compose tool outputs into:
  - candidate summary
  - rationale
  - evidence
  - next questions

### B. `tools/`

Future enzyme-mining tool layer should contain:

- source adapters
  - EBI/UniProt/UniRef
  - JGI
  - NCBI domain screen
- processing tools
  - FASTA merge
  - exact dedupe
  - domain hits repair/filter
  - set intersection
- ranking bridge
  - CataPro input builder
  - CataPro environment/status wrapper

### C. Artifacts

Recommended artifact types:

- `candidate_source_summary`
- `candidate_fasta_pool`
- `domain_filter_table`
- `candidate_set`
- `ranking_input`
- `ranking_result`
- `experiment_scaffold`

### D. `records/`

For research-grade traceability, enzyme-mining runs should record:

- source files used
- query focus
- sequence counts before and after dedupe
- domain filter criteria
- final candidate count
- whether CataPro was:
  - ready
  - run
  - blocked
- notebook handoff status

### E. Notebook handoff

Notebook should remain a downstream execution surface.

The enzyme-mining skill should hand off:

- candidate table summary
- ranking-ready input assumptions
- substrate SMILES assumptions
- notebook sections to generate

It should not automatically call notebook execution when the user asks for validation.

## 7. First Real MVP Boundary Under Current Runtime

Under the current runtime, the minimum credible first implementation is:

1. Keep `enzyme-mining` as the main native skill.
2. Add thin wrappers around existing local workflow steps:
   - merge
   - dedupe
   - domain filter
   - intersection
3. Add a `CataPro` ranking bridge that can at least produce:
   - ranking input CSV
   - environment/model readiness status
   - optional inference result if local execution is safe
4. Keep notebook follow-up limited to:
   - `experiment_scaffold`
   - notebook-ready handoff metadata

This boundary is intentionally smaller than "full enzyme mining automation", but large enough to preserve the user's real workflow.

## 8. Recommended Next Implementation Order

### 1. Candidate mining bridge first

Reason:

- it preserves the user's actual workflow backbone
- it is less environment-fragile than model inference
- it creates the candidate set that every later ranking step depends on
- it gives immediate value even if CataPro cannot run yet

### 2. CataPro ranking bridge second

Reason:

- it depends on candidate sequences plus substrate SMILES already being structured
- it has heavier model/runtime assumptions
- it should enter as an explicit bridge with clear readiness checks

### 3. Notebook handoff enrichment third

Reason:

- once candidate set and ranking status are structured, notebook generation becomes a cleaner downstream validation surface
- this keeps notebook in its proper place rather than forcing it to become the center again

## Final Integration Conclusion

The user's existing "挖酶" workflow is already a strong recipe. The right move is not to replace it with a vague new agent loop. The right move is:

- preserve the source-specific acquisition logic as adapters
- preserve merge/dedupe/domain/intersection as tools
- keep enzyme-mining skill responsible for orchestration and synthesis
- treat CataPro as a ranking bridge
- keep notebook as optional validation handoff, not default execution truth

That is the smallest integration path that respects both the existing user assets and the current CellCausal runtime design.
