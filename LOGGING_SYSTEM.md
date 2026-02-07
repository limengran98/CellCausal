# 4-Tier Logging System Documentation

## Overview

CellScientist implements a comprehensive 4-tier logging system that captures complete execution evidence chain while providing a clean, user-friendly console experience.

## Log Files Structure

After running the pipeline, you'll find these files in `results/<DATASET>/run_logs/pipeline_<timestamp>/`:

```
pipeline_20260206_154241/
├── console_output.log          # Tier 1: Console-visible content backup (NEW v2)
├── execution_detail.log        # Tier 3: Complete execution trace with all prefixes (NEW)
├── experiment.log              # Tier 2: Experiment stage details
├── review.log                  # Tier 2: Review stage details
├── evidence_chain.json         # Tier 4: Structured audit trail (NEW)
├── notebook.ipynb              # Generated code
├── best_model.pkl              # Trained model
└── metrics/
    ├── iteration_scores.json
    └── advanced_metrics.json
```

## The 4 Tiers

### Tier 1: Console Output (User-Facing)
**Purpose**: Clean, hierarchical progress display with tree structure (~20-30 lines)

**Features**:
- Tree-structured format using ├─, └─, │ characters
- Visual indicators: ✅ ❌ ⚠️ 📈 📊 🧠 ⚙️ ⏱️ 📁 🔎 📚
- Clear stage separation
- Key metrics only (no debug spam)
- Real-time progress tracking

**Example**:
```
🔬 EXPERIMENT STAGE
│
├─ 🔎 Iteration 1/3
│  ├─ 📚 Knowledge: Retrieved 15 papers and 25 pathways
│  ├─ 🧠 Generate: Strategy synthesized (2.2KB) → Code generated (13.1KB, 9 cells)
│  ├─ ⚙️  Execute: 9 cells → ✅ Success (no autofix)
│  ├─ 💰 Cost: 12.5K tokens | 4.2s LLM time
│  └─ 📊 Score: PCC=0.1769 (Target: >0.5) ⚠️ Below threshold
│
├─ 🔎 Iteration 2/3
│  ├─ 📚 Knowledge: Retrieved 15 papers and 25 pathways
│  ├─ 🧠 Generate: New strategy → Code patched
│  ├─ ⚙️  Execute: Cell T6 ❌ → 🔧 Auto-fix (Round 1) → ✅ Recovered
│  ├─ 💰 Cost: 15.1K tokens | 5.8s LLM time
│  └─ 📊 Score: PCC=0.2832 📈 +60% improvement
│
└─ 🔎 Iteration 3/3
   ├─ 📚 Knowledge: Retrieved 15 papers and 25 pathways
   ├─ 🧠 Generate: Refined architecture
   ├─ ⚙️  Execute: Full pipeline → ✅ Success
   ├─ 💰 Cost: 13.8K tokens | 4.9s LLM time
   └─ 📊 Score: PCC=0.3791 📈 Best@Budget ✅

⏱️  Experiment completed in 2377.5s

🔬 REVIEW STAGE
├─ 🔎 Iteration 1/5: Baseline PCC=0.3791
│  ├─ 🧠 Strategy: Architecture refinement → Loss function tuning
│  ├─ ⚙️  Execute: Training... → ✅ Success
│  ├─ 💰 Cost: 8.2K tokens
│  └─ 📊 Result: PCC=0.3654 ⚠️ Regression (-3.6%)
│
├─ 🔎 Iteration 2/5
│  ├─ 🧠 Strategy: Rollback + Data fusion → Attention mechanism
│  ├─ ⚙️  Execute: Model training... → ✅ Success
│  ├─ 💰 Cost: 9.1K tokens
│  └─ 📊 Result: PCC=0.4123 📈 Best score! (+8.8%)

⏱️  Review completed in 1172.4s
⏱️  Total pipeline time: 3549.9s
```

**This output is also saved to**: `console_output.log` for future reference

### Tier 2: Stage Logs (Detailed Stage Execution)
**Files**: `experiment.log`, `review.log`

**Purpose**: Detailed per-stage execution information

**Content**:
- Per-iteration detailed execution
- LLM API calls and responses
- Code generation details
- Execution results
- Error messages and fixes

**Format**: Standard log format with timestamps
```
2026-02-06 15:42:41 [INFO] Starting iteration 1/3
2026-02-06 15:43:12 [DEBUG] LLM request: model=gemini-3-pro-all, tokens=4521
2026-02-06 15:45:23 [INFO] Strategy generated: 2.2KB
2026-02-06 15:47:56 [INFO] Code generated: 13.1KB
2026-02-06 15:52:18 [INFO] Execution completed: PCC=0.1769
```

### Tier 3: Full Execution Log (Complete Trace)
**File**: `execution_detail.log` (NEW)

**Purpose**: Complete stdout/stderr capture with NO filtering - captures EVERYTHING

**Content - Three Types of Output**:

1. **[CELL_CONSOLE] prefixed lines** - User-facing progress (also shown in console)
   - Iteration headers
   - Knowledge retrieval summaries
   - Generation summaries
   - Execution progress
   - Cost and score information

2. **[DETAIL] prefixed lines** - Debug and detailed information (console-silent)
   - [SETUP], [LOOP], [ARCHIVE] - Setup and loop details
   - [ORCH], [GEN], [EXEC], [LLM] - Module-specific debug messages
   - [FIX], [GRAPH], [DATA] - Internal processing details
   - File paths, audits, internal state

3. **Unprefixed lines** - Third-party library output (console-silent)
   - Python warnings and errors
   - NumPy, Pandas, PyTorch messages
   - Jupyter notebook outputs
   - Other library diagnostics

**Format**: Raw subprocess output with prefixes
```
[CELL_CONSOLE] 🔬 EXPERIMENT STAGE
[CELL_CONSOLE] 
[CELL_CONSOLE] ├─ 🔎 Iteration 1/3
[DETAIL] [SETUP] Idea Mode: OFF
[DETAIL] [SETUP] API Key Injected: ...xyz7
[DETAIL] [LOOP] Max Iters: 3 | Target: PCC > 0.5
[DETAIL] [LOOP] Workspace Prefix: workspace
[DETAIL] [LOOP] Save Root: /path/to/results
[CELL_CONSOLE] ├─ 📚 Knowledge: Retrieved 15 papers and 25 pathways
[DETAIL] [GEN] Saving prompt snapshot to: /path/to/prompt_snapshot.yaml
[CELL_CONSOLE] ├─ 🧠 Generate: Strategy synthesized (2.2KB) → Code generated (13.1KB, 9 cells)
[DETAIL] [EXEC] Running Notebook: /path/to/notebook.ipynb
[DETAIL] [EXEC] Cell 1/9: Data Loading
UserWarning: NumPy version mismatch
[DETAIL] [EXEC] Cell 2/9: Preprocessing
[CELL_CONSOLE] ├─ ⚙️  Execute: 9 cells → ✅ Success (no autofix)
[CELL_CONSOLE] ├─ 💰 Cost: 12.5K tokens | 4.2s LLM time
[DETAIL] [CHECK] workspace_iter001 | PCC: 0.1769 (Target > 0.5)
[CELL_CONSOLE] └─ 📊 Score: PCC=0.1769 (Target: >0.5) ⚠️ Below threshold
```

**Key Features**:
- Preserves exact subprocess output with prefixes
- Enables reconstruction of full execution flow
- Debugging third-party integration issues
- Complete audit trail for compliance

### Tier 4: Evidence Chain (Structured Audit Trail)
**File**: `evidence_chain.json` (NEW)

**Purpose**: Machine-readable structured audit trail

**Structure**:
```json
{
  "pipeline_id": "BBBC036_20260206_154241",
  "dataset": "BBBC036",
  "start_time": "2026-02-06T15:42:41Z",
  "end_time": "2026-02-06T16:41:30Z",
  "total_duration_seconds": 3549.0,
  "config_snapshot": {
    "pipeline": {...},
    "stages": {
      "experiment": {...},
      "review": {...}
    }
  },
  "experiment_stage": {
    "iterations": [
      {
        "iteration": 1,
        "timestamp_start": "2026-02-06T15:42:41Z",
        "timestamp_end": "2026-02-06T15:52:18Z",
        "duration_seconds": 577.3,
        "strategy_generation": {
          "model": "gemini-3-pro-all",
          "hypothesis": "...",
          "evidence_citations": ["L1", "L2", "B1"]
        },
        "code_generation": {
          "code_size_bytes": 13421,
          "rationale": "..."
        },
        "execution_trace": {
          "cells_executed": 9,
          "successful_cells": 9,
          "failed_cells": 0
        },
        "evaluation": {
          "metrics": {"PCC": 0.1769},
          "status": "below_threshold"
        }
      }
    ]
  },
  "review_stage": {
    "iterations": [...]
  }
}
```

**Features**:
- Complete structured history
- Machine-readable JSON format
- Timestamped events
- Evidence linking between stages
- Iteration-to-iteration learning captured

## Usage

### Basic Usage
The logging system is automatically initialized when you run the pipeline:

```bash
python run_pipeline.py --task BBBC036
```

### Programmatic Usage
You can also use the TieredLogger directly in your code:

```python
from cellscientist.pipeline.logging_system import create_tiered_logger

# Initialize logger
logger = create_tiered_logger(
    run_dir="/path/to/logs",
    config=pipeline_config,
    dataset_name="BBBC036"
)

# Tier 1: Console output
logger.console_info("Starting experiment", level=0)
logger.console_info("Iteration 1", level=1, symbol="🔎")

# Tier 2: Stage-specific logs
logger.stage_log("experiment", "Detailed message", "info")

# Tier 3: Full execution log
logger.full_log("Debug information")

# Tier 4: Evidence chain
logger.add_evidence(
    stage="experiment",
    iteration=1,
    evidence_type="strategy_generation",
    data={
        "model": "gemini-3-pro-all",
        "hypothesis": "VAE architecture...",
        "citations": ["L1", "B2"]
    }
)

# Save and finalize
logger.finalize()  # Saves evidence_chain.json
```

## Enhanced Configuration Display

The pipeline now shows comprehensive configuration at startup:

| Column | Description |
|--------|-------------|
| Stage | Experiment or Review |
| Model | LLM model name |
| Literature | Status (✓/✗) and top-k value |
| BioKB | Status (✓/✗) and top-k value |
| GPU | GPU device number or CPU |
| Timeout | Human-readable timeout (100h, 5h, etc.) |
| Max Iters | Maximum iterations for the stage |
| Cache | LLM cache status (✓/✗) |

## Implementation Details

### V2 Refactor (Comprehensive Logging)

The logging system was comprehensively refactored to eliminate scattered print statements and provide unified, user-friendly output:

1. **Modified**: `cellscientist/pipeline/utils.py` 
   - Enhanced `run_cmd_streamed()` with prefix-based routing
   - Recognizes `[CELL_CONSOLE]` and `[DETAIL]` prefixes
   - Routes output to appropriate destinations

2. **Modified**: `run_pipeline.py`
   - Renamed `console_filtered.log` → `console_output.log`
   - Proper file handle management
   - TieredLogger integration

3. **Modified**: All subprocess modules (11 files)
   - Added unified `_log(msg, *, console=bool)` helper function
   - Replaced ~215 print() statements across:
     - `cellscientist/core/execution_workflow.py` (28 prints)
     - `cellscientist/core/prompt_orchestrator.py` (17 prints)
     - `cellscientist/core/prompt_generator.py` (12 prints)
     - `cellscientist/core/prompt_executor.py` (22 prints)
     - `cellscientist/core/executor_engine.py` (21 prints)
     - `cellscientist/core/llm_client.py` (11 prints)
     - `cellscientist/core/notebook_autofix.py` (6 prints)
     - `cellscientist/core/review_workflow.py` (85 prints)
     - `cellscientist/pipeline/utils.py` (13 prints)

4. **Enhanced**: Console output format
   - Tree-structured hierarchy with `├─`, `│`, `└─`
   - Rich emoji indicators: 🔎 📚 🧠 ⚙️ 💰 📊 ✅ ❌ 🔧 ⚠️ 📈
   - Smart number formatting (12.5K, 1.2MB)
   - Clear progress tracking

### Design Decisions

**Why prefix-based routing?**
- Subprocess modules can't share TieredLogger instances (separate processes)
- Print-based communication is simple and reliable
- Parent process filters output based on prefixes
- Zero dependencies on complex IPC mechanisms

**Logging Strategy**:
- `console=True` → User-facing progress (iterations, scores, costs, errors)
- `console=False` → Debug details (setup, orchestration, internal state)

### Backward Compatibility

✅ **Fully backward compatible**:
- experiment.log and review.log continue to capture full subprocess output
- Existing log parsers still work
- All subprocess execution unchanged
- Only added new files (console_output.log, execution_detail.log)

## Benefits

1. **Clean Console**: Tree-structured output with only essential progress (~20-30 lines per iteration)
2. **Complete Capture**: Everything logged to execution_detail.log with prefixes
3. **User-Friendly**: Real-time progress with visual indicators and smart formatting
4. **Structured Audit**: Machine-readable evidence chain in JSON format
5. **Backward Compatible**: Existing logs (experiment.log, review.log) still work
6. **Extensible**: Easy to add more console messages or detail logs
7. **No Long Silences**: Users see progress continuously, not just at iteration end
8. **Debug-Friendly**: Full detail log preserves all information for troubleshooting

## Troubleshooting

### "Where did all my debug output go?"
Debug output is still captured in `execution_detail.log`. Only console display is filtered.

### "How do I see more detail in console?"
Subprocess modules control console visibility via `_log(msg, console=True)`. To add more:
1. Find the relevant _log() call in the module
2. Change `console=False` to `console=True`

### "Console output seems delayed"
This is normal - subprocesses buffer output. The system uses `flush=True` to minimize delays.

### "How do I parse the logs programmatically?"
Use `evidence_chain.json` for structured data, or parse `execution_detail.log` by prefix:
```python
with open('execution_detail.log') as f:
    for line in f:
        if line.startswith('[CELL_CONSOLE]'):
            # User-facing message
            msg = line[len('[CELL_CONSOLE] '):]
        elif line.startswith('[DETAIL]'):
            # Debug message
            msg = line[len('[DETAIL] '):]
        else:
            # Third-party output
            pass
```
