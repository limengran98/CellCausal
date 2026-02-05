# CellCausal

**CellCausal** is an autonomous AI agent framework designed for Virtual Cell Modeling (VCM), integrating Design, Execution, and Review loops into a unified pipeline.

## 🚀 Quick Start

Run the complete pipeline (Design & Execution → Review & Optimization):

```bash
python run_pipeline.py
```

### Common Options

| Argument | Description |
| :--- | :--- |
| `--skip-review` | Run only the experiment stage (Phase 2), skipping optimization. |
| `--pipeline-config` | Support custom pipeline configuration path. |
| `--experiment-config` | Override Experiment (Phase 2) configuration. |
| `--review-config` | Override Review (Phase 3) configuration. |

## 📂 Project Structure

A complete overview of the **CellCausal** architecture:

```
CellCausal/                         <-- Project Root
│
├── run_pipeline.py                 # 🚀 MAIN ENTRY POINT: Orchestrates the full lifecycle
│
├── cellscientist/                  # 📦 CORE PACKAGE: Main logic implementation
│   │
│   ├── core/                       # 🧠 INTELLIGENCE LAYER: AI Workflows
│   │   ├── execution_workflow.py     # Phase 2 Entry: Design ideas & execute experiments
│   │   ├── review_workflow.py        # Phase 3 Entry: Analyze results & optimize code
│   │   ├── prompt_orchestrator.py    # Master Controller: Manages agent state & tasks
│   │   ├── prompt_generator.py       # Code Gen: Creates executable Jupyter notebooks
│   │   ├── executor_engine.py        # Sandbox: Executes generated code safely
│   │   ├── task_graph.py             # Dependency Manager: Handles complex task DAGs
│   │   ├── llm_client.py             # LLM Interface: Handles API calls & token counting
│   │   ├── notebook_autofix.py       # Self-Healing: Automatically fixes coding errors
│   │   └── external_knowledge_*.py   # RAG: Retrieves external biological context
│   │
│   └── pipeline/                   # 🔧 INFRASTRUCTURE LAYER: Support systems
│       ├── config.py                 # Config Manager: Merges JSON configs & CLI args
│       ├── metrics.py                # Analytics: Calculates Success Rate, PCC, etc.
│       ├── report.py                 # Reporting: Generates final summaries (PDF/MD)
│       ├── advanced_metrics.py       # Deep Dive: Advanced statistical analysis
│       └── utils.py                  # Utils: Logging, paths, and subprocess helpers
│
├── configs/                        # ⚙️ CONFIGURATION: Control parameters
│   ├── pipeline_config.json        # Global Settings: Dataset, LLM models, Env
│   ├── experiment_config.json      # Phase 2 Props: Search width, fix rounds, timeouts
│   └── review_config.json          # Phase 3 Props: Optimization rounds, top-k selection
│
├── prompts/                        # 📝 PROMPT TEMPLATES: System instructions (YAML)
│   ├── pipeline_prompt.yaml        # General agent behaviors and personas
│   ├── idea.yml                    # Hypothesis generation prompts
│   ├── autofix.yml                 # Error correction strategies
│   ├── review_optimize.yaml        # Code critique & optimization guides
│   └── final_report.yaml           # Report generation templates
│
├── data/                           # 💾 DATASETS: Input biological data (H5/CSV)
│
└── results/                        # 📊 ARTIFACTS: All generated outputs
    ├── <dataset_name>/
        ├── generate_execution/     # Phase 2 Outputs (Notebooks, Logs, Figures)
        ├── review_feedback/        # Phase 3 Outputs (Optimized Code, Reviews)
        └── run_logs/               # System logs and terminal streams
```

## ⚙️ Key Configurations

The framework uses a **Dual-Space Bilevel Optimization** strategy, controlled via `configs/`:

*   **LLM Engine**: Gemini 3 Pro (default).
*   **Design Phase**: Parallel hypothesis generation with self-correction.
*   **Execution Phase**: Long-running context with global timeouts (up to 100h).
*   **Review Phase**: Iterative optimization based on feedback (e.g., Pearson Correlation).

## 📊 Cost & Efficiency

**CellCausal** implements a **Contextual Memory** mechanism to optimize token usage:
*   Reduces token load by ~60% in later iterations.
*   Enables complex, multi-step optimizations at low cost.
