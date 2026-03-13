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
| `--skip-review` | Run only the experiment stage, skipping optimization. |
| `--pipeline-config` | Support custom pipeline configuration path. |
| `--experiment-config` | Override Experiment stage configuration. |
| `--review-config` | Override Review stage configuration. |

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
│   │   ├── __init__.py                  # Package init
│   │   ├── config_loader.py             # Load/merge config + ${VAR} expansion
│   │   ├── llm_client.py                # LLM client + token metering
│   │   ├── idea_generator.py            # Idea/hypothesis generation
│   │   ├── execution_workflow.py        # Experiment stage entry: design & execute
│   │   ├── review_workflow.py           # Review stage entry: review & optimize
│   │   ├── prompt_orchestrator.py       # Orchestrate prompts/tasks across stages
│   │   ├── prompt_generator.py          # Generate notebook/code content
│   │   ├── prompt_executor.py           # GraphExecutor + prompt execution
│   │   ├── executor_engine.py           # Execution engine used by review
│   │   ├── notebook_autofix.py          # Auto-fix loop for failing notebooks
│   │   ├── experiment_report.py         # Experiment report generator
│   │   ├── task_graph.py                # Task DAG + dependency handling
│   │   ├── task_logger.py               # Task-level logging utilities
│   │   └── external_knowledge_mirothink.py # External knowledge integration
│   │
│   └── pipeline/                   # 🔧 INFRASTRUCTURE LAYER: Support systems
│       ├── config.py                 # Config Manager: Merges JSON configs & CLI args
│       ├── metrics.py                # Analytics: Calculates Success Rate, PCC, etc.
│       ├── report.py                 # Reporting: Generates final summaries (PDF/MD)
│       ├── advanced_metrics.py       # Deep Dive: Advanced statistical analysis
│       └── utils.py                  # Utils: Logging, paths, and subprocess helpers
│
├── configs/                        # ⚙️ CONFIGURATION: 3-Tier Inheritance Architecture
│   ├── pipeline_config.json        # Tier 1: Global defaults (dataset, LLM, literature, bio_kb)
│   ├── experiment_config.json      # Tier 2: Experiment overrides (model, timeouts, viz)
│   └── review_config.json          # Tier 3: Review overrides (model, optimization params)
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
        ├── generate_execution/     # Experiment stage outputs (Notebooks, Logs, Figures)
        ├── review_feedback/        # Review stage outputs (Optimized Code, Reviews)
        └── run_logs/               # System logs and terminal streams
```

## ⚙️ Key Configurations

The framework uses a **3-Tier Inheritance Architecture** for configuration management:

### Configuration Files

```
configs/
├── pipeline_config.json       # Tier 1: Global defaults (shared by all stages)
├── experiment_config.json     # Tier 2: Experiment-specific overrides only
└── review_config.json         # Tier 3: Review-specific overrides only
```

### Inheritance Pattern

**Stage configs inherit from pipeline config and override specific values:**

```python
# Effective config = pipeline_config ⊕ stage_config
# Where ⊕ means: stage_config overrides pipeline_config
```

### Key Benefits

*   **Single Source of Truth**: Shared settings (API keys, LLM configs, BioKB, Literature) defined once in `pipeline_config.json`
*   **Clear Overrides**: Stage configs only contain differences, making customization obvious
*   **Reduced Redundancy**: 17% total size reduction, eliminating duplicate API keys and settings
*   **Easy Maintenance**: Update shared settings in one place, apply everywhere

### Configuration Sections

*   **Global Settings** (in `pipeline_config.json`):
    *   `dataset_name`, `split_name`: Dataset and split configuration
    *   `llm`, `llm_report`: LLM API endpoints, keys, and model defaults
    *   `literature`: Literature search API keys (Serper, Jina) and parameters
    *   `bio_kb`: Biological knowledge base (ChEMBL, Reactome) settings
    *   `paths`: Data paths and output directories
    *   `exec`: Execution timeouts and fix rounds
    
*   **Experiment Stage** (overrides in `experiment_config.json`):
    *   LLM model selection for experiment generation
    *   Experiment-specific paths and parameters
    *   Hypergraph visualization settings
    
*   **Review Stage** (overrides in `review_config.json`):
    *   LLM model selection for review/optimization
    *   Review-specific paths and parameters
    *   Optimization hierarchy and protected sections

### Strategy

*   **LLM Engine**: Gemini 3 Pro (default).
*   **Experiment Stage**: Parallel hypothesis generation with self-correction.
*   **Execution Stage**: Long-running context with global timeouts (up to 100h).
*   **Review Stage**: Iterative optimization based on feedback (e.g., Pearson Correlation).

## 📊 Cost & Efficiency

**CellCausal** implements a **Contextual Memory** mechanism to optimize token usage:
*   Reduces token load by ~60% in later iterations.
*   Enables complex, multi-step optimizations at low cost.

## ❓FAQ: "为什么是 10 次迭代、为什么只看 10 个分子？"

### 1) 迭代次数在哪里控制？

*   评审优化轮数由配置项 `review.max_iterations` 控制（默认在 `review_config.json` 里是 `10`）。
*   实验阶段同样有 `experiment.max_iterations`，两阶段可分别设置。
*   若你希望更激进/更保守，可以直接把 `10` 改成你希望的值（例如 `3`、`20`）。

### 2) "只搜索 10 个分子"到底是什么意思？

这个项目里有**三层不同的“数量限制”**，很容易混淆：

1.  **提示词展示样本（默认前 10 个）**
    *   在给建模 LLM 的 prompt 中，`SMILES` 常以 `smiles_list[:10]` 展示。
    *   这主要是为了控制 token，不是把训练数据裁成 10 个。

2.  **机制先验构建样本（默认前 20 个）**
    *   机制链路（SMILES→target→pathway）在先验构建阶段默认取 `smiles_list[:20]`，用于“方向约束”。
    *   这是先验注入，不等于只对 20 个分子建模。

3.  **证据注入上限（可配置）**
    *   文献/BioKB 证据会做 `inject_max_items` 限制，避免把 prompt 淹没。
    *   BioKB 证据转为最终 evidence 时还有上限保护（最多 10 条），同样是为了可读性和稳定性。

### 3) 如果我有 1000+ 分子，只取 10 个有啥用？

核心逻辑是：**“全量数据训练 + 小样本知识注入”**。

*   大规模分子主要通过你的 H5/数据管线进入模型训练与评估。
*   “10/20 条 SMILES + 限量证据”用于给 LLM 提供可控的生物学锚点（避免 prompt 爆炸和噪声过大）。
*   这不是数据裁剪策略，而是**上下文预算策略**。

### 4) 你应该怎么调这套体系（1000+ 分子的建议）

*   先固定较小的 `max_iterations`（如 3-5）验证稳定性，再逐步提高。
*   在 `literature.bio_kb` 下使用分层采样（如 `sampling_strategy` / `adaptive_ratio` / `max_total_smiles`）控制 BioKB 查询成本。
*   按任务阶段调 `inject_max_items`：
    *   早期探索：小一些（例如 5-10）
    *   后期收敛：可适度加大
*   观察 `results/<dataset>/run_logs` 中每轮增益，如果后几轮几乎不涨，优先提升知识质量而不是盲目加轮次。
