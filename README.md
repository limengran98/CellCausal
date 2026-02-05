# CellMechanist

CellMechanist is an autonomous AI agent framework designed for Virtual Cell Modeling (VCM).


## 🛠️ Installation

```bash
conda create --name CellMechanist python=3.11.14
conda activate CellMechanist
cd CellMechanist
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f [https://download.pytorch.org/whl/cu118/torch_stable.html](https://download.pytorch.org/whl/cu118/torch_stable.html)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-2.0.1+cu118.html](https://data.pyg.org/whl/torch-2.0.1+cu118.html)
pip install -r requirements.txt

```

## 📂 Project Structure

```
CellMechanist/                  <-- Project Root
│
├── data/                       <-- Data Storage Directory (Inferred from H5 path fix)
│
├── results/                    <-- Output Directory (Automatically generated)
│
├── configs/                    <-- Configuration Files Directory (Inferred from run_pipeline.py)
│   ├── pipeline_config.json
│   ├── experiment_config.json
│   └── review_config.json
│
├── prompts/                    <-- Prompt YAML Storage Directory
│   ├── pipeline_prompt.yaml
│   ├── review_optimize.yaml
│   ├── experiment_report.yaml
│   ├── final_report.yaml
│   ├── advanced_metrics.yaml
│   ├── autofix.yml
│   └── idea.yml
│
│
├── cellscientist/              <-- Package Root
│   ├── __init__.py
│   │
│   ├── core/                   <-- Core Logic Layer (Phase 2 & 3 implementation)
│   │   ├── __init__.py
│   │   ├── config_loader.py    <-- Configuration loader and variable parser
│   │   ├── llm_client.py       <-- TokenMeter and LLM call wrapper
│   │   ├── execution_workflow.py  <-- Phase 2 Entry (Design & Execute)
│   │   ├── review_workflow.py     <-- Phase 3 Entry (Review & Optimize)
│   │   ├── prompt_orchestrator.py <-- Coordinator for generation, execution, and analysis
│   │   ├── prompt_generator.py    <-- Notebook content generator
│   │   ├── prompt_executor.py     <-- Contains GraphExecutor class
│   │   ├── executor_engine.py     <-- Pure execution engine (Used in Review phase)
│   │   ├── notebook_autofix.py    <-- Auto-fix logic
│   │   ├── experiment_report.py   <-- Experiment report generator
│   │   ├── task_graph.py          <-- Task graph management (Used in Review phase)
│   │   └── external_knowledge_mirothink.py
│   │
│   └── pipeline/               <-- Pipeline Orchestration Layer (Runner logic)
│       ├── __init__.py
│       ├── run_pipeline.py     <-- Unified entry (Refactored version of run_cellscientist.py)
│       ├── config.py           <-- Pipeline configuration merging logic
│       ├── metrics.py          <-- Metric extraction and scoreboard (Regex fix applied here)
│       ├── report.py           <-- Final summary report generation
│       ├── advanced_metrics.py <-- Advanced metrics analysis
│       └── utils.py            <-- Path discovery and log streaming utilities
│
└── run_cellscientist.py         <-- Root-level startup script, typically calls cellscientist.pipeline.run_pipeline.main

```

## ⚙️ Experiment Settings & Environment

### Hardware & Software Infrastructure

Experiments are conducted on high-performance nodes tailored.

* **CPU:** Dual Intel Xeon Platinum 8336C @ 2.30GHz
* **GPU:** NVIDIA RTX 5880 Ada Generation (48GB VRAM)
* **Memory:** 512 GB DDR4 ECC
* **Software:** Python 3.11.14, PyTorch 2.0.1+cu118, PyG 2.3.0, CUDA 11.8

### Hyperparameters (Key Configurations)

The Dual-Space Bilevel Optimization is controlled via hierarchical configs:

* **LLM Engine:** Gemini 3 Pro (Temp: 0.5 - 0.7)
* **Design Phase:** 4 parallel hypothesis branches; Max 3 self-correction fix rounds.
* **Execution Phase:** Global timeout 100h; Step timeout 5h; Max 5 debugging rounds.
* **Review Phase:** Max 10 optimization iterations; Optimized via Pearson Correlation Coefficient (PCC).

### Cost Efficiency

CellMechanist minimizes cost through a **Contextual Memory** mechanism that reduces token load by ~60% in later iterations.

* **Average Run (3-5 iterations):** $1.00 - $2.00 USD
* **Complex Run (10 iterations):** < $5.00 USD

## 🚀 Usage


```bash
python run_pipeline.py

```

