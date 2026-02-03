# CellScientist

CellScientist is an autonomous AI agent framework designed for Virtual Cell Modeling (VCM). It employs a Dual-Space Bilevel Optimization strategy to align symbolic scientific hypotheses with computational code implementation.

The system operates through a structured Task Hypergraph, performing evolutionary optimization to discover robust biological models.

## 🛠️ Installation

```bash
conda create --name CellScientist python=3.11.14
conda activate CellScientist
cd CellScientist
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f [https://download.pytorch.org/whl/cu118/torch_stable.html](https://download.pytorch.org/whl/cu118/torch_stable.html)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-2.0.1+cu118.html](https://data.pyg.org/whl/torch-2.0.1+cu118.html)
pip install -r requirements.txt

```

## 📂 Project Structure

```
CellScientist/                  <-- 项目根目录 (Project Root)
│
├── data/                       <-- 数据存放目录 (根据 H5 path fix 推断)
│   └── BBBC036/
│       └── BBBC036_smiles_split.h5
│
├── configs/                    <-- 配置文件目录 (根据 run_pipeline.py 推断)
│   ├── pipeline_config.json
│   ├── experiment_config.json
│   └── review_config.json
│
├── prompts/                    <-- Prompt YAML 存放目录
│   ├── pipeline_prompt.yaml
│   ├── review_optimize.yaml
│   ├── experiment_report.yaml
│   ├── final_report.yaml
│   ├── advanced_metrics.yaml
│   ├── autofix.yml
│   └── idea.yml
│
├── results/                    <-- 输出目录 (自动生成)
│
├── cellscientist/              <-- 主包 (Package Root)
│   ├── __init__.py
│   │
│   ├── core/                   <-- 核心逻辑层 (Phase 2 & 3 implementation)
│   │   ├── __init__.py
│   │   ├── config_loader.py    <-- 加载配置，解析变量
│   │   ├── llm_client.py       <-- TokenMeter, LLM 调用封装
│   │   ├── execution_workflow.py  <-- Phase 2 入口 (Design & Execute)
│   │   ├── review_workflow.py     <-- Phase 3 入口 (Review & Optimize)
│   │   ├── prompt_orchestrator.py <-- 协调生成、执行、分析
│   │   ├── prompt_generator.py    <-- 生成 Notebook 内容
│   │   ├── prompt_executor.py     <-- 包含 GraphExecutor 类
│   │   ├── executor_engine.py     <-- 纯执行引擎 (Review 阶段用)
│   │   ├── notebook_autofix.py    <-- 自动修复逻辑
│   │   ├── experiment_report.py   <-- 生成实验报告
│   │   ├── task_graph.py          <-- 任务图管理 (Review 阶段用)
│   │   └── external_knowledge_mirothink.py
│   │
│   └── pipeline/               <-- 管道编排层 (Runner logic)
│       ├── __init__.py
│       ├── run_pipeline.py     <-- 统一入口 (run_cellscientist.py 的重构版)
│       ├── config.py           <-- 管道配置合并逻辑
│       ├── metrics.py          <-- 指标提取与记分板 (Regex fix applied here)
│       ├── report.py           <-- 最终总结报告生成
│       ├── advanced_metrics.py <-- 高级指标分析
│       └── utils.py            <-- 路径查找、日志流式处理
│
└── run_cellscientist.py        <-- (可选) 根目录下的启动脚本，通常调用 cellscientist.pipeline.run_pipeline.main

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

CellScientist minimizes cost through a **Contextual Memory** mechanism that reduces token load by ~60% in later iterations.

* **Average Run (3-5 iterations):** $1.00 - $2.00 USD
* **Complex Run (10 iterations):** < $5.00 USD

## 🚀 Usage


```bash
python run_pipeline.py

```

