#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test config merging to verify 3-tier inheritance architecture works correctly."""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cellscientist.pipeline.config import (
    load_pipeline_config,
    apply_pipeline_overrides,
)
from cellscientist.pipeline.utils import load_json, project_root


def test_experiment_config_merge():
    """Test that Experiment config properly inherits from pipeline config."""
    print("\n=== Testing Experiment Config Merge ===")
    
    # Load pipeline config
    pipe_cfg_path = os.path.join(project_root(), "configs", "pipeline_config.json")
    pipe_cfg = load_json(pipe_cfg_path)
    
    # Load experiment config
    exp_cfg_path = os.path.join(project_root(), "configs", "experiment_config.json")
    exp_cfg = load_json(exp_cfg_path)
    
    # Merge them
    merged = apply_pipeline_overrides("Experiment", exp_cfg, pipe_cfg)
    
    # Verify inherited values (from pipeline)
    assert merged["dataset_name"] == "BBBC036", "dataset_name should be inherited from pipeline"
    assert merged["split_name"] == "smiles", "split_name should be inherited from pipeline"
    assert merged["llm"]["api_key"] == "sk-00fKNXuFU5uFWIgC31666619E0044b78B64fB614904aCd9d", \
        "llm.api_key should be inherited from pipeline"
    assert merged["llm"]["base_url"] == "https://vip.yi-zhan.top/v1", \
        "llm.base_url should be inherited from pipeline"
    assert merged["literature"]["enabled"] == True, "literature.enabled should be inherited from pipeline"
    assert merged["literature"]["serper_api_key"] == "de9bb4416923dfd915bb4fb89d843cb0229e5973", \
        "literature.serper_api_key should be inherited from pipeline"
    assert merged["bio_kb"]["enabled"] == True, "bio_kb.enabled should be inherited from pipeline"
    assert merged["bio_kb"]["max_smiles"] == 25, "bio_kb.max_smiles should be inherited from pipeline"
    
    # Verify overridden values (from experiment)
    assert merged["llm"]["model"] == "gemini-3-pro-all", \
        "llm.model should be overridden by experiment config"
    assert merged["llm"]["temperature"] == 0.5, \
        "llm.temperature should be overridden by experiment config"
    assert merged["experiment"]["max_iterations"] == 3, \
        "experiment.max_iterations should be from experiment config"
    assert merged["exec"]["timeout_seconds"] == 360000, \
        "exec.timeout_seconds should be overridden by experiment config"
    assert merged["exec"]["cuda_device_id"] == 3, \
        "exec.cuda_device_id should be overridden by experiment config"
    
    # Verify merged paths
    assert "data_root" in merged["paths"], "paths.data_root should be inherited"
    assert merged["paths"]["data_root"] == "./data", "paths.data_root should have correct value"
    assert "design_execution_root" in merged["paths"], "paths.design_execution_root should be present"
    assert merged["paths"]["design_execution_root"] == "./results/${dataset_name}/generate_execution", \
        "paths.design_execution_root should be from experiment config"
    
    print("✅ Experiment config merge test passed!")
    return merged


def test_review_config_merge():
    """Test that Review config properly inherits from pipeline config."""
    print("\n=== Testing Review Config Merge ===")
    
    # Load pipeline config
    pipe_cfg_path = os.path.join(project_root(), "configs", "pipeline_config.json")
    pipe_cfg = load_json(pipe_cfg_path)
    
    # Load review config
    review_cfg_path = os.path.join(project_root(), "configs", "review_config.json")
    review_cfg = load_json(review_cfg_path)
    
    # Merge them
    merged = apply_pipeline_overrides("Review", review_cfg, pipe_cfg)
    
    # Verify inherited values (from pipeline)
    assert merged["dataset_name"] == "BBBC036", "dataset_name should be inherited from pipeline"
    assert merged["split_name"] == "smiles", "split_name should be inherited from pipeline"
    assert merged["llm"]["api_key"] == "sk-00fKNXuFU5uFWIgC31666619E0044b78B64fB614904aCd9d", \
        "llm.api_key should be inherited from pipeline"
    assert merged["llm"]["base_url"] == "https://vip.yi-zhan.top/v1", \
        "llm.base_url should be inherited from pipeline"
    assert merged["literature"]["enabled"] == True, "literature.enabled should be inherited from pipeline"
    assert merged["bio_kb"]["enabled"] == True, "bio_kb.enabled should be inherited from pipeline"
    
    # Verify overridden values (from review)
    assert merged["llm"]["model"] == "gemini-3-pro-preview-thinking", \
        "llm.model should be overridden by review config"
    assert merged["llm"]["temperature"] == 0.7, \
        "llm.temperature should be overridden by review config"
    assert merged["review"]["max_iterations"] == 5, \
        "review.max_iterations should be from review config"
    assert merged["exec"]["timeout_seconds"] == 18000, \
        "exec.timeout_seconds should be overridden by review config"
    
    # Verify merged paths
    assert "data_root" in merged["paths"], "paths.data_root should be inherited"
    assert merged["paths"]["data_root"] == "./data", "paths.data_root should have correct value"
    assert "review_feedback_root" in merged["paths"], "paths.review_feedback_root should be present"
    assert merged["paths"]["review_feedback_root"] == "./results/${dataset_name}/review_feedback", \
        "paths.review_feedback_root should be from review config"
    
    print("✅ Review config merge test passed!")
    return merged


def test_config_differences():
    """Test that stage configs have different overrides as expected."""
    print("\n=== Testing Config Differences ===")
    
    pipe_cfg_path = os.path.join(project_root(), "configs", "pipeline_config.json")
    pipe_cfg = load_json(pipe_cfg_path)
    
    exp_cfg_path = os.path.join(project_root(), "configs", "experiment_config.json")
    exp_cfg = load_json(exp_cfg_path)
    
    review_cfg_path = os.path.join(project_root(), "configs", "review_config.json")
    review_cfg = load_json(review_cfg_path)
    
    exp_merged = apply_pipeline_overrides("Experiment", exp_cfg, pipe_cfg)
    review_merged = apply_pipeline_overrides("Review", review_cfg, pipe_cfg)
    
    # Verify different models
    assert exp_merged["llm"]["model"] != review_merged["llm"]["model"], \
        "Experiment and Review should use different LLM models"
    assert exp_merged["llm"]["model"] == "gemini-3-pro-all", \
        "Experiment should use gemini-3-pro-all"
    assert review_merged["llm"]["model"] == "gemini-3-pro-preview-thinking", \
        "Review should use gemini-3-pro-preview-thinking"
    
    # Verify different timeouts
    assert exp_merged["exec"]["timeout_seconds"] != review_merged["exec"]["timeout_seconds"], \
        "Experiment and Review should have different timeouts"
    assert exp_merged["exec"]["timeout_seconds"] == 360000, "Experiment timeout should be 360000"
    assert review_merged["exec"]["timeout_seconds"] == 18000, "Review timeout should be 18000"
    
    # Verify unique sections
    assert "experiment" in exp_merged, "Experiment config should have 'experiment' section"
    assert "review" in review_merged, "Review config should have 'review' section"
    assert "experiment" not in review_merged, "Review config should not have 'experiment' section"
    assert "review" not in exp_merged, "Experiment config should not have 'review' section"
    
    print("✅ Config differences test passed!")


def print_summary(exp_merged, review_merged):
    """Print a summary showing the inheritance working."""
    print("\n" + "=" * 80)
    print("CONFIG MERGE VERIFICATION SUMMARY")
    print("=" * 80)
    
    print("\n📊 Size Comparison:")
    pipe_path = os.path.join(project_root(), "configs", "pipeline_config.json")
    exp_path = os.path.join(project_root(), "configs", "experiment_config.json")
    review_path = os.path.join(project_root(), "configs", "review_config.json")
    
    pipe_size = os.path.getsize(pipe_path)
    exp_size = os.path.getsize(exp_path)
    review_size = os.path.getsize(review_path)
    total_size = pipe_size + exp_size + review_size
    
    print(f"  pipeline_config.json:   {pipe_size:4d} bytes")
    print(f"  experiment_config.json: {exp_size:4d} bytes")
    print(f"  review_config.json:     {review_size:4d} bytes")
    print(f"  Total:                  {total_size:4d} bytes")
    
    print("\n✅ Shared Settings (Single Source of Truth):")
    print(f"  dataset_name: {exp_merged['dataset_name']}")
    print(f"  split_name: {exp_merged['split_name']}")
    print(f"  llm.api_key: {exp_merged['llm']['api_key'][:20]}...")
    print(f"  literature.enabled: {exp_merged['literature']['enabled']}")
    print(f"  bio_kb.enabled: {exp_merged['bio_kb']['enabled']}")
    
    print("\n🔀 Stage-Specific Overrides:")
    print(f"  Experiment LLM model: {exp_merged['llm']['model']}")
    print(f"  Review LLM model:     {review_merged['llm']['model']}")
    print(f"  Experiment timeout:   {exp_merged['exec']['timeout_seconds']} seconds")
    print(f"  Review timeout:       {review_merged['exec']['timeout_seconds']} seconds")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Config inheritance working correctly!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        # Run tests
        exp_merged = test_experiment_config_merge()
        review_merged = test_review_config_merge()
        test_config_differences()
        
        # Print summary
        print_summary(exp_merged, review_merged)
        
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
