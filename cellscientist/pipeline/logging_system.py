#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-Tier Logging System for CellScientist Pipeline

Tier 1: Console Output (User-Facing) - Clean, hierarchical progress display
Tier 2: Stage Logs (Detailed Stage Execution) - experiment.log, review.log
Tier 3: Full Execution Log (Complete Trace) - execution_detail.log
Tier 4: Evidence Chain (Structured Audit Trail) - evidence_chain.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


class TieredLogger:
    """
    Comprehensive 4-tier logging system that captures all execution evidence.
    
    Tier 1: Console - Clean hierarchical output with visual indicators
    Tier 2: Stage Logs - Detailed per-stage execution logs
    Tier 3: Full Log - Complete stdout/stderr capture
    Tier 4: Evidence Chain - Structured JSON audit trail
    """
    
    def __init__(self, run_dir: Path, config: dict, dataset_name: str = ""):
        """Initialize all 4 logging tiers.
        
        Args:
            run_dir: Directory where logs will be stored
            config: Pipeline configuration dictionary
            dataset_name: Name of the dataset being processed
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.config = config
        # Use consistent ISO format for all timestamps
        self.pipeline_id = f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        
        # Tier 1: Console (direct print + console_output.log)
        self.console_enabled = True
        self.console_log_path = self.run_dir / "console_output.log"
        self.console_log_fp: Optional[TextIO] = None
        self._setup_console_log()
        
        # Tier 2: Stage loggers
        self.stage_loggers: Dict[str, logging.Logger] = {}
        self._setup_stage_loggers()
        
        # Tier 3: Full execution logger
        self.full_logger = self._setup_full_logger()
        
        # Tier 4: Evidence chain
        self.evidence_chain: Dict[str, Any] = {
            "pipeline_id": self.pipeline_id,
            "dataset": dataset_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": self._sanitize_config(config),
            "experiment_stage": {
                "iterations": []
            },
            "review_stage": {
                "iterations": []
            }
        }
        
        # Capture state
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._capture_buffer: Optional[StringIO] = None
        self._tee_stdout: Optional[TextIO] = None
        self._tee_stderr: Optional[TextIO] = None
        
    def _setup_console_log(self) -> None:
        """Setup Tier 1 console output log file."""
        try:
            self.console_log_fp = open(self.console_log_path, 'w', encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not open console_output.log: {e}")
            self.console_log_fp = None
    
    def _sanitize_config(self, config: dict) -> dict:
        """Remove sensitive data from config before logging."""
        import copy
        safe = copy.deepcopy(config)
        # Remove API keys
        if isinstance(safe.get("llm"), dict):
            if "api_key" in safe["llm"]:
                safe["llm"]["api_key"] = "***REDACTED***"
        return safe
    
    def _setup_stage_loggers(self) -> None:
        """Setup Tier 2: Stage-specific loggers."""
        for stage in ["experiment", "review"]:
            logger = logging.getLogger(f"cellscientist.{stage}")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # File handler
            log_path = self.run_dir / f"{stage}.log"
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            self.stage_loggers[stage] = logger
    
    def _setup_full_logger(self) -> logging.Logger:
        """Setup Tier 3: Full execution trace logger."""
        logger = logging.getLogger("cellscientist.full_execution")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler for complete trace
        log_path = self.run_dir / "execution_detail.log"
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def console_info(self, message: str, level: int = 0, symbol: str = ""):
        """Tier 1: Print to console with hierarchical formatting.
        
        Args:
            message: Message to display
            level: Indentation level (0, 1, 2, ...)
            symbol: Optional symbol/emoji to prefix
        """
        if not self.console_enabled:
            return
        
        # Hierarchical tree structure
        indent = ""
        if level == 1:
            indent = "├─ "
        elif level == 2:
            indent = "│  ├─ "
        elif level == 3:
            indent = "│  │  ├─ "
        elif level > 3:
            indent = "│  " * (level - 1) + "├─ "
        
        prefix = f"{indent}{symbol} " if symbol else indent
        output = f"{prefix}{message}"
        
        # Print to console
        print(output, flush=True)
        
        # Write to console_output.log
        if self.console_log_fp:
            try:
                self.console_log_fp.write(output + "\n")
                self.console_log_fp.flush()
            except Exception:
                pass
        
        # Also log to Tier 3
        self.full_log(f"[CONSOLE] {output}")
    
    def stage_log(self, stage: str, message: str, level: str = "info"):
        """Tier 2: Write to stage-specific log.
        
        Args:
            stage: Stage name ("experiment" or "review")
            message: Message to log
            level: Log level (debug, info, warning, error)
        """
        if stage not in self.stage_loggers:
            return
        
        logger = self.stage_loggers[stage]
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
        
        # Also log to Tier 3
        self.full_log(f"[{stage.upper()}] {message}")
    
    def full_log(self, message: str):
        """Tier 3: Write to complete execution log.
        
        Args:
            message: Message to log
        """
        self.full_logger.debug(message)
    
    def add_evidence(
        self,
        stage: str,
        iteration: int,
        evidence_type: str,
        data: Dict[str, Any]
    ):
        """Tier 4: Add to structured evidence chain.
        
        Args:
            stage: Stage name ("experiment" or "review")
            evidence_type: Type of evidence (e.g., "strategy_generation", "execution_trace")
            iteration: Iteration number
            data: Evidence data dictionary
        """
        stage_key = f"{stage}_stage"
        if stage_key not in self.evidence_chain:
            self.evidence_chain[stage_key] = {"iterations": []}
        
        # Find or create iteration record
        iterations = self.evidence_chain[stage_key]["iterations"]
        iter_record = None
        for rec in iterations:
            if rec.get("iteration") == iteration:
                iter_record = rec
                break
        
        if iter_record is None:
            iter_record = {
                "iteration": iteration,
                "timestamp_start": datetime.now(timezone.utc).isoformat()
            }
            iterations.append(iter_record)
        
        # Add evidence data
        iter_record[evidence_type] = data
        
        # Update timestamp
        iter_record["timestamp_last_update"] = datetime.now(timezone.utc).isoformat()
    
    @contextmanager
    def capture_stdout(self):
        """Context manager to capture all stdout/stderr to Tier 3.
        
        Usage:
            with logger.capture_stdout():
                # All prints here are captured
                print("This goes to full log")
        """
        class TeeWriter:
            """Writer that duplicates output to both original stream and logger."""
            def __init__(self, original_stream, logger_func, prefix=""):
                self.original = original_stream
                self.logger = logger_func
                self.prefix = prefix
                self.buffer = ""
            
            def write(self, data):
                # Write to original
                self.original.write(data)
                self.original.flush()
                
                # Accumulate for logging (to avoid partial lines)
                self.buffer += data
                
                # Log complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line:  # Don't log empty lines
                        self.logger(f"{self.prefix}{line}")
            
            def flush(self):
                if self.buffer:
                    self.logger(f"{self.prefix}{self.buffer}")
                    self.buffer = ""
                self.original.flush()
            
            def isatty(self):
                return self.original.isatty()
        
        # Create tee writers
        tee_stdout = TeeWriter(sys.stdout, lambda msg: self.full_log(f"[STDOUT] {msg}"))
        tee_stderr = TeeWriter(sys.stderr, lambda msg: self.full_log(f"[STDERR] {msg}"))
        
        # Replace streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        try:
            yield self
        finally:
            # Flush any remaining buffer
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            if hasattr(sys.stderr, 'flush'):
                sys.stderr.flush()
            
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def save_evidence_chain(self):
        """Save Tier 4 evidence chain to JSON file."""
        # Add end time
        self.evidence_chain["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(self.evidence_chain["start_time"])
            end = datetime.fromisoformat(self.evidence_chain["end_time"])
            duration = (end - start).total_seconds()
            self.evidence_chain["total_duration_seconds"] = duration
        except Exception as e:
            # Log parsing failure for debugging
            self.full_log(f"Warning: Failed to calculate duration - {e}")
        
        # Save to file
        evidence_path = self.run_dir / "evidence_chain.json"
        with open(evidence_path, 'w', encoding='utf-8') as f:
            json.dump(self.evidence_chain, f, indent=2, ensure_ascii=False)
        
        self.console_info(f"📊 Evidence chain saved to: {evidence_path}", level=0, symbol="")
        self.full_log(f"Evidence chain saved: {evidence_path}")
    
    def print_stage_header(self, stage_name: str):
        """Print a visual header for a stage."""
        self.console_info("", level=0)
        self.console_info(f"🔄 {stage_name.upper()} STAGE", level=0)
    
    def print_iteration_header(self, iteration: int, max_iterations: int):
        """Print iteration progress header."""
        self.console_info(f"🔎 Iteration {iteration}/{max_iterations}", level=1)
    
    def print_subsection(self, title: str, level: int = 2, symbol: str = ""):
        """Print a subsection title."""
        self.console_info(title, level=level, symbol=symbol)
    
    def print_metric(self, name: str, value: Any, level: int = 2):
        """Print a metric value."""
        self.console_info(f"{name}: {value}", level=level, symbol="📊")
    
    def print_success(self, message: str, level: int = 2):
        """Print a success message."""
        self.console_info(message, level=level, symbol="✅")
    
    def print_error(self, message: str, level: int = 2):
        """Print an error message."""
        self.console_info(message, level=level, symbol="❌")
    
    def print_warning(self, message: str, level: int = 2):
        """Print a warning message."""
        self.console_info(message, level=level, symbol="⚠️")
    
    def print_timing(self, message: str):
        """Print timing information."""
        self.console_info(message, level=0, symbol="⏱️")
    
    def finalize(self):
        """Finalize all logging tiers."""
        # Save evidence chain
        self.save_evidence_chain()
        
        # Close console log
        if self.console_log_fp:
            try:
                self.console_log_fp.close()
            except Exception:
                pass
        
        # Close stage loggers
        for logger in self.stage_loggers.values():
            for handler in logger.handlers:
                handler.close()
        
        # Close full logger
        for handler in self.full_logger.handlers:
            handler.close()
        
        self.full_log("=" * 80)
        self.full_log("PIPELINE EXECUTION COMPLETED")
        self.full_log("=" * 80)


def create_tiered_logger(run_dir: str, config: dict, dataset_name: str = "") -> TieredLogger:
    """Factory function to create a TieredLogger instance.
    
    Args:
        run_dir: Directory where logs will be stored
        config: Pipeline configuration dictionary
        dataset_name: Name of the dataset being processed
    
    Returns:
        Initialized TieredLogger instance
    """
    return TieredLogger(Path(run_dir), config, dataset_name)
