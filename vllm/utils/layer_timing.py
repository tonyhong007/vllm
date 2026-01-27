# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility module for measuring latency of attention and FFN computations.

This module provides a simple timing manager to accumulate attention and FFN
time across all transformer layer forward passes.

Uses file-based IPC to share timing data between vLLM worker processes
and the main process.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

# Global file path for cross-process communication
_TIMING_FILE = Path(tempfile.gettempdir()) / "vllm_layer_timing.json"
_ENABLED_FLAG_FILE = Path(tempfile.gettempdir()) / "vllm_layer_timing_enabled"


def _is_timing_enabled() -> bool:
    """Check if timing is enabled (via flag file)."""
    return _ENABLED_FLAG_FILE.exists()


def _write_timing_data(attention_ms: float, ffn_ms: float, total_layer_ms: float):
    """Append timing data to the shared file (called from worker process)."""
    try:
        # Read existing data
        data = {"attention_ms": 0.0, "ffn_ms": 0.0, "total_layer_ms": 0.0, "count": 0}
        if _TIMING_FILE.exists():
            try:
                with open(_TIMING_FILE, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Accumulate
        data["attention_ms"] += attention_ms
        data["ffn_ms"] += ffn_ms
        data["total_layer_ms"] += total_layer_ms
        data["count"] += 1
        
        # Write back
        with open(_TIMING_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass  # Silently ignore errors to not affect inference


def _read_timing_data() -> dict:
    """Read timing data from the shared file (called from main process)."""
    data = {"attention_ms": 0.0, "ffn_ms": 0.0, "total_layer_ms": 0.0, "count": 0}
    try:
        if _TIMING_FILE.exists():
            with open(_TIMING_FILE, "r") as f:
                data = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return data


def _clear_timing_data():
    """Clear the timing data file."""
    try:
        if _TIMING_FILE.exists():
            os.remove(_TIMING_FILE)
    except OSError:
        pass


class LayerTimingManager:
    """
    Manager for measuring aggregate attention and FFN timing across all layers.
    
    Uses file-based IPC to share data between vLLM worker processes and main process.
    
    Usage:
        timing_manager = LayerTimingManager.get_instance()
        timing_manager.enable()
        
        # Run inference
        ...
        
        # Get results
        results = timing_manager.get_aggregated_summary()
        timing_manager.print_summary()
    """
    
    _instance: Optional["LayerTimingManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "LayerTimingManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def enable(self, sync_after_each: bool = True):
        """Enable timing measurements."""
        # Create flag file to signal worker processes
        try:
            _ENABLED_FLAG_FILE.touch()
        except OSError:
            pass
    
    def disable(self):
        """Disable timing measurements."""
        # Remove flag file
        try:
            if _ENABLED_FLAG_FILE.exists():
                os.remove(_ENABLED_FLAG_FILE)
        except OSError:
            pass
    
    @property
    def is_enabled(self) -> bool:
        """Check if timing is enabled (checks flag file for cross-process compat)."""
        return _is_timing_enabled()
    
    def record_timing(
        self,
        layer_idx: int,
        attention_ms: float,
        ffn_ms: float,
        input_layernorm_ms: float = 0.0,
        post_attn_layernorm_ms: float = 0.0,
        total_layer_ms: float = 0.0,
    ):
        """Record timing - writes to shared file for cross-process communication."""
        if not _is_timing_enabled():
            return
        _write_timing_data(attention_ms, ffn_ms, total_layer_ms)
    
    def get_aggregated_summary(self) -> dict:
        """Get aggregated summary of all timing (reads from shared file)."""
        data = _read_timing_data()
        count = data.get("count", 0)
        attn_ms = data.get("attention_ms", 0.0)
        ffn_ms = data.get("ffn_ms", 0.0)
        total_ms = data.get("total_layer_ms", 0.0)
        
        return {
            "attention": {
                "name": "Attention",
                "total_ms": attn_ms,
                "mean_ms": attn_ms / count if count > 0 else 0,
                "count": count,
            },
            "ffn": {
                "name": "FFN",
                "total_ms": ffn_ms,
                "mean_ms": ffn_ms / count if count > 0 else 0,
                "count": count,
            },
            "total_layer": {
                "name": "Total Layer",
                "total_ms": total_ms,
                "mean_ms": total_ms / count if count > 0 else 0,
                "count": count,
            },
            "num_layers": count,
        }
    
    def print_summary(self):
        """Print a formatted summary of timing results."""
        summary = self.get_aggregated_summary()
        
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        print(f"Total layer forward calls: {summary['num_layers']}")
        print("-" * 70)
        
        attn = summary["attention"]
        ffn = summary["ffn"]
        total = summary["total_layer"]
        
        if total["total_ms"] > 0:
            attn_pct = attn["total_ms"] / total["total_ms"] * 100
            ffn_pct = ffn["total_ms"] / total["total_ms"] * 100
            
            print(f"\nAttention: {attn['total_ms']:.2f} ms ({attn_pct:.1f}%)")
            print(f"FFN:       {ffn['total_ms']:.2f} ms ({ffn_pct:.1f}%)")
            print(f"Total:     {total['total_ms']:.2f} ms")
            
            if ffn["total_ms"] > 0:
                ratio = attn["total_ms"] / ffn["total_ms"]
                print(f"\nAttention/FFN ratio: {ratio:.2f}x")
        
        print("=" * 70 + "\n")
    
    def clear(self):
        """Clear all timing data."""
        _clear_timing_data()
    
    def reset(self):
        """Reset the manager (clear data and disable)."""
        self.clear()
        self.disable()
