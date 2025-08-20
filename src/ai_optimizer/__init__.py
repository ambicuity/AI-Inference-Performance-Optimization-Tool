"""
AI Inference Performance Optimization Tool

A comprehensive Python-based tool for profiling and optimizing AI inference workloads.
Measures key performance metrics like throughput and latency, identifies resource 
bottlenecks, and provides optimization recommendations.
"""

__version__ = "1.0.0"
__author__ = "AI Performance Team"
__email__ = "team@example.com"

from .profiler import InferenceProfiler
from .monitor import ResourceMonitor
from .analyzer import PerformanceAnalyzer
from .optimizer import OptimizationRecommender

__all__ = [
    "InferenceProfiler",
    "ResourceMonitor", 
    "PerformanceAnalyzer",
    "OptimizationRecommender",
]