#!/usr/bin/env python3
"""
Example inference function for demonstration purposes.

This module provides sample functions that can be profiled using the
AI Inference Performance Optimization Tool.
"""

import time
import random
import numpy as np
from typing import List, Any


def simple_inference(batch_size: int = 1) -> str:
    """
    Simple inference function that simulates model prediction.
    
    Args:
        batch_size: Number of samples to process
        
    Returns:
        String describing the processing result
    """
    # Simulate inference time (50ms base + 10ms per additional sample)
    inference_time = 0.05 + (batch_size - 1) * 0.01
    time.sleep(inference_time)
    
    return f"Processed batch of {batch_size} samples"


def variable_latency_inference(batch_size: int = 1) -> str:
    """
    Inference function with variable latency to demonstrate performance variability.
    
    Args:
        batch_size: Number of samples to process
        
    Returns:
        String describing the processing result
    """
    # Base time with random variation
    base_time = 0.03
    variation = random.uniform(0.01, 0.15)  # 10-150ms variation
    batch_overhead = batch_size * 0.005
    
    total_time = base_time + variation + batch_overhead
    time.sleep(total_time)
    
    return f"Variable latency processing of {batch_size} samples"


def cpu_intensive_inference(batch_size: int = 1) -> List[float]:
    """
    CPU-intensive inference function that performs actual computation.
    
    Args:
        batch_size: Number of samples to process
        
    Returns:
        List of computed results
    """
    results = []
    
    for i in range(batch_size):
        # Simulate CPU-intensive computation
        # Matrix multiplication to consume CPU cycles
        size = 100 + i * 10  # Increasing complexity per sample
        matrix_a = np.random.random((size, size))
        matrix_b = np.random.random((size, size))
        
        # Perform computation
        result = np.dot(matrix_a, matrix_b)
        results.append(float(np.mean(result)))
    
    return results


def memory_intensive_inference(batch_size: int = 1) -> List[float]:
    """
    Memory-intensive inference function that allocates large amounts of memory.
    
    Args:
        batch_size: Number of samples to process
        
    Returns:
        List of computed results
    """
    results = []
    
    # Allocate memory proportional to batch size
    memory_per_sample = 10_000_000  # 10MB per sample
    
    for i in range(batch_size):
        # Allocate large array
        large_array = np.random.random(memory_per_sample // 8)  # 8 bytes per float64
        
        # Perform some computation
        result = np.sum(large_array) / len(large_array)
        results.append(result)
        
        # Clean up to avoid accumulation
        del large_array
    
    return results


def batch_optimized_inference(batch: List[Any]) -> List[str]:
    """
    Batch-optimized inference function that processes multiple samples efficiently.
    
    Args:
        batch: List of input samples to process
        
    Returns:
        List of processing results
    """
    batch_size = len(batch)
    
    # Efficient batch processing - fixed overhead + per-sample time
    fixed_overhead = 0.02  # 20ms fixed overhead
    per_sample_time = 0.005  # 5ms per sample
    
    total_time = fixed_overhead + (batch_size * per_sample_time)
    time.sleep(total_time)
    
    # Return results for each sample
    return [f"Batch processed sample {i}" for i in range(batch_size)]


def inference_with_occasional_delay(batch_size: int = 1) -> str:
    """
    Inference function that occasionally has high latency (simulates system issues).
    
    Args:
        batch_size: Number of samples to process
        
    Returns:
        String describing the processing result
    """
    # 5% chance of a slow request (simulating system issues)
    if random.random() < 0.05:
        # Slow request - 500ms to 2s delay
        delay = random.uniform(0.5, 2.0)
        time.sleep(delay)
        return f"SLOW: Processed {batch_size} samples with {delay:.2f}s delay"
    else:
        # Normal fast request
        normal_time = 0.03 + batch_size * 0.005
        time.sleep(normal_time)
        return f"FAST: Processed {batch_size} samples normally"


# Default function for easy testing
inference = simple_inference