"""
Core profiler module for measuring AI inference performance.

This module provides the InferenceProfiler class that can profile any callable
function or model to measure throughput, latency, and other performance metrics.
"""

import time
import statistics
import threading
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


@dataclass
class ProfileResult:
    """Results from a profiling run"""
    total_samples: int
    total_time: float
    throughput: float  # samples per second
    avg_latency: float  # seconds per sample
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    latency_std: float
    individual_times: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.individual_times:
            self.min_latency = min(self.individual_times)
            self.max_latency = max(self.individual_times)
            self.avg_latency = statistics.mean(self.individual_times)
            self.median_latency = statistics.median(self.individual_times)
            self.latency_std = statistics.stdev(self.individual_times) if len(self.individual_times) > 1 else 0.0
            
            # Calculate percentiles
            sorted_times = sorted(self.individual_times)
            self.p95_latency = np.percentile(sorted_times, 95)
            self.p99_latency = np.percentile(sorted_times, 99)
            
            # Calculate throughput
            if self.total_time > 0:
                self.throughput = self.total_samples / self.total_time


class InferenceProfiler:
    """
    Profiler for AI inference workloads.
    
    This class provides methods to profile inference functions/models and measure
    key performance metrics including throughput and latency.
    """
    
    def __init__(self, warmup_runs: int = 10):
        """
        Initialize the profiler.
        
        Args:
            warmup_runs: Number of warmup runs before actual profiling
        """
        self.warmup_runs = warmup_runs
        
    def profile_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        num_runs: int = 100,
        concurrent_requests: int = 1,
        timeout: Optional[float] = None
    ) -> ProfileResult:
        """
        Profile a function's performance.
        
        Args:
            func: Function to profile
            args: Arguments to pass to function
            kwargs: Keyword arguments to pass to function
            num_runs: Number of times to run the function
            concurrent_requests: Number of concurrent requests (for load testing)
            timeout: Optional timeout for the entire profiling session
            
        Returns:
            ProfileResult containing performance metrics
        """
        if kwargs is None:
            kwargs = {}
            
        # Warmup runs
        print(f"Running {self.warmup_runs} warmup iterations...")
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Warning: Warmup run failed: {e}")
                
        print(f"Profiling {num_runs} runs with {concurrent_requests} concurrent requests...")
        
        individual_times = []
        start_time = time.perf_counter()
        
        if concurrent_requests == 1:
            # Sequential execution
            for i in range(num_runs):
                run_start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    run_end = time.perf_counter()
                    individual_times.append(run_end - run_start)
                except Exception as e:
                    print(f"Warning: Run {i+1} failed: {e}")
                    
                if timeout and (time.perf_counter() - start_time) > timeout:
                    print(f"Timeout reached after {len(individual_times)} runs")
                    break
        else:
            # Concurrent execution
            individual_times = self._profile_concurrent(
                func, args, kwargs, num_runs, concurrent_requests, timeout, start_time
            )
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return ProfileResult(
            total_samples=len(individual_times),
            total_time=total_time,
            throughput=0.0,  # Will be calculated in __post_init__
            avg_latency=0.0,  # Will be calculated in __post_init__
            median_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
            min_latency=0.0,
            max_latency=0.0,
            latency_std=0.0,
            individual_times=individual_times
        )
        
    def _profile_concurrent(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        num_runs: int,
        concurrent_requests: int,
        timeout: Optional[float],
        start_time: float
    ) -> List[float]:
        """Profile function with concurrent requests"""
        individual_times = []
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit all tasks
            futures = []
            for i in range(num_runs):
                future = executor.submit(self._timed_execution, func, args, kwargs)
                futures.append(future)
                
            # Collect results
            for future in as_completed(futures):
                if timeout and (time.perf_counter() - start_time) > timeout:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    print(f"Timeout reached after {len(individual_times)} runs")
                    break
                    
                try:
                    execution_time = future.result()
                    individual_times.append(execution_time)
                except Exception as e:
                    print(f"Warning: Concurrent run failed: {e}")
                    
        return individual_times
        
    def _timed_execution(self, func: Callable, args: tuple, kwargs: dict) -> float:
        """Execute function and return execution time"""
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        return end - start
        
    def profile_dataset(
        self,
        func: Callable,
        dataset: List[Any],
        batch_size: int = 1,
        concurrent_requests: int = 1
    ) -> ProfileResult:
        """
        Profile inference on a dataset.
        
        Args:
            func: Function to profile (should accept batch of data)
            dataset: List of input data samples
            batch_size: Number of samples per batch
            concurrent_requests: Number of concurrent requests
            
        Returns:
            ProfileResult containing performance metrics
        """
        print(f"Profiling dataset with {len(dataset)} samples, batch_size={batch_size}")
        
        # Create batches
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batches.append(batch)
            
        individual_times = []
        start_time = time.perf_counter()
        
        if concurrent_requests == 1:
            # Sequential processing
            for batch in batches:
                batch_start = time.perf_counter()
                try:
                    func(batch)
                    batch_end = time.perf_counter()
                    batch_time = batch_end - batch_start
                    # Calculate per-sample time
                    per_sample_time = batch_time / len(batch)
                    individual_times.extend([per_sample_time] * len(batch))
                except Exception as e:
                    print(f"Warning: Batch processing failed: {e}")
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(self._process_batch, func, batch)
                    futures.append(future)
                    
                for future in as_completed(futures):
                    try:
                        batch_times = future.result()
                        individual_times.extend(batch_times)
                    except Exception as e:
                        print(f"Warning: Concurrent batch processing failed: {e}")
                        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return ProfileResult(
            total_samples=len(individual_times),
            total_time=total_time,
            throughput=0.0,  # Will be calculated in __post_init__
            avg_latency=0.0,  # Will be calculated in __post_init__
            median_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
            min_latency=0.0,
            max_latency=0.0,
            latency_std=0.0,
            individual_times=individual_times
        )
        
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[float]:
        """Process a batch and return per-sample times"""
        batch_start = time.perf_counter()
        func(batch)
        batch_end = time.perf_counter()
        batch_time = batch_end - batch_start
        per_sample_time = batch_time / len(batch)
        return [per_sample_time] * len(batch)
        
    def compare_configurations(
        self,
        configs: Dict[str, Dict[str, Any]],
        func: Callable,
        base_args: tuple = (),
        base_kwargs: dict = None,
        num_runs: int = 50
    ) -> Dict[str, ProfileResult]:
        """
        Compare performance across different configurations.
        
        Args:
            configs: Dictionary mapping config names to config parameters
            func: Function to profile
            base_args: Base arguments for the function
            base_kwargs: Base keyword arguments
            num_runs: Number of runs per configuration
            
        Returns:
            Dictionary mapping config names to ProfileResult
        """
        if base_kwargs is None:
            base_kwargs = {}
            
        results = {}
        
        print(f"Comparing {len(configs)} configurations...")
        
        for config_name, config_params in configs.items():
            print(f"Testing configuration: {config_name}")
            
            # Merge config parameters with base kwargs
            kwargs = {**base_kwargs, **config_params}
            
            try:
                result = self.profile_function(
                    func=func,
                    args=base_args,
                    kwargs=kwargs,
                    num_runs=num_runs
                )
                results[config_name] = result
                print(f"  Throughput: {result.throughput:.2f} req/s, "
                      f"Avg Latency: {result.avg_latency*1000:.2f} ms")
            except Exception as e:
                print(f"  Configuration failed: {e}")
                
        return results