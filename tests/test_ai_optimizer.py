"""
Test suite for the AI Inference Performance Optimization Tool
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

# Import the modules we want to test
from ai_optimizer.profiler import InferenceProfiler, ProfileResult
from ai_optimizer.monitor import ResourceMonitor, ResourceSnapshot, ResourceStats
from ai_optimizer.analyzer import PerformanceAnalyzer
from ai_optimizer.optimizer import OptimizationRecommender, OptimizationCategory


class TestInferenceProfiler:
    """Test cases for the InferenceProfiler class"""
    
    def test_profile_simple_function(self):
        """Test profiling a simple function"""
        def simple_func():
            time.sleep(0.01)  # 10ms
            return "result"
        
        profiler = InferenceProfiler(warmup_runs=2)
        result = profiler.profile_function(simple_func, num_runs=5)
        
        assert result.total_samples == 5
        assert result.total_time > 0
        assert result.throughput > 0
        assert result.avg_latency > 0.008  # Should be around 10ms
        assert len(result.individual_times) == 5
        
    def test_profile_with_args(self):
        """Test profiling a function with arguments"""
        def func_with_args(x, y, multiply=1):
            time.sleep(0.005)
            return x + y * multiply
        
        profiler = InferenceProfiler(warmup_runs=1)
        result = profiler.profile_function(
            func_with_args, 
            args=(1, 2), 
            kwargs={"multiply": 2}, 
            num_runs=3
        )
        
        assert result.total_samples == 3
        assert result.avg_latency > 0.003
        
    def test_profile_concurrent(self):
        """Test profiling with concurrent requests"""
        def concurrent_func():
            time.sleep(0.01)
            return "concurrent_result"
        
        profiler = InferenceProfiler(warmup_runs=1)
        result = profiler.profile_function(
            concurrent_func, 
            num_runs=4, 
            concurrent_requests=2
        )
        
        assert result.total_samples == 4
        # With concurrency, total time should be less than sequential
        # But individual latencies should still be around 10ms
        assert result.avg_latency > 0.008
        
    def test_compare_configurations(self):
        """Test configuration comparison"""
        def configurable_func(delay=0.01):
            time.sleep(delay)
            return f"delay_{delay}"
        
        configs = {
            "fast": {"delay": 0.005},
            "slow": {"delay": 0.02}
        }
        
        profiler = InferenceProfiler(warmup_runs=1)
        results = profiler.compare_configurations(
            configs=configs,
            func=configurable_func,
            num_runs=3
        )
        
        assert len(results) == 2
        assert "fast" in results
        assert "slow" in results
        assert results["fast"].avg_latency < results["slow"].avg_latency


class TestResourceMonitor:
    """Test cases for the ResourceMonitor class"""
    
    def test_get_current_usage(self):
        """Test getting current resource usage"""
        monitor = ResourceMonitor(monitor_gpu=False)
        snapshot = monitor.get_current_usage()
        
        assert snapshot is not None
        assert snapshot.cpu_percent >= 0
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_used_mb > 0
        assert snapshot.memory_available_mb > 0
        assert snapshot.timestamp > 0
        
    def test_monitoring_context(self):
        """Test monitoring within a context"""
        from ai_optimizer.monitor import ContextResourceMonitor
        
        monitor = ResourceMonitor(sample_interval=0.05, monitor_gpu=False)
        
        with ContextResourceMonitor(monitor) as context:
            time.sleep(0.2)  # Monitor for 200ms
            
        stats = context.get_stats()
        assert stats is not None
        assert len(stats.snapshots) > 0
        assert stats.cpu_avg >= 0
        assert stats.memory_avg >= 0
        
    def test_detect_bottlenecks(self):
        """Test bottleneck detection"""
        monitor = ResourceMonitor()
        
        # Create mock stats with high CPU usage
        mock_stats = ResourceStats(
            cpu_avg=95.0, cpu_max=98.0, cpu_min=90.0,
            memory_avg=30.0, memory_max=35.0, memory_min=25.0,
            memory_used_avg_mb=4000, memory_used_max_mb=4500
        )
        
        bottlenecks = monitor.detect_bottlenecks(mock_stats)
        assert len(bottlenecks) > 0
        assert any("CPU" in b for b in bottlenecks)


class TestPerformanceAnalyzer:
    """Test cases for the PerformanceAnalyzer class"""
    
    def test_analyze_latency_distribution(self):
        """Test latency distribution analysis"""
        # Create mock ProfileResult
        latencies = [0.01, 0.012, 0.009, 0.011, 0.15, 0.01, 0.008, 0.013]  # One outlier
        result = ProfileResult(
            total_samples=len(latencies),
            total_time=sum(latencies),
            throughput=0, avg_latency=0, median_latency=0,
            p95_latency=0, p99_latency=0, min_latency=0,
            max_latency=0, latency_std=0,
            individual_times=latencies
        )
        
        analyzer = PerformanceAnalyzer()
        insights = analyzer.analyze_latency_distribution(result)
        
        assert insights.outlier_percentage > 0  # Should detect the 0.15s outlier
        assert "distribution" in insights.latency_distribution.lower()
        assert len(insights.recommendations) > 0
        assert insights.statistical_summary["mean_latency_ms"] > 0
        
    def test_compare_multiple_results(self):
        """Test comparing multiple profile results"""
        # Create two mock results
        result1 = ProfileResult(
            total_samples=10, total_time=1.0, throughput=10.0,
            avg_latency=0.1, median_latency=0.1, p95_latency=0.15,
            p99_latency=0.2, min_latency=0.08, max_latency=0.25,
            latency_std=0.02, individual_times=[0.1] * 10
        )
        
        result2 = ProfileResult(
            total_samples=10, total_time=2.0, throughput=5.0,
            avg_latency=0.2, median_latency=0.2, p95_latency=0.3,
            p99_latency=0.4, min_latency=0.15, max_latency=0.5,
            latency_std=0.05, individual_times=[0.2] * 10
        )
        
        results = {"fast_config": result1, "slow_config": result2}
        
        analyzer = PerformanceAnalyzer()
        df = analyzer.compare_multiple_results(results, metric="throughput")
        
        assert len(df) == 2
        assert "fast_config" in df["Configuration"].values
        assert "slow_config" in df["Configuration"].values
        assert df.iloc[0]["Throughput"] > df.iloc[1]["Throughput"]  # Sorted by throughput
        
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Create latencies with clear outliers
        normal_latencies = [0.01] * 95
        outliers = [0.5, 0.6, 0.7, 0.8, 0.9]  # Clear outliers
        all_latencies = normal_latencies + outliers
        
        result = ProfileResult(
            total_samples=len(all_latencies), total_time=sum(all_latencies),
            throughput=0, avg_latency=0, median_latency=0,
            p95_latency=0, p99_latency=0, min_latency=0,
            max_latency=0, latency_std=0,
            individual_times=all_latencies
        )
        
        analyzer = PerformanceAnalyzer()
        anomalous_values, anomalous_indices = analyzer.detect_performance_anomalies(result)
        
        assert len(anomalous_values) > 0
        assert len(anomalous_indices) > 0
        assert all(val > 0.3 for val in anomalous_values)  # Outliers should be > 0.3s


class TestOptimizationRecommender:
    """Test cases for the OptimizationRecommender class"""
    
    def test_generate_recommendations_low_throughput(self):
        """Test recommendations for low throughput scenario"""
        # Create result with low throughput
        result = ProfileResult(
            total_samples=10, total_time=10.0, throughput=5.0,  # Low throughput
            avg_latency=0.2, median_latency=0.2, p95_latency=0.3,
            p99_latency=0.4, min_latency=0.15, max_latency=0.5,
            latency_std=0.02, individual_times=[0.2] * 10
        )
        
        recommender = OptimizationRecommender()
        recommendations = recommender.generate_recommendations(result)
        
        assert len(recommendations) > 0
        # Should suggest batch size optimization for low throughput
        batch_recs = [r for r in recommendations if r.category == OptimizationCategory.BATCH_SIZE]
        assert len(batch_recs) > 0
        
    def test_generate_recommendations_with_targets(self):
        """Test recommendations when performance targets are provided"""
        result = ProfileResult(
            total_samples=10, total_time=1.0, throughput=10.0,
            avg_latency=0.1, median_latency=0.1, p95_latency=0.15,
            p99_latency=0.2, min_latency=0.08, max_latency=0.25,
            latency_std=0.02, individual_times=[0.1] * 10
        )
        
        targets = {"throughput": 50.0}  # Target much higher than current
        
        recommender = OptimizationRecommender()
        recommendations = recommender.generate_recommendations(result, target_metrics=targets)
        
        assert len(recommendations) > 0
        # Should suggest concurrency improvements to meet throughput target
        concurrency_recs = [r for r in recommendations if r.category == OptimizationCategory.CONCURRENCY]
        assert len(concurrency_recs) > 0
        
    def test_prioritize_recommendations(self):
        """Test recommendation prioritization"""
        result = ProfileResult(
            total_samples=10, total_time=1.0, throughput=10.0,
            avg_latency=0.1, median_latency=0.1, p95_latency=0.15,
            p99_latency=0.2, min_latency=0.08, max_latency=0.25,
            latency_std=0.02, individual_times=[0.1] * 10
        )
        
        recommender = OptimizationRecommender()
        recommendations = recommender.generate_recommendations(result)
        
        # Test prioritization with constraints
        prioritized = recommender.prioritize_recommendations(
            recommendations, 
            constraints={"prefer_low_effort": True}
        )
        
        assert len(prioritized) == len(recommendations)
        # Check that it's sorted by priority
        priorities = [r.priority for r in prioritized]
        assert priorities == sorted(priorities)
        
    def test_generate_implementation_plan(self):
        """Test implementation plan generation"""
        result = ProfileResult(
            total_samples=10, total_time=1.0, throughput=5.0,  # Low throughput
            avg_latency=0.2, median_latency=0.2, p95_latency=0.3,
            p99_latency=0.4, min_latency=0.15, max_latency=0.5,
            latency_std=0.1, individual_times=[0.2] * 10
        )
        
        recommender = OptimizationRecommender()
        recommendations = recommender.generate_recommendations(result)
        plan = recommender.generate_implementation_plan(recommendations)
        
        assert isinstance(plan, dict)
        assert "Phase 1" in str(plan) or "Low Effort" in str(plan)
        # Should have multiple phases
        assert len(plan) >= 1


if __name__ == "__main__":
    pytest.main([__file__])