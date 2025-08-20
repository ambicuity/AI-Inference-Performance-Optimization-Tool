"""
Optimization recommender module for providing AI inference optimization suggestions.

This module provides the OptimizationRecommender class that analyzes performance
data and system characteristics to provide actionable optimization recommendations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .profiler import ProfileResult
from .monitor import ResourceStats
from .analyzer import PerformanceInsights


class OptimizationCategory(Enum):
    """Categories of optimization recommendations"""
    BATCH_SIZE = "batch_size"
    CONCURRENCY = "concurrency" 
    RESOURCE = "resource"
    MODEL = "model"
    SYSTEM = "system"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class OptimizationRecommendation:
    """Individual optimization recommendation"""
    category: OptimizationCategory
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # "Low", "Medium", "High"
    priority: int  # 1 (highest) to 5 (lowest)
    specific_actions: List[str]


class OptimizationRecommender:
    """
    Generate optimization recommendations based on performance analysis.
    
    This class analyzes profiling results, resource usage, and system characteristics
    to provide actionable recommendations for optimizing AI inference performance.
    """
    
    def __init__(self):
        """Initialize the optimization recommender"""
        self.recommendation_history: List[OptimizationRecommendation] = []
        
    def generate_recommendations(
        self,
        profile_result: ProfileResult,
        resource_stats: Optional[ResourceStats] = None,
        system_info: Optional[Dict[str, Any]] = None,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Generate comprehensive optimization recommendations.
        
        Args:
            profile_result: Performance profiling results
            resource_stats: Optional resource utilization statistics
            system_info: Optional system information (CPU cores, memory, GPU, etc.)
            target_metrics: Optional target performance metrics to achieve
            
        Returns:
            List of OptimizationRecommendation objects sorted by priority
        """
        recommendations = []
        
        # Analyze performance metrics
        perf_recs = self._analyze_performance_metrics(profile_result, target_metrics)
        recommendations.extend(perf_recs)
        
        # Analyze resource utilization
        if resource_stats:
            resource_recs = self._analyze_resource_utilization(resource_stats, profile_result)
            recommendations.extend(resource_recs)
            
        # Analyze system configuration
        if system_info:
            system_recs = self._analyze_system_configuration(system_info, profile_result)
            recommendations.extend(system_recs)
            
        # Analyze latency patterns
        latency_recs = self._analyze_latency_patterns(profile_result)
        recommendations.extend(latency_recs)
        
        # Sort by priority and remove duplicates
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations.sort(key=lambda x: x.priority)
        
        # Store in history
        self.recommendation_history.extend(recommendations)
        
        return recommendations
        
    def _analyze_performance_metrics(
        self, 
        result: ProfileResult, 
        targets: Optional[Dict[str, float]]
    ) -> List[OptimizationRecommendation]:
        """Analyze core performance metrics and generate recommendations"""
        recommendations = []
        
        # Throughput analysis
        if result.throughput < 10:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.BATCH_SIZE,
                title="Increase Batch Size for Higher Throughput",
                description=f"Current throughput is {result.throughput:.2f} req/s, which is relatively low. "
                           "Increasing batch size can significantly improve throughput by better utilizing compute resources.",
                expected_improvement="20-100% throughput increase",
                implementation_effort="Low",
                priority=1,
                specific_actions=[
                    "Test with 2x, 4x, and 8x current batch size",
                    "Monitor memory usage when increasing batch size",
                    "Find optimal batch size that balances throughput and latency"
                ]
            ))
            
        # Latency analysis
        if result.avg_latency > 0.1:  # 100ms
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.MODEL,
                title="Optimize Model for Lower Latency",
                description=f"Average latency is {result.avg_latency*1000:.2f}ms. "
                           "Consider model optimization techniques to reduce inference time.",
                expected_improvement="10-50% latency reduction",
                implementation_effort="Medium",
                priority=2,
                specific_actions=[
                    "Apply model quantization (INT8/FP16)",
                    "Use model pruning to reduce model size",
                    "Consider knowledge distillation for smaller models",
                    "Optimize model architecture for target hardware"
                ]
            ))
            
        # Latency variability
        cv = result.latency_std / result.avg_latency if result.avg_latency > 0 else 0
        if cv > 0.3:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.SYSTEM,
                title="Reduce Latency Variability",
                description=f"High latency variability detected (CV: {cv:.3f}). "
                           "Inconsistent performance can impact user experience.",
                expected_improvement="More predictable performance",
                implementation_effort="Medium",
                priority=3,
                specific_actions=[
                    "Implement request queuing and load balancing",
                    "Optimize memory allocation patterns",
                    "Consider CPU/GPU frequency scaling settings",
                    "Implement connection pooling for data sources"
                ]
            ))
            
        # Target-based recommendations
        if targets:
            if "throughput" in targets and result.throughput < targets["throughput"]:
                gap = targets["throughput"] - result.throughput
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.CONCURRENCY,
                    title="Scale Concurrency to Meet Throughput Target",
                    description=f"Current throughput ({result.throughput:.2f} req/s) is "
                               f"{gap:.2f} req/s below target ({targets['throughput']:.2f} req/s).",
                    expected_improvement=f"Achieve target throughput of {targets['throughput']:.2f} req/s",
                    implementation_effort="Low",
                    priority=1,
                    specific_actions=[
                        "Increase number of worker processes/threads",
                        "Implement horizontal scaling with multiple instances",
                        "Optimize request routing and load balancing"
                    ]
                ))
                
        return recommendations
        
    def _analyze_resource_utilization(
        self, 
        stats: ResourceStats, 
        result: ProfileResult
    ) -> List[OptimizationRecommendation]:
        """Analyze resource utilization patterns"""
        recommendations = []
        
        # CPU analysis
        if stats.cpu_avg > 90:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.RESOURCE,
                title="Address High CPU Utilization",
                description=f"CPU usage is very high ({stats.cpu_avg:.1f}% average, {stats.cpu_max:.1f}% peak). "
                           "This indicates a CPU bottleneck that limits scalability.",
                expected_improvement="Improved scalability and reduced queuing",
                implementation_effort="Medium",
                priority=1,
                specific_actions=[
                    "Scale to instances with more CPU cores",
                    "Optimize CPU-intensive operations",
                    "Consider CPU-specific optimizations (SIMD, vectorization)",
                    "Implement request throttling to prevent overload"
                ]
            ))
        elif stats.cpu_avg < 30:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.CONCURRENCY,
                title="Increase Load to Better Utilize CPU",
                description=f"CPU is underutilized ({stats.cpu_avg:.1f}% average). "
                           "You can likely handle more concurrent requests or larger batch sizes.",
                expected_improvement="Higher throughput without additional resources",
                implementation_effort="Low",
                priority=3,
                specific_actions=[
                    "Increase concurrent request limit",
                    "Process larger batches to improve CPU utilization",
                    "Consider multi-threading for parallel processing"
                ]
            ))
            
        # Memory analysis
        if stats.memory_avg > 85:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.RESOURCE,
                title="Address High Memory Usage",
                description=f"Memory usage is high ({stats.memory_avg:.1f}% average). "
                           "This may cause performance degradation or system instability.",
                expected_improvement="More stable performance and scalability headroom",
                implementation_effort="Medium",
                priority=2,
                specific_actions=[
                    "Reduce batch size to lower memory requirements",
                    "Implement memory pooling and reuse strategies",
                    "Optimize data structures and memory layout",
                    "Consider memory-efficient model architectures"
                ]
            ))
            
        # GPU analysis
        if stats.gpu_util_avg is not None:
            if stats.gpu_util_avg > 90:
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.RESOURCE,
                    title="GPU is Performance Bottleneck",
                    description=f"GPU utilization is very high ({stats.gpu_util_avg:.1f}% average). "
                               "Performance is likely GPU-bound.",
                    expected_improvement="Identify if additional GPU resources are needed",
                    implementation_effort="High",
                    priority=1,
                    specific_actions=[
                        "Consider scaling to more powerful GPU",
                        "Implement multi-GPU processing if available",
                        "Optimize GPU memory usage and data transfers",
                        "Use mixed precision training/inference"
                    ]
                ))
            elif stats.gpu_util_avg < 40:
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.BATCH_SIZE,
                    title="Increase GPU Utilization",
                    description=f"GPU is underutilized ({stats.gpu_util_avg:.1f}% average). "
                               "You can likely achieve higher performance with current hardware.",
                    expected_improvement="Better GPU ROI and higher throughput",
                    implementation_effort="Low",
                    priority=2,
                    specific_actions=[
                        "Increase batch size to better utilize GPU",
                        "Verify operations are running on GPU, not CPU",
                        "Optimize GPU memory usage for larger batches",
                        "Consider processing multiple requests in parallel"
                    ]
                ))
                
        return recommendations
        
    def _analyze_system_configuration(
        self, 
        system_info: Dict[str, Any], 
        result: ProfileResult
    ) -> List[OptimizationRecommendation]:
        """Analyze system configuration for optimization opportunities"""
        recommendations = []
        
        # CPU core analysis
        if "cpu_cores" in system_info:
            cores = system_info["cpu_cores"]
            if cores < 4 and result.throughput < 50:
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.INFRASTRUCTURE,
                    title="Scale to More CPU Cores",
                    description=f"System has {cores} CPU cores. For AI inference workloads, "
                               "more cores typically provide better parallel processing capability.",
                    expected_improvement="Improved parallel processing and throughput",
                    implementation_effort="High",
                    priority=3,
                    specific_actions=[
                        "Migrate to instance types with more CPU cores",
                        "Implement multi-process inference serving",
                        "Optimize thread pool sizes for available cores"
                    ]
                ))
                
        # Memory analysis
        if "memory_gb" in system_info:
            memory_gb = system_info["memory_gb"]
            if memory_gb < 8:
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.INFRASTRUCTURE,
                    title="Consider More System Memory",
                    description=f"System has {memory_gb}GB RAM. AI inference workloads often "
                               "benefit from more memory for larger batches and model caching.",
                    expected_improvement="Support for larger batches and better caching",
                    implementation_effort="High",
                    priority=4,
                    specific_actions=[
                        "Scale to instances with more memory",
                        "Implement model weight sharing across processes",
                        "Optimize memory usage patterns"
                    ]
                ))
                
        return recommendations
        
    def _analyze_latency_patterns(self, result: ProfileResult) -> List[OptimizationRecommendation]:
        """Analyze latency patterns for optimization opportunities"""
        recommendations = []
        
        # Tail latency analysis
        p99_to_median_ratio = result.p99_latency / result.median_latency if result.median_latency > 0 else 0
        if p99_to_median_ratio > 3:
            recommendations.append(OptimizationRecommendation(
                category=OptimizationCategory.SYSTEM,
                title="Address Tail Latency Issues",
                description=f"P99 latency ({result.p99_latency*1000:.2f}ms) is {p99_to_median_ratio:.1f}x "
                           f"higher than median ({result.median_latency*1000:.2f}ms). "
                           "This indicates occasional very slow requests.",
                expected_improvement="More consistent user experience",
                implementation_effort="Medium",
                priority=2,
                specific_actions=[
                    "Implement request timeout and circuit breakers",
                    "Add performance monitoring and alerting",
                    "Investigate system-level bottlenecks",
                    "Consider request retry with exponential backoff"
                ]
            ))
            
        # Warm-up effects
        latencies = np.array(result.individual_times)
        if len(latencies) > 20:
            first_20 = np.mean(latencies[:20])
            last_20 = np.mean(latencies[-20:])
            if first_20 > last_20 * 1.5:
                recommendations.append(OptimizationRecommendation(
                    category=OptimizationCategory.SYSTEM,
                    title="Implement Proper Model Warm-up",
                    description="Initial requests are significantly slower, indicating cold start effects. "
                               "Implementing proper warm-up can improve initial response times.",
                    expected_improvement="Reduced cold start latency",
                    implementation_effort="Low",
                    priority=3,
                    specific_actions=[
                        "Implement model pre-loading and warm-up",
                        "Keep models in memory between requests",
                        "Use connection keep-alive for data sources",
                        "Consider model serving frameworks with warm-up"
                    ]
                ))
                
        return recommendations
        
    def _deduplicate_recommendations(
        self, 
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Remove duplicate recommendations based on title"""
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)
                
        return unique_recommendations
        
    def prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[OptimizationRecommendation]:
        """
        Re-prioritize recommendations based on constraints and business requirements.
        
        Args:
            recommendations: List of recommendations to prioritize
            constraints: Dictionary of constraints (budget, timeline, resources)
            
        Returns:
            Re-prioritized list of recommendations
        """
        if not constraints:
            return recommendations
            
        # Adjust priorities based on constraints
        for rec in recommendations:
            if constraints.get("prefer_low_effort", False):
                if rec.implementation_effort == "Low":
                    rec.priority = max(1, rec.priority - 1)
                elif rec.implementation_effort == "High":
                    rec.priority = min(5, rec.priority + 1)
                    
            if constraints.get("budget_limited", False):
                if rec.category == OptimizationCategory.INFRASTRUCTURE:
                    rec.priority = min(5, rec.priority + 2)
                    
        return sorted(recommendations, key=lambda x: x.priority)
        
    def estimate_improvement_potential(
        self,
        recommendations: List[OptimizationRecommendation],
        current_result: ProfileResult
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate potential improvements from implementing recommendations.
        
        Args:
            recommendations: List of recommendations to analyze
            current_result: Current performance baseline
            
        Returns:
            Dictionary with improvement estimates per recommendation
        """
        estimates = {}
        
        for rec in recommendations:
            category = rec.category
            
            # Rough estimates based on category and current metrics
            if category == OptimizationCategory.BATCH_SIZE:
                potential_throughput = current_result.throughput * 1.5  # 50% improvement
                potential_latency = current_result.avg_latency * 1.1    # 10% worse
            elif category == OptimizationCategory.CONCURRENCY:
                potential_throughput = current_result.throughput * 2.0  # 100% improvement
                potential_latency = current_result.avg_latency * 1.0    # Same
            elif category == OptimizationCategory.MODEL:
                potential_throughput = current_result.throughput * 1.3  # 30% improvement
                potential_latency = current_result.avg_latency * 0.7    # 30% better
            elif category == OptimizationCategory.SYSTEM:
                potential_throughput = current_result.throughput * 1.2  # 20% improvement
                potential_latency = current_result.avg_latency * 0.9    # 10% better
            else:
                potential_throughput = current_result.throughput * 1.1  # 10% improvement
                potential_latency = current_result.avg_latency * 0.95   # 5% better
                
            estimates[rec.title] = {
                "current_throughput": current_result.throughput,
                "potential_throughput": potential_throughput,
                "throughput_improvement": (potential_throughput - current_result.throughput) / current_result.throughput,
                "current_latency_ms": current_result.avg_latency * 1000,
                "potential_latency_ms": potential_latency * 1000,
                "latency_improvement": (current_result.avg_latency - potential_latency) / current_result.avg_latency
            }
            
        return estimates
        
    def generate_implementation_plan(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> Dict[str, List[OptimizationRecommendation]]:
        """
        Generate a phased implementation plan for recommendations.
        
        Args:
            recommendations: List of recommendations to plan
            
        Returns:
            Dictionary mapping phases to recommendation lists
        """
        # Group by implementation effort and priority
        low_effort = [r for r in recommendations if r.implementation_effort == "Low"]
        medium_effort = [r for r in recommendations if r.implementation_effort == "Medium"]
        high_effort = [r for r in recommendations if r.implementation_effort == "High"]
        
        # Sort each group by priority
        low_effort.sort(key=lambda x: x.priority)
        medium_effort.sort(key=lambda x: x.priority)
        high_effort.sort(key=lambda x: x.priority)
        
        return {
            "Phase 1 - Quick Wins (Low Effort)": low_effort,
            "Phase 2 - Medium Effort Improvements": medium_effort,
            "Phase 3 - High Effort Transformations": high_effort
        }