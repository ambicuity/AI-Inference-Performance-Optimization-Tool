"""
Performance analyzer module for statistical analysis of profiling results.

This module provides the PerformanceAnalyzer class that performs statistical
analysis on large performance datasets to identify patterns, anomalies, and
optimization opportunities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

from .profiler import ProfileResult
from .monitor import ResourceStats


@dataclass
class PerformanceInsights:
    """Statistical insights from performance analysis"""
    latency_distribution: str  # Description of latency distribution
    performance_stability: str  # Description of performance stability
    outlier_percentage: float
    anomaly_threshold: float
    trend_analysis: str
    statistical_summary: Dict[str, float]
    recommendations: List[str]


class PerformanceAnalyzer:
    """
    Analyze performance data and provide statistical insights.
    
    This class performs comprehensive statistical analysis on inference
    performance data to identify patterns, detect anomalies, and provide
    optimization recommendations based on large performance datasets.
    """
    
    def __init__(self):
        """Initialize the performance analyzer"""
        self.results_history: List[ProfileResult] = []
        self.resource_history: List[ResourceStats] = []
        
    def add_profile_result(self, result: ProfileResult):
        """Add a profile result to the analysis history"""
        self.results_history.append(result)
        
    def add_resource_stats(self, stats: ResourceStats):
        """Add resource statistics to the analysis history"""
        self.resource_history.append(stats)
        
    def analyze_latency_distribution(self, result: ProfileResult) -> PerformanceInsights:
        """
        Analyze the latency distribution and provide insights.
        
        Args:
            result: ProfileResult to analyze
            
        Returns:
            PerformanceInsights with statistical analysis
        """
        latencies = np.array(result.individual_times)
        
        # Basic statistical measures
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        std_lat = np.std(latencies)
        cv = std_lat / mean_lat if mean_lat > 0 else 0  # Coefficient of variation
        
        # Distribution analysis
        _, p_value = stats.normaltest(latencies)
        is_normal = p_value > 0.05
        
        # Skewness and kurtosis
        skewness = stats.skew(latencies)
        kurtosis = stats.kurtosis(latencies)
        
        # Outlier detection using IQR method
        q1, q3 = np.percentile(latencies, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = latencies[(latencies < lower_bound) | (latencies > upper_bound)]
        outlier_percentage = len(outliers) / len(latencies) * 100
        
        # Performance stability assessment
        if cv < 0.1:
            stability = "Very stable performance (low variance)"
        elif cv < 0.2:
            stability = "Stable performance"
        elif cv < 0.4:
            stability = "Moderate performance variability"
        else:
            stability = "High performance variability (unstable)"
            
        # Distribution description
        if is_normal:
            if abs(skewness) < 0.5:
                distribution = "Normal distribution with symmetric latencies"
            elif skewness > 0.5:
                distribution = "Positively skewed distribution (occasional high latencies)"
            else:
                distribution = "Negatively skewed distribution (occasional low latencies)"
        else:
            if skewness > 1:
                distribution = "Highly right-skewed distribution (frequent outliers)"
            elif skewness < -1:
                distribution = "Highly left-skewed distribution"
            else:
                distribution = "Non-normal distribution"
                
        # Trend analysis (if we have multiple results)
        trend_analysis = self._analyze_performance_trend()
        
        # Generate recommendations
        recommendations = self._generate_latency_recommendations(
            cv, outlier_percentage, skewness, result
        )
        
        statistical_summary = {
            "mean_latency_ms": mean_lat * 1000,
            "median_latency_ms": median_lat * 1000,
            "std_latency_ms": std_lat * 1000,
            "coefficient_of_variation": cv,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "p95_latency_ms": result.p95_latency * 1000,
            "p99_latency_ms": result.p99_latency * 1000,
        }
        
        return PerformanceInsights(
            latency_distribution=distribution,
            performance_stability=stability,
            outlier_percentage=outlier_percentage,
            anomaly_threshold=upper_bound,
            trend_analysis=trend_analysis,
            statistical_summary=statistical_summary,
            recommendations=recommendations
        )
        
    def _analyze_performance_trend(self) -> str:
        """Analyze performance trend over multiple results"""
        if len(self.results_history) < 2:
            return "Insufficient data for trend analysis"
            
        # Get last 10 results for trend analysis
        recent_results = self.results_history[-10:]
        throughputs = [r.throughput for r in recent_results]
        
        # Linear regression for trend
        x = np.arange(len(throughputs))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, throughputs)
        
        if abs(slope) < 0.01:
            return "Stable performance trend"
        elif slope > 0:
            return f"Improving performance trend (+{slope:.3f} req/s per run)"
        else:
            return f"Declining performance trend ({slope:.3f} req/s per run)"
            
    def _generate_latency_recommendations(
        self, 
        cv: float, 
        outlier_percentage: float, 
        skewness: float, 
        result: ProfileResult
    ) -> List[str]:
        """Generate recommendations based on latency analysis"""
        recommendations = []
        
        if cv > 0.3:
            recommendations.append("High latency variability detected - investigate system load and resource contention")
            
        if outlier_percentage > 5:
            recommendations.append(f"{outlier_percentage:.1f}% outliers detected - consider implementing request timeout and retry logic")
            
        if skewness > 1:
            recommendations.append("Right-skewed latency distribution indicates occasional slow requests - investigate tail latencies")
            
        if result.throughput < 10:
            recommendations.append("Low throughput detected - consider batch processing or model optimization")
            
        if result.p99_latency / result.median_latency > 3:
            recommendations.append("High P99/median ratio indicates tail latency issues - investigate system bottlenecks")
            
        return recommendations
        
    def compare_multiple_results(
        self, 
        results: Dict[str, ProfileResult],
        metric: str = "throughput"
    ) -> pd.DataFrame:
        """
        Compare multiple profile results across different configurations.
        
        Args:
            results: Dictionary mapping config names to ProfileResult
            metric: Metric to compare ('throughput', 'latency', 'p95', 'p99')
            
        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []
        
        for config_name, result in results.items():
            if metric == "throughput":
                primary_value = result.throughput
                unit = "req/s"
            elif metric == "latency":
                primary_value = result.avg_latency * 1000
                unit = "ms"
            elif metric == "p95":
                primary_value = result.p95_latency * 1000
                unit = "ms"
            elif metric == "p99":
                primary_value = result.p99_latency * 1000
                unit = "ms"
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            comparison_data.append({
                "Configuration": config_name,
                f"{metric.title()}": primary_value,
                "Unit": unit,
                "Samples": result.total_samples,
                "Total_Time": result.total_time,
                "Throughput": result.throughput,
                "Avg_Latency_ms": result.avg_latency * 1000,
                "P95_Latency_ms": result.p95_latency * 1000,
                "P99_Latency_ms": result.p99_latency * 1000,
                "Latency_Std_ms": result.latency_std * 1000,
            })
            
        df = pd.DataFrame(comparison_data)
        
        # Add ranking
        ascending = metric == "latency" or "latency" in metric.lower()
        df[f"{metric.title()}_Rank"] = df[f"{metric.title()}"].rank(ascending=ascending)
        
        return df.sort_values(f"{metric.title()}_Rank")
        
    def detect_performance_anomalies(
        self, 
        result: ProfileResult, 
        threshold_std: float = 3.0
    ) -> Tuple[List[float], List[int]]:
        """
        Detect anomalous latency measurements using statistical methods.
        
        Args:
            result: ProfileResult to analyze
            threshold_std: Number of standard deviations for anomaly threshold
            
        Returns:
            Tuple of (anomalous_values, anomalous_indices)
        """
        latencies = np.array(result.individual_times)
        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        
        # Z-score based anomaly detection
        z_scores = np.abs((latencies - mean_lat) / std_lat)
        anomaly_mask = z_scores > threshold_std
        
        anomalous_values = latencies[anomaly_mask].tolist()
        anomalous_indices = np.where(anomaly_mask)[0].tolist()
        
        return anomalous_values, anomalous_indices
        
    def cluster_performance_patterns(
        self, 
        results: List[ProfileResult], 
        n_clusters: int = 3
    ) -> Dict[str, Any]:
        """
        Cluster performance results to identify patterns.
        
        Args:
            results: List of ProfileResult objects
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results and insights
        """
        if len(results) < n_clusters:
            return {"error": "Insufficient results for clustering"}
            
        # Extract features for clustering
        features = []
        labels = []
        
        for i, result in enumerate(results):
            features.append([
                result.throughput,
                result.avg_latency * 1000,  # Convert to ms
                result.p95_latency * 1000,
                result.latency_std * 1000,
                len(result.individual_times)
            ])
            labels.append(f"Result_{i}")
            
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            
            cluster_analysis[f"Cluster_{cluster_id}"] = {
                "count": int(np.sum(cluster_mask)),
                "avg_throughput": float(np.mean(cluster_features[:, 0])),
                "avg_latency_ms": float(np.mean(cluster_features[:, 1])),
                "avg_p95_ms": float(np.mean(cluster_features[:, 2])),
                "member_indices": np.where(cluster_mask)[0].tolist()
            }
            
        return {
            "cluster_labels": cluster_labels.tolist(),
            "cluster_analysis": cluster_analysis,
            "feature_names": ["throughput", "avg_latency_ms", "p95_latency_ms", "latency_std_ms", "samples"],
            "scaler": scaler,
            "kmeans_model": kmeans
        }
        
    def generate_performance_report(
        self, 
        result: ProfileResult, 
        resource_stats: Optional[ResourceStats] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive performance analysis report.
        
        Args:
            result: ProfileResult to analyze
            resource_stats: Optional ResourceStats to include
            save_path: Optional path to save the report
            
        Returns:
            String containing the formatted report
        """
        insights = self.analyze_latency_distribution(result)
        
        report = []
        report.append("=" * 60)
        report.append("AI INFERENCE PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Samples: {result.total_samples:,}")
        report.append(f"Total Time: {result.total_time:.2f} seconds")
        report.append(f"Throughput: {result.throughput:.2f} requests/second")
        report.append(f"Average Latency: {result.avg_latency*1000:.2f} ms")
        report.append(f"P95 Latency: {result.p95_latency*1000:.2f} ms")
        report.append(f"P99 Latency: {result.p99_latency*1000:.2f} ms")
        report.append("")
        
        # Statistical Analysis
        report.append("STATISTICAL ANALYSIS")
        report.append("-" * 20)
        report.append(f"Distribution: {insights.latency_distribution}")
        report.append(f"Stability: {insights.performance_stability}")
        report.append(f"Outliers: {insights.outlier_percentage:.1f}% of samples")
        report.append(f"Trend: {insights.trend_analysis}")
        report.append("")
        
        # Detailed Statistics
        report.append("DETAILED STATISTICS")
        report.append("-" * 20)
        for key, value in insights.statistical_summary.items():
            if "_ms" in key:
                report.append(f"{key.replace('_', ' ').title()}: {value:.2f} ms")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value:.4f}")
        report.append("")
        
        # Resource Usage (if available)
        if resource_stats:
            report.append("RESOURCE UTILIZATION")
            report.append("-" * 20)
            report.append(f"CPU Usage: {resource_stats.cpu_avg:.1f}% avg, {resource_stats.cpu_max:.1f}% peak")
            report.append(f"Memory Usage: {resource_stats.memory_avg:.1f}% avg, {resource_stats.memory_max:.1f}% peak")
            report.append(f"Memory Used: {resource_stats.memory_used_avg_mb:.0f} MB avg, {resource_stats.memory_used_max_mb:.0f} MB peak")
            
            if resource_stats.gpu_util_avg is not None:
                report.append(f"GPU Utilization: {resource_stats.gpu_util_avg:.1f}% avg, {resource_stats.gpu_util_max:.1f}% peak")
                report.append(f"GPU Memory: {resource_stats.gpu_memory_used_avg_mb:.0f} MB avg, {resource_stats.gpu_memory_used_max_mb:.0f} MB peak")
                
            if resource_stats.gpu_temp_avg is not None:
                report.append(f"GPU Temperature: {resource_stats.gpu_temp_avg:.1f}°C avg, {resource_stats.gpu_temp_max:.1f}°C peak")
            report.append("")
        
        # Recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("-" * 30)
        for i, rec in enumerate(insights.recommendations, 1):
            report.append(f"{i}. {rec}")
            
        if resource_stats:
            from .monitor import ResourceMonitor
            monitor = ResourceMonitor()
            resource_recs = monitor.get_resource_recommendations(resource_stats)
            for i, rec in enumerate(resource_recs, len(insights.recommendations) + 1):
                report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
            
        return report_text
        
    def visualize_latency_distribution(
        self, 
        result: ProfileResult, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualization of latency distribution.
        
        Args:
            result: ProfileResult to visualize
            save_path: Optional path to save the plot
        """
        latencies_ms = np.array(result.individual_times) * 1000
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI Inference Latency Analysis', fontsize=16)
        
        # Histogram
        ax1.hist(latencies_ms, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(latencies_ms), color='red', linestyle='--', label=f'Mean: {np.mean(latencies_ms):.2f} ms')
        ax1.axvline(np.median(latencies_ms), color='orange', linestyle='--', label=f'Median: {np.median(latencies_ms):.2f} ms')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Density')
        ax1.set_title('Latency Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(latencies_ms, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Time series
        ax3.plot(range(len(latencies_ms)), latencies_ms, alpha=0.6, linewidth=0.5)
        ax3.set_xlabel('Request Index')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(latencies_ms, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normal Distribution)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
            
    def export_results_to_csv(
        self, 
        results: Dict[str, ProfileResult], 
        filename: str
    ) -> None:
        """
        Export multiple ProfileResults to CSV for further analysis.
        
        Args:
            results: Dictionary mapping config names to ProfileResult
            filename: Output CSV filename
        """
        df = self.compare_multiple_results(results)
        df.to_csv(filename, index=False)
        print(f"Results exported to: {filename}")