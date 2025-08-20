"""
Command-line interface for the AI Inference Performance Optimization Tool.

This module provides a comprehensive CLI for profiling AI inference workloads,
monitoring system resources, and generating optimization recommendations.
"""

import click
import time
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import importlib.util
import numpy as np

from . import InferenceProfiler, ResourceMonitor, PerformanceAnalyzer, OptimizationRecommender
from .profiler import ProfileResult
from .monitor import ContextResourceMonitor, ResourceStats
from .optimizer import OptimizationRecommendation


def load_function_from_file(file_path: str, function_name: str) -> Callable:
    """Load a function from a Python file"""
    spec = importlib.util.spec_from_file_location("user_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def create_dummy_inference_function(latency_ms: float = 50) -> Callable:
    """Create a dummy inference function for demonstration"""
    def dummy_inference(batch_size: int = 1):
        # Simulate inference with configurable latency
        time.sleep(latency_ms / 1000.0 * batch_size)
        return f"Processed batch of {batch_size} items"
    return dummy_inference


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """AI Inference Performance Optimization Tool
    
    A comprehensive tool for profiling and optimizing AI inference workloads.
    Measures throughput, latency, and resource utilization to identify 
    bottlenecks and provide optimization recommendations.
    """
    pass


@cli.command()
@click.option('--function-file', type=str, help='Python file containing the function to profile')
@click.option('--function-name', type=str, default='inference', help='Name of function to profile')
@click.option('--runs', default=100, help='Number of profiling runs')
@click.option('--warmup', default=10, help='Number of warmup runs')
@click.option('--concurrent', default=1, help='Number of concurrent requests')
@click.option('--batch-size', default=1, help='Batch size for processing')
@click.option('--timeout', type=float, help='Timeout for profiling session')
@click.option('--monitor-resources', is_flag=True, help='Monitor system resources during profiling')
@click.option('--output', type=str, help='Output file for results (JSON)')
@click.option('--demo', is_flag=True, help='Run with demo function')
@click.option('--demo-latency', default=50.0, help='Demo function latency in ms')
def profile(function_file, function_name, runs, warmup, concurrent, batch_size, 
           timeout, monitor_resources, output, demo, demo_latency):
    """Profile an AI inference function"""
    
    click.echo("🚀 AI Inference Performance Profiler")
    click.echo("=" * 50)
    
    # Load or create function to profile
    if demo:
        func = create_dummy_inference_function(demo_latency)
        click.echo(f"📋 Using demo function with {demo_latency}ms latency")
    elif function_file:
        if not os.path.exists(function_file):
            click.echo(f"❌ Error: Function file '{function_file}' not found")
            sys.exit(1)
        try:
            func = load_function_from_file(function_file, function_name)
            click.echo(f"📋 Loaded function '{function_name}' from '{function_file}'")
        except Exception as e:
            click.echo(f"❌ Error loading function: {e}")
            sys.exit(1)
    else:
        click.echo("❌ Error: Must specify either --function-file or --demo")
        sys.exit(1)
    
    # Setup profiler and monitor
    profiler = InferenceProfiler(warmup_runs=warmup)
    resource_monitor = ResourceMonitor() if monitor_resources else None
    
    click.echo(f"⚙️  Configuration:")
    click.echo(f"   • Runs: {runs}")
    click.echo(f"   • Warmup: {warmup}")
    click.echo(f"   • Concurrent requests: {concurrent}")
    click.echo(f"   • Batch size: {batch_size}")
    if timeout:
        click.echo(f"   • Timeout: {timeout}s")
    if monitor_resources:
        click.echo(f"   • Resource monitoring: enabled")
    click.echo()
    
    # Run profiling with optional resource monitoring
    resource_stats = None
    if resource_monitor:
        with ContextResourceMonitor(resource_monitor) as context:
            result = profiler.profile_function(
                func=func,
                kwargs={"batch_size": batch_size},
                num_runs=runs,
                concurrent_requests=concurrent,
                timeout=timeout
            )
        resource_stats = context.get_stats()
    else:
        result = profiler.profile_function(
            func=func,
            kwargs={"batch_size": batch_size}, 
            num_runs=runs,
            concurrent_requests=concurrent,
            timeout=timeout
        )
    
    # Display results
    click.echo("📊 Performance Results")
    click.echo("-" * 30)
    click.echo(f"Total samples: {result.total_samples:,}")
    click.echo(f"Total time: {result.total_time:.2f}s")
    click.echo(f"Throughput: {result.throughput:.2f} req/s")
    click.echo(f"Average latency: {result.avg_latency*1000:.2f}ms")
    click.echo(f"Median latency: {result.median_latency*1000:.2f}ms")
    click.echo(f"P95 latency: {result.p95_latency*1000:.2f}ms")
    click.echo(f"P99 latency: {result.p99_latency*1000:.2f}ms")
    click.echo(f"Min latency: {result.min_latency*1000:.2f}ms")
    click.echo(f"Max latency: {result.max_latency*1000:.2f}ms")
    click.echo(f"Latency std dev: {result.latency_std*1000:.2f}ms")
    click.echo()
    
    # Display resource usage if monitored
    if resource_stats:
        click.echo("💻 Resource Utilization")
        click.echo("-" * 30)
        click.echo(f"CPU: {resource_stats.cpu_avg:.1f}% avg, {resource_stats.cpu_max:.1f}% max")
        click.echo(f"Memory: {resource_stats.memory_avg:.1f}% avg, {resource_stats.memory_max:.1f}% max")
        click.echo(f"Memory used: {resource_stats.memory_used_avg_mb:.0f}MB avg, {resource_stats.memory_used_max_mb:.0f}MB max")
        
        if resource_stats.gpu_util_avg is not None:
            click.echo(f"GPU util: {resource_stats.gpu_util_avg:.1f}% avg, {resource_stats.gpu_util_max:.1f}% max")
            click.echo(f"GPU memory: {resource_stats.gpu_memory_used_avg_mb:.0f}MB avg, {resource_stats.gpu_memory_used_max_mb:.0f}MB max")
            
        if resource_stats.gpu_temp_avg is not None:
            click.echo(f"GPU temp: {resource_stats.gpu_temp_avg:.1f}°C avg, {resource_stats.gpu_temp_max:.1f}°C max")
        click.echo()
    
    # Save results if requested
    if output:
        results_data = {
            "profile_result": {
                "total_samples": result.total_samples,
                "total_time": result.total_time,
                "throughput": result.throughput,
                "avg_latency": result.avg_latency,
                "median_latency": result.median_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "min_latency": result.min_latency,
                "max_latency": result.max_latency,
                "latency_std": result.latency_std,
                "individual_times": result.individual_times
            }
        }
        
        if resource_stats:
            results_data["resource_stats"] = {
                "cpu_avg": resource_stats.cpu_avg,
                "cpu_max": resource_stats.cpu_max,
                "memory_avg": resource_stats.memory_avg,
                "memory_max": resource_stats.memory_max,
                "memory_used_avg_mb": resource_stats.memory_used_avg_mb,
                "memory_used_max_mb": resource_stats.memory_used_max_mb,
                "gpu_util_avg": resource_stats.gpu_util_avg,
                "gpu_util_max": resource_stats.gpu_util_max,
                "gpu_memory_used_avg_mb": resource_stats.gpu_memory_used_avg_mb,
                "gpu_memory_used_max_mb": resource_stats.gpu_memory_used_max_mb,
                "gpu_temp_avg": resource_stats.gpu_temp_avg,
                "gpu_temp_max": resource_stats.gpu_temp_max
            }
        
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"💾 Results saved to: {output}")


@cli.command()
@click.option('--config', type=str, required=True, help='Configuration file (YAML/JSON)')
@click.option('--output', type=str, help='Output file for comparison results')
def compare(config, output):
    """Compare performance across different configurations"""
    
    click.echo("🔍 Configuration Comparison Tool")
    click.echo("=" * 50)
    
    # Load configuration file
    if not os.path.exists(config):
        click.echo(f"❌ Error: Configuration file '{config}' not found")
        sys.exit(1)
        
    try:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
    except Exception as e:
        click.echo(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    required_fields = ['function_file', 'function_name', 'configurations']
    for field in required_fields:
        if field not in config_data:
            click.echo(f"❌ Error: Missing required field '{field}' in configuration")
            sys.exit(1)
    
    # Load function
    try:
        func = load_function_from_file(config_data['function_file'], config_data['function_name'])
        click.echo(f"📋 Loaded function '{config_data['function_name']}' from '{config_data['function_file']}'")
    except Exception as e:
        click.echo(f"❌ Error loading function: {e}")
        sys.exit(1)
    
    # Setup profiler
    profiler = InferenceProfiler(
        warmup_runs=config_data.get('warmup_runs', 10)
    )
    
    # Run comparisons
    num_runs = config_data.get('num_runs', 50)
    configurations = config_data['configurations']
    
    click.echo(f"🔄 Testing {len(configurations)} configurations with {num_runs} runs each")
    click.echo()
    
    results = profiler.compare_configurations(
        configs=configurations,
        func=func,
        base_args=(),
        base_kwargs={},
        num_runs=num_runs
    )
    
    # Display comparison results
    click.echo("📊 Comparison Results")
    click.echo("-" * 50)
    
    # Create comparison table
    analyzer = PerformanceAnalyzer()
    comparison_df = analyzer.compare_multiple_results(results, metric="throughput")
    
    # Display table
    for _, row in comparison_df.iterrows():
        config_name = row['Configuration']
        throughput = row['Throughput']
        avg_latency = row['Avg_Latency_ms']
        p95_latency = row['P95_Latency_ms']
        rank = row['Throughput_Rank']
        
        click.echo(f"#{int(rank)} {config_name}")
        click.echo(f"   Throughput: {throughput:.2f} req/s")
        click.echo(f"   Avg Latency: {avg_latency:.2f}ms")
        click.echo(f"   P95 Latency: {p95_latency:.2f}ms")
        click.echo()
    
    # Save results if requested
    if output:
        comparison_df.to_csv(output, index=False)
        click.echo(f"💾 Comparison results saved to: {output}")


@cli.command()
@click.option('--input', type=str, required=True, help='Input JSON file with profiling results')
@click.option('--output', type=str, help='Output file for analysis report')
@click.option('--visualize', is_flag=True, help='Generate latency distribution plots')
def analyze(input, output, visualize):
    """Analyze profiling results and generate insights"""
    
    click.echo("📈 Performance Analysis Tool")
    click.echo("=" * 50)
    
    # Load results
    if not os.path.exists(input):
        click.echo(f"❌ Error: Input file '{input}' not found")
        sys.exit(1)
        
    try:
        with open(input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        click.echo(f"❌ Error loading input file: {e}")
        sys.exit(1)
    
    # Reconstruct ProfileResult
    profile_data = data['profile_result']
    result = ProfileResult(
        total_samples=profile_data['total_samples'],
        total_time=profile_data['total_time'],
        throughput=profile_data['throughput'],
        avg_latency=profile_data['avg_latency'],
        median_latency=profile_data['median_latency'],
        p95_latency=profile_data['p95_latency'],
        p99_latency=profile_data['p99_latency'],
        min_latency=profile_data['min_latency'],
        max_latency=profile_data['max_latency'],
        latency_std=profile_data['latency_std'],
        individual_times=profile_data['individual_times']
    )
    
    # Reconstruct ResourceStats if available
    resource_stats = None
    if 'resource_stats' in data:
        from .monitor import ResourceSnapshot
        res_data = data['resource_stats']
        resource_stats = ResourceStats(
            cpu_avg=res_data['cpu_avg'],
            cpu_max=res_data['cpu_max'],
            cpu_min=0.0,  # Not stored in JSON
            memory_avg=res_data['memory_avg'],
            memory_max=res_data['memory_max'],
            memory_min=0.0,  # Not stored in JSON
            memory_used_avg_mb=res_data['memory_used_avg_mb'],
            memory_used_max_mb=res_data['memory_used_max_mb'],
            gpu_util_avg=res_data.get('gpu_util_avg'),
            gpu_util_max=res_data.get('gpu_util_max'),
            gpu_memory_used_avg_mb=res_data.get('gpu_memory_used_avg_mb'),
            gpu_memory_used_max_mb=res_data.get('gpu_memory_used_max_mb'),
            gpu_temp_avg=res_data.get('gpu_temp_avg'),
            gpu_temp_max=res_data.get('gpu_temp_max'),
            snapshots=[]
        )
    
    # Perform analysis
    analyzer = PerformanceAnalyzer()
    insights = analyzer.analyze_latency_distribution(result)
    
    # Generate and display report
    report = analyzer.generate_performance_report(result, resource_stats, output)
    click.echo(report)
    
    # Generate visualizations if requested
    if visualize:
        viz_path = input.replace('.json', '_latency_distribution.png')
        analyzer.visualize_latency_distribution(result, viz_path)


@cli.command()
@click.option('--input', type=str, required=True, help='Input JSON file with profiling results')
@click.option('--system-info', type=str, help='System info file (JSON)')
@click.option('--targets', type=str, help='Performance targets file (JSON)')
@click.option('--output', type=str, help='Output file for recommendations')
def optimize(input, system_info, targets, output):
    """Generate optimization recommendations"""
    
    click.echo("🎯 Optimization Recommender")
    click.echo("=" * 50)
    
    # Load profiling results
    if not os.path.exists(input):
        click.echo(f"❌ Error: Input file '{input}' not found")
        sys.exit(1)
        
    try:
        with open(input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        click.echo(f"❌ Error loading input file: {e}")
        sys.exit(1)
    
    # Reconstruct ProfileResult and ResourceStats
    profile_data = data['profile_result']
    result = ProfileResult(
        total_samples=profile_data['total_samples'],
        total_time=profile_data['total_time'],
        throughput=profile_data['throughput'],
        avg_latency=profile_data['avg_latency'],
        median_latency=profile_data['median_latency'],
        p95_latency=profile_data['p95_latency'],
        p99_latency=profile_data['p99_latency'],
        min_latency=profile_data['min_latency'],
        max_latency=profile_data['max_latency'],
        latency_std=profile_data['latency_std'],
        individual_times=profile_data['individual_times']
    )
    
    resource_stats = None
    if 'resource_stats' in data:
        res_data = data['resource_stats']
        resource_stats = ResourceStats(
            cpu_avg=res_data['cpu_avg'],
            cpu_max=res_data['cpu_max'],
            cpu_min=0.0,
            memory_avg=res_data['memory_avg'],
            memory_max=res_data['memory_max'],
            memory_min=0.0,
            memory_used_avg_mb=res_data['memory_used_avg_mb'],
            memory_used_max_mb=res_data['memory_used_max_mb'],
            gpu_util_avg=res_data.get('gpu_util_avg'),
            gpu_util_max=res_data.get('gpu_util_max'),
            gpu_memory_used_avg_mb=res_data.get('gpu_memory_used_avg_mb'),
            gpu_memory_used_max_mb=res_data.get('gpu_memory_used_max_mb'),
            gpu_temp_avg=res_data.get('gpu_temp_avg'),
            gpu_temp_max=res_data.get('gpu_temp_max'),
            snapshots=[]
        )
    
    # Load optional files
    system_data = None
    if system_info and os.path.exists(system_info):
        with open(system_info, 'r') as f:
            system_data = json.load(f)
    
    target_data = None
    if targets and os.path.exists(targets):
        with open(targets, 'r') as f:
            target_data = json.load(f)
    
    # Generate recommendations
    recommender = OptimizationRecommender()
    recommendations = recommender.generate_recommendations(
        profile_result=result,
        resource_stats=resource_stats,
        system_info=system_data,
        target_metrics=target_data
    )
    
    click.echo(f"🎯 Generated {len(recommendations)} optimization recommendations:")
    click.echo()
    
    # Display recommendations by priority
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = ["🔥", "⚡", "💡", "📝", "💭"][rec.priority - 1]
        effort_color = {"Low": "green", "Medium": "yellow", "High": "red"}[rec.implementation_effort]
        
        click.echo(f"{priority_emoji} #{i} {rec.title}")
        click.echo(f"   Category: {rec.category.value}")
        click.echo(f"   Priority: {rec.priority}/5")
        click.echo(f"   Effort: {click.style(rec.implementation_effort, fg=effort_color.lower())}")
        click.echo(f"   Expected: {rec.expected_improvement}")
        click.echo(f"   Description: {rec.description}")
        
        if rec.specific_actions:
            click.echo("   Actions:")
            for action in rec.specific_actions[:3]:  # Show top 3 actions
                click.echo(f"   • {action}")
                
        click.echo()
    
    # Generate implementation plan
    plan = recommender.generate_implementation_plan(recommendations)
    
    click.echo("📋 Implementation Plan")
    click.echo("-" * 30)
    for phase, recs in plan.items():
        if recs:
            click.echo(f"\n{phase}:")
            for rec in recs:
                click.echo(f"  • {rec.title} (Priority: {rec.priority})")
    
    # Save recommendations if requested
    if output:
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                "title": rec.title,
                "category": rec.category.value,
                "description": rec.description,
                "expected_improvement": rec.expected_improvement,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority,
                "specific_actions": rec.specific_actions
            })
        
        output_data = {
            "recommendations": recommendations_data,
            "implementation_plan": {
                phase: [{"title": r.title, "priority": r.priority} for r in recs]
                for phase, recs in plan.items()
            }
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\n💾 Recommendations saved to: {output}")


@cli.command()
@click.option('--sample-interval', default=0.1, help='Resource sampling interval in seconds')
@click.option('--duration', default=60, help='Monitoring duration in seconds') 
@click.option('--output', type=str, help='Output file for resource data')
def monitor(sample_interval, duration, output):
    """Monitor system resources in real-time"""
    
    click.echo("📊 System Resource Monitor")
    click.echo("=" * 50)
    
    monitor = ResourceMonitor(sample_interval=sample_interval)
    
    click.echo(f"⏱️  Monitoring system resources for {duration} seconds...")
    click.echo("Press Ctrl+C to stop early")
    
    try:
        with ContextResourceMonitor(monitor) as context:
            # Show real-time updates
            start_time = time.time()
            while time.time() - start_time < duration:
                current = monitor.get_current_usage()
                if current:
                    elapsed = time.time() - start_time
                    click.echo(f"\r⏱️  {elapsed:5.1f}s | "
                             f"CPU: {current.cpu_percent:5.1f}% | "
                             f"Memory: {current.memory_percent:5.1f}% "
                             f"({current.memory_used_mb:6.0f}MB)", nl=False)
                    
                    if current.gpu_utilization is not None:
                        click.echo(f" | GPU: {current.gpu_utilization:5.1f}% "
                                 f"({current.gpu_memory_used_mb:6.0f}MB)", nl=False)
                
                time.sleep(1)
                
        stats = context.get_stats()
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Monitoring stopped by user")
        stats = monitor.stop_monitoring()
    
    click.echo("\n\n📊 Resource Usage Summary")
    click.echo("-" * 30)
    click.echo(f"CPU: {stats.cpu_avg:.1f}% avg, {stats.cpu_max:.1f}% max")
    click.echo(f"Memory: {stats.memory_avg:.1f}% avg, {stats.memory_max:.1f}% max")
    click.echo(f"Memory used: {stats.memory_used_avg_mb:.0f}MB avg, {stats.memory_used_max_mb:.0f}MB max")
    
    if stats.gpu_util_avg is not None:
        click.echo(f"GPU util: {stats.gpu_util_avg:.1f}% avg, {stats.gpu_util_max:.1f}% max")
        click.echo(f"GPU memory: {stats.gpu_memory_used_avg_mb:.0f}MB avg, {stats.gpu_memory_used_max_mb:.0f}MB max")
    
    if stats.gpu_temp_avg is not None:
        click.echo(f"GPU temp: {stats.gpu_temp_avg:.1f}°C avg, {stats.gpu_temp_max:.1f}°C max")
    
    # Detect bottlenecks
    bottlenecks = monitor.detect_bottlenecks(stats)
    if bottlenecks:
        click.echo("\n⚠️  Detected Bottlenecks:")
        for bottleneck in bottlenecks:
            click.echo(f"   • {bottleneck}")
    
    # Save data if requested
    if output:
        stats_data = {
            "cpu_avg": stats.cpu_avg,
            "cpu_max": stats.cpu_max,
            "memory_avg": stats.memory_avg,
            "memory_max": stats.memory_max,
            "memory_used_avg_mb": stats.memory_used_avg_mb,
            "memory_used_max_mb": stats.memory_used_max_mb,
            "gpu_util_avg": stats.gpu_util_avg,
            "gpu_util_max": stats.gpu_util_max,
            "gpu_memory_used_avg_mb": stats.gpu_memory_used_avg_mb,
            "gpu_memory_used_max_mb": stats.gpu_memory_used_max_mb,
            "gpu_temp_avg": stats.gpu_temp_avg,
            "gpu_temp_max": stats.gpu_temp_max,
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "memory_used_mb": s.memory_used_mb,
                    "gpu_utilization": s.gpu_utilization,
                    "gpu_memory_used_mb": s.gpu_memory_used_mb,
                    "gpu_temperature": s.gpu_temperature
                }
                for s in stats.snapshots
            ]
        }
        
        with open(output, 'w') as f:
            json.dump(stats_data, f, indent=2)
        click.echo(f"\n💾 Resource data saved to: {output}")


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()