# AI Inference Performance Optimization Tool

A comprehensive Python-based tool for profiling and optimizing the performance of AI inference workloads. This tool measures key performance metrics like throughput and latency, identifies resource bottlenecks in datacenter environments, and provides actionable optimization recommendations based on statistical performance analysis.

## Features

- **🎯 Performance Profiling**: Measure throughput, latency, and detailed performance statistics
- **📊 Resource Monitoring**: Track CPU, memory, and GPU utilization during inference
- **📈 Statistical Analysis**: Comprehensive analysis of performance patterns and distributions  
- **🔍 Bottleneck Detection**: Automatically identify system and resource bottlenecks
- **💡 Optimization Recommendations**: AI-driven suggestions for performance improvements
- **⚡ Batch Processing**: Optimized handling of large performance datasets
- **🔧 Configuration Comparison**: Compare performance across different configurations
- **📋 Detailed Reporting**: Generate comprehensive performance analysis reports

## Installation

### Option 1: Install from Source

```bash
git clone https://github.com/ambicuity/AI-Inference-Performance-Optimization-Tool.git
cd AI-Inference-Performance-Optimization-Tool
pip install -e .
```

### Option 2: Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

### Option 3: Install with GPU Monitoring Support

```bash
pip install -e ".[gpu]"
```

## Quick Start

### 1. Profile with Demo Function

```bash
# Profile a demo function with default settings
ai-optimizer profile --demo --runs 100

# Profile with resource monitoring
ai-optimizer profile --demo --runs 100 --monitor-resources --output results.json
```

### 2. Profile Your Own Function

Create a Python file with your inference function:

```python
# my_inference.py
import time

def inference(batch_size=1):
    # Your AI inference code here
    time.sleep(0.01 * batch_size)  # Simulate processing
    return f"Processed {batch_size} samples"
```

Profile it:

```bash
ai-optimizer profile --function-file my_inference.py --function-name inference --runs 100 --batch-size 4
```

### 3. Compare Different Configurations

Create a configuration file:

```yaml
# config.yaml
function_file: "my_inference.py"
function_name: "inference"
num_runs: 50

configurations:
  small_batch:
    batch_size: 1
  medium_batch:
    batch_size: 4
  large_batch:
    batch_size: 8
```

Run comparison:

```bash
ai-optimizer compare --config config.yaml --output comparison_results.csv
```

### 4. Get Optimization Recommendations

```bash
# Generate recommendations from profiling results
ai-optimizer optimize --input results.json --output recommendations.json

# Include system information for better recommendations
ai-optimizer optimize --input results.json --system-info system.json --targets targets.json
```

## Command Line Interface

### Profile Command

Profile inference functions and measure performance metrics.

```bash
ai-optimizer profile [OPTIONS]
```

**Options:**
- `--function-file`: Python file containing the function to profile
- `--function-name`: Name of function to profile (default: 'inference')
- `--runs`: Number of profiling runs (default: 100)
- `--warmup`: Number of warmup runs (default: 10)
- `--concurrent`: Number of concurrent requests (default: 1)
- `--batch-size`: Batch size for processing (default: 1)
- `--timeout`: Timeout for profiling session
- `--monitor-resources`: Monitor system resources during profiling
- `--output`: Output file for results (JSON)
- `--demo`: Run with demo function
- `--demo-latency`: Demo function latency in ms (default: 50.0)

### Compare Command

Compare performance across different configurations.

```bash
ai-optimizer compare --config CONFIG_FILE [--output OUTPUT_FILE]
```

### Analyze Command

Analyze profiling results and generate insights.

```bash
ai-optimizer analyze --input RESULTS_FILE [--output REPORT_FILE] [--visualize]
```

### Optimize Command

Generate optimization recommendations.

```bash
ai-optimizer optimize --input RESULTS_FILE [--system-info INFO_FILE] [--targets TARGETS_FILE] [--output RECOMMENDATIONS_FILE]
```

### Monitor Command

Monitor system resources in real-time.

```bash
ai-optimizer monitor [--duration SECONDS] [--sample-interval INTERVAL] [--output OUTPUT_FILE]
```

## Python API Usage

### Basic Profiling

```python
from ai_optimizer import InferenceProfiler

def my_inference_function(batch_size=1):
    # Your inference code here
    pass

# Create profiler
profiler = InferenceProfiler(warmup_runs=10)

# Profile the function
result = profiler.profile_function(
    func=my_inference_function,
    kwargs={"batch_size": 4},
    num_runs=100,
    concurrent_requests=1
)

print(f"Throughput: {result.throughput:.2f} req/s")
print(f"Average latency: {result.avg_latency*1000:.2f} ms")
print(f"P95 latency: {result.p95_latency*1000:.2f} ms")
```

### Resource Monitoring

```python
from ai_optimizer import ResourceMonitor
from ai_optimizer.monitor import ContextResourceMonitor

monitor = ResourceMonitor(sample_interval=0.1)

# Monitor during inference
with ContextResourceMonitor(monitor) as context:
    # Run your inference code here
    result = profiler.profile_function(my_inference_function, num_runs=50)

# Get resource statistics
stats = context.get_stats()
print(f"Average CPU usage: {stats.cpu_avg:.1f}%")
print(f"Average memory usage: {stats.memory_avg:.1f}%")
```

### Performance Analysis

```python
from ai_optimizer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze latency distribution
insights = analyzer.analyze_latency_distribution(result)
print(f"Distribution: {insights.latency_distribution}")
print(f"Stability: {insights.performance_stability}")

# Generate comprehensive report
report = analyzer.generate_performance_report(result, stats)
print(report)

# Create visualizations
analyzer.visualize_latency_distribution(result, save_path="latency_plot.png")
```

### Optimization Recommendations

```python
from ai_optimizer import OptimizationRecommender

recommender = OptimizationRecommender()

# Generate recommendations
recommendations = recommender.generate_recommendations(
    profile_result=result,
    resource_stats=stats,
    system_info={"cpu_cores": 8, "memory_gb": 32},
    target_metrics={"throughput": 100.0}
)

# Display recommendations
for rec in recommendations[:5]:  # Top 5 recommendations
    print(f"Priority {rec.priority}: {rec.title}")
    print(f"Category: {rec.category.value}")
    print(f"Expected improvement: {rec.expected_improvement}")
    print(f"Implementation effort: {rec.implementation_effort}")
    print()
```

## Configuration Files

### System Information (system_info.json)

```json
{
  "cpu_cores": 8,
  "memory_gb": 32,
  "gpu_available": true,
  "gpu_memory_gb": 12,
  "storage_type": "SSD",
  "network_bandwidth_mbps": 1000,
  "os": "Linux",
  "python_version": "3.9",
  "cpu_model": "Intel Xeon",
  "architecture": "x86_64"
}
```

### Performance Targets (performance_targets.json)

```json
{
  "throughput": 100.0,
  "avg_latency": 0.05,
  "p95_latency": 0.1,
  "p99_latency": 0.2,
  "max_acceptable_latency": 0.5
}
```

### Comparison Configuration (comparison_config.yaml)

```yaml
function_file: "my_inference.py"
function_name: "inference"
warmup_runs: 10
num_runs: 100

configurations:
  baseline:
    batch_size: 1
    
  optimized_batch:
    batch_size: 8
    
  high_concurrency:
    batch_size: 4
    concurrent_requests: 4
```

## Performance Metrics

The tool measures and analyzes the following key metrics:

### Throughput Metrics
- **Requests per second**: Overall processing rate
- **Samples per second**: Rate accounting for batch processing

### Latency Metrics
- **Average latency**: Mean response time
- **Median latency**: 50th percentile response time
- **P95 latency**: 95th percentile response time
- **P99 latency**: 99th percentile response time
- **Min/Max latency**: Range of response times
- **Latency standard deviation**: Response time variability

### Resource Metrics
- **CPU utilization**: Processor usage percentage
- **Memory usage**: RAM utilization percentage and absolute amounts
- **GPU utilization**: Graphics processor usage (if available)
- **GPU memory**: Graphics memory usage (if available)
- **GPU temperature**: Thermal monitoring (if available)

## Optimization Categories

The tool provides recommendations across several categories:

### 🔢 Batch Size Optimization
- Optimize batch sizes for better throughput
- Balance latency vs. throughput trade-offs
- Memory-aware batch sizing

### ⚡ Concurrency Optimization
- Scale concurrent request handling
- Load balancing improvements
- Thread/process pool optimization

### 💻 Resource Optimization
- CPU, memory, and GPU utilization improvements
- Resource allocation recommendations
- Bottleneck resolution strategies

### 🧠 Model Optimization
- Model architecture optimizations
- Quantization and pruning suggestions
- Hardware-specific optimizations

### 🔧 System Optimization
- Operating system and runtime optimizations
- Memory management improvements
- I/O and networking optimizations

### 🏗️ Infrastructure Optimization
- Hardware scaling recommendations
- Cloud instance optimization
- Storage and networking improvements

## Examples

See the `examples/` directory for complete example scripts and configurations:

- `sample_inference.py`: Example inference functions with different characteristics
- `batch_size_comparison.yaml`: Configuration for comparing batch sizes
- `system_info.json`: Sample system information file
- `performance_targets.json`: Sample performance targets

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ai_optimizer --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Requirements

### Core Dependencies
- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- psutil >= 5.8.0
- click >= 8.0.0
- pyyaml >= 5.4.0
- tqdm >= 4.60.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

### Optional Dependencies
- pynvml >= 11.0 (for GPU monitoring)
- gputil >= 1.4 (for additional GPU utilities)

### Development Dependencies
- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 21.0 (code formatting)
- flake8 >= 3.8 (linting)
- mypy >= 0.910 (type checking)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Support

For issues, feature requests, or questions:

1. Check the [documentation](README.md)
2. Search [existing issues](https://github.com/ambicuity/AI-Inference-Performance-Optimization-Tool/issues)
3. Create a [new issue](https://github.com/ambicuity/AI-Inference-Performance-Optimization-Tool/issues/new) with detailed information

## Changelog

### Version 1.0.0
- Initial release
- Core profiling functionality
- Resource monitoring with CPU, memory, and GPU support
- Statistical performance analysis
- Optimization recommendation engine
- Command-line interface
- Python API
- Comprehensive test suite
- Documentation and examples