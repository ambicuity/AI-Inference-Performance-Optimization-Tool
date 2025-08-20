"""
Resource monitoring module for tracking system resources during AI inference.

This module provides the ResourceMonitor class that tracks CPU, memory, 
and optionally GPU usage during inference profiling.
"""

import time
import threading
import psutil
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass, field
import statistics

try:
    import pynvml
    NVIDIA_GPU_AVAILABLE = True
except ImportError:
    NVIDIA_GPU_AVAILABLE = False
    pynvml = None


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None


@dataclass
class ResourceStats:
    """Aggregated resource statistics over a monitoring period"""
    cpu_avg: float
    cpu_max: float
    cpu_min: float
    memory_avg: float
    memory_max: float
    memory_min: float
    memory_used_avg_mb: float
    memory_used_max_mb: float
    gpu_util_avg: Optional[float] = None
    gpu_util_max: Optional[float] = None
    gpu_memory_used_avg_mb: Optional[float] = None
    gpu_memory_used_max_mb: Optional[float] = None
    gpu_temp_avg: Optional[float] = None
    gpu_temp_max: Optional[float] = None
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    
    @classmethod
    def from_snapshots(cls, snapshots: List[ResourceSnapshot]) -> 'ResourceStats':
        """Create ResourceStats from a list of snapshots"""
        if not snapshots:
            return cls(
                cpu_avg=0, cpu_max=0, cpu_min=0,
                memory_avg=0, memory_max=0, memory_min=0,
                memory_used_avg_mb=0, memory_used_max_mb=0
            )
            
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        memory_used_values = [s.memory_used_mb for s in snapshots]
        
        gpu_util_values = [s.gpu_utilization for s in snapshots if s.gpu_utilization is not None]
        gpu_memory_values = [s.gpu_memory_used_mb for s in snapshots if s.gpu_memory_used_mb is not None]
        gpu_temp_values = [s.gpu_temperature for s in snapshots if s.gpu_temperature is not None]
        
        return cls(
            cpu_avg=statistics.mean(cpu_values),
            cpu_max=max(cpu_values),
            cpu_min=min(cpu_values),
            memory_avg=statistics.mean(memory_values),
            memory_max=max(memory_values),
            memory_min=min(memory_values),
            memory_used_avg_mb=statistics.mean(memory_used_values),
            memory_used_max_mb=max(memory_used_values),
            gpu_util_avg=statistics.mean(gpu_util_values) if gpu_util_values else None,
            gpu_util_max=max(gpu_util_values) if gpu_util_values else None,
            gpu_memory_used_avg_mb=statistics.mean(gpu_memory_values) if gpu_memory_values else None,
            gpu_memory_used_max_mb=max(gpu_memory_values) if gpu_memory_values else None,
            gpu_temp_avg=statistics.mean(gpu_temp_values) if gpu_temp_values else None,
            gpu_temp_max=max(gpu_temp_values) if gpu_temp_values else None,
            snapshots=snapshots
        )


class ResourceMonitor:
    """
    Monitor system resources during AI inference profiling.
    
    This class tracks CPU, memory, and optionally GPU usage in real-time
    during inference operations to help identify resource bottlenecks.
    """
    
    def __init__(self, sample_interval: float = 0.1, monitor_gpu: bool = True):
        """
        Initialize the resource monitor.
        
        Args:
            sample_interval: Time interval between resource samples (seconds)
            monitor_gpu: Whether to monitor GPU resources (requires pynvml)
        """
        self.sample_interval = sample_interval
        self.monitor_gpu = monitor_gpu and NVIDIA_GPU_AVAILABLE
        self.snapshots: List[ResourceSnapshot] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        if self.monitor_gpu:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use first GPU
                print("GPU monitoring enabled")
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self.monitor_gpu = False
        
    def start_monitoring(self):
        """Start monitoring resources in a background thread"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self.snapshots.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> ResourceStats:
        """
        Stop monitoring and return aggregated statistics.
        
        Returns:
            ResourceStats with aggregated resource usage statistics
        """
        if not self._monitoring:
            return ResourceStats.from_snapshots([])
            
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            
        return ResourceStats.from_snapshots(self.snapshots)
        
    def _monitor_loop(self):
        """Main monitoring loop running in background thread"""
        while self._monitoring:
            snapshot = self._take_snapshot()
            if snapshot:
                self.snapshots.append(snapshot)
            time.sleep(self.sample_interval)
            
    def _take_snapshot(self) -> Optional[ResourceSnapshot]:
        """Take a snapshot of current resource usage"""
        try:
            # Get basic system info
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024
            )
            
            # Add GPU info if available
            if self.monitor_gpu:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(
                        self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    
                    snapshot.gpu_utilization = gpu_util.gpu
                    snapshot.gpu_memory_used_mb = gpu_memory.used / 1024 / 1024
                    snapshot.gpu_memory_total_mb = gpu_memory.total / 1024 / 1024
                    snapshot.gpu_temperature = gpu_temp
                    
                except Exception as e:
                    # GPU monitoring failed, but continue with CPU/memory
                    pass
                    
            return snapshot
            
        except Exception as e:
            print(f"Warning: Failed to take resource snapshot: {e}")
            return None
            
    def get_current_usage(self) -> Optional[ResourceSnapshot]:
        """Get current resource usage without starting monitoring"""
        return self._take_snapshot()
        
    def detect_bottlenecks(self, stats: ResourceStats) -> List[str]:
        """
        Detect potential resource bottlenecks from monitoring statistics.
        
        Args:
            stats: ResourceStats from a monitoring session
            
        Returns:
            List of detected bottleneck descriptions
        """
        bottlenecks = []
        
        # CPU bottleneck detection
        if stats.cpu_avg > 90:
            bottlenecks.append(f"High CPU usage: {stats.cpu_avg:.1f}% average, {stats.cpu_max:.1f}% peak")
        elif stats.cpu_avg > 70:
            bottlenecks.append(f"Moderate CPU usage: {stats.cpu_avg:.1f}% average")
            
        # Memory bottleneck detection  
        if stats.memory_avg > 90:
            bottlenecks.append(f"High memory usage: {stats.memory_avg:.1f}% average, {stats.memory_max:.1f}% peak")
        elif stats.memory_avg > 75:
            bottlenecks.append(f"Moderate memory usage: {stats.memory_avg:.1f}% average")
            
        # GPU bottleneck detection
        if stats.gpu_util_avg is not None:
            if stats.gpu_util_avg > 90:
                bottlenecks.append(f"High GPU utilization: {stats.gpu_util_avg:.1f}% average")
            elif stats.gpu_util_avg < 30:
                bottlenecks.append(f"Low GPU utilization: {stats.gpu_util_avg:.1f}% average (potential underutilization)")
                
        if stats.gpu_memory_used_avg_mb is not None and stats.gpu_memory_used_max_mb is not None:
            gpu_memory_percent = (stats.gpu_memory_used_max_mb / 
                                (stats.snapshots[0].gpu_memory_total_mb or 1)) * 100
            if gpu_memory_percent > 90:
                bottlenecks.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
                
        if stats.gpu_temp_avg is not None:
            if stats.gpu_temp_avg > 80:
                bottlenecks.append(f"High GPU temperature: {stats.gpu_temp_avg:.1f}°C average")
                
        return bottlenecks
        
    def get_resource_recommendations(self, stats: ResourceStats) -> List[str]:
        """
        Get optimization recommendations based on resource usage patterns.
        
        Args:
            stats: ResourceStats from a monitoring session
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # CPU recommendations
        if stats.cpu_avg > 85:
            recommendations.append("Consider reducing batch size or concurrent requests to lower CPU usage")
            recommendations.append("Optimize model architecture for better CPU efficiency")
        elif stats.cpu_avg < 30:
            recommendations.append("CPU is underutilized - consider increasing batch size or concurrent requests")
            
        # Memory recommendations
        if stats.memory_avg > 85:
            recommendations.append("High memory usage detected - consider reducing batch size")
            recommendations.append("Implement memory-efficient data loading techniques")
        
        # GPU recommendations
        if stats.gpu_util_avg is not None:
            if stats.gpu_util_avg > 85:
                recommendations.append("GPU is highly utilized - performance is likely GPU-bound")
            elif stats.gpu_util_avg < 40:
                recommendations.append("GPU is underutilized - consider increasing batch size")
                recommendations.append("Verify that operations are running on GPU, not CPU")
                
        if stats.gpu_memory_used_avg_mb is not None:
            total_gpu_memory = stats.snapshots[0].gpu_memory_total_mb if stats.snapshots else 0
            if total_gpu_memory > 0:
                gpu_memory_percent = (stats.gpu_memory_used_max_mb / total_gpu_memory) * 100
                if gpu_memory_percent > 85:
                    recommendations.append("High GPU memory usage - consider reducing batch size")
                elif gpu_memory_percent < 30:
                    recommendations.append("GPU memory is underutilized - consider increasing batch size")
                    
        return recommendations


class ContextResourceMonitor:
    """Context manager for convenient resource monitoring"""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.stats: Optional[ResourceStats] = None
        
    def __enter__(self):
        self.monitor.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats = self.monitor.stop_monitoring()
        
    def get_stats(self) -> Optional[ResourceStats]:
        return self.stats