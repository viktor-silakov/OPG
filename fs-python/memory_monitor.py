#!/usr/bin/env python3
"""Real-time memory monitoring during Fish Speech inference"""

import psutil
import torch
import time
import threading
import subprocess
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from cli_tts import setup_fish_speech, synthesize_speech_cli


class MemoryMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.process = psutil.Process()
        
    def start_monitoring(self):
        self.monitoring = True
        self.data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        return self.data
        
    def _monitor_loop(self):
        start_time = time.time()
        while self.monitoring:
            current_time = time.time() - start_time
            memory_info = self.process.memory_info()
            
            data_point = {
                'time': current_time,
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': self.process.cpu_percent(),
                'system_memory_percent': psutil.virtual_memory().percent
            }
            
            # Add MPS memory if available
            if torch.backends.mps.is_available():
                try:
                    data_point['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
                    data_point['mps_reserved_mb'] = torch.mps.driver_allocated_memory() / 1024 / 1024
                except:
                    data_point['mps_allocated_mb'] = 0
                    data_point['mps_reserved_mb'] = 0
            
            self.data.append(data_point)
            time.sleep(self.interval)


def benchmark_with_monitoring(device, text):
    print(f"\n🔍 Monitoring {device.upper()} inference...")
    
    monitor = MemoryMonitor(interval=0.5)  # Monitor every 0.5 seconds
    
    # Clear caches
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("⚙️ Setting up model...")
    fish_speech_dir, checkpoints_dir = setup_fish_speech("1.5")
    
    print("🎯 Starting monitoring and synthesis...")
    monitor.start_monitoring()
    
    start_time = time.time()
    success = synthesize_speech_cli(
        text=text,
        output_path=f"output/monitor_{device}.wav",
        device=device,
        checkpoints_dir=checkpoints_dir,
        use_cache=False
    )
    end_time = time.time()
    
    data = monitor.stop_monitoring()
    
    if not success:
        print(f"❌ Synthesis failed")
        return None
    
    print(f"✅ Synthesis completed in {end_time - start_time:.2f}s")
    
    # Analyze data
    if data:
        rss_values = [d['rss_mb'] for d in data]
        mps_values = [d.get('mps_allocated_mb', 0) for d in data]
        cpu_values = [d['cpu_percent'] for d in data]
        
        print(f"\n📊 Memory Analysis:")
        print(f"   RSS Memory - Min: {min(rss_values):.1f}MB, Max: {max(rss_values):.1f}MB, Avg: {sum(rss_values)/len(rss_values):.1f}MB")
        print(f"   Memory Delta: {max(rss_values) - min(rss_values):.1f}MB")
        
        if any(mps_values):
            print(f"   MPS Memory - Min: {min(mps_values):.1f}MB, Max: {max(mps_values):.1f}MB, Avg: {sum(mps_values)/len(mps_values):.1f}MB")
            print(f"   MPS Delta: {max(mps_values) - min(mps_values):.1f}MB")
        else:
            print(f"   MPS Memory: No GPU memory usage detected")
        
        print(f"   CPU Usage - Max: {max(cpu_values):.1f}%, Avg: {sum(cpu_values)/len(cpu_values):.1f}%")
        
    return {
        'device': device,
        'synthesis_time': end_time - start_time,
        'success': success,
        'monitoring_data': data
    }


def get_system_memory_info():
    """Get detailed system memory information"""
    vm = psutil.virtual_memory()
    print(f"\n💻 System Memory Info:")
    print(f"   Total RAM: {vm.total / 1024**3:.1f}GB")
    print(f"   Available: {vm.available / 1024**3:.1f}GB ({vm.percent:.1f}% used)")
    print(f"   Used: {vm.used / 1024**3:.1f}GB")
    
    # Check if we can get GPU memory info
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Memory' in line and 'MB' in line:
                    print(f"   GPU Memory: {line.strip()}")
                    break
    except:
        pass


def main():
    print("🔬 Fish Speech Memory Monitoring")
    print("=" * 50)
    
    get_system_memory_info()
    
    # Test text
    text = "Короткая тестовая фраза для проверки использования памяти."
    print(f"\n📝 Test text: {text}")
    
    results = {}
    
    # Test CPU
    print("\n" + "="*30 + " CPU TEST " + "="*30)
    results['cpu'] = benchmark_with_monitoring('cpu', text)
    
    # Wait between tests
    time.sleep(3)
    
    # Test MPS
    if torch.backends.mps.is_available():
        print("\n" + "="*30 + " MPS TEST " + "="*30)
        results['mps'] = benchmark_with_monitoring('mps', text)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DETAILED MEMORY COMPARISON")
    print("=" * 70)
    
    if results.get('cpu') and results.get('mps'):
        cpu_data = results['cpu']['monitoring_data']
        mps_data = results['mps']['monitoring_data']
        
        cpu_peak_rss = max(d['rss_mb'] for d in cpu_data)
        mps_peak_rss = max(d['rss_mb'] for d in mps_data)
        
        cpu_peak_mps = max(d.get('mps_allocated_mb', 0) for d in cpu_data)
        mps_peak_mps = max(d.get('mps_allocated_mb', 0) for d in mps_data)
        
        print(f"\n⏱️ Performance:")
        print(f"   CPU time: {results['cpu']['synthesis_time']:.2f}s")
        print(f"   MPS time: {results['mps']['synthesis_time']:.2f}s")
        speedup = results['cpu']['synthesis_time'] / results['mps']['synthesis_time']
        print(f"   Speedup: {speedup:.2f}x")
        
        print(f"\n🧠 Peak Memory Usage:")
        print(f"   CPU RSS: {cpu_peak_rss:.1f}MB")
        print(f"   MPS RSS: {mps_peak_rss:.1f}MB")
        print(f"   Difference: {mps_peak_rss - cpu_peak_rss:+.1f}MB")
        
        print(f"\n🎯 GPU Memory Usage:")
        print(f"   CPU mode MPS: {cpu_peak_mps:.1f}MB")
        print(f"   MPS mode MPS: {mps_peak_mps:.1f}MB")
        
        if mps_peak_mps > 0:
            print(f"   ✅ MPS GPU memory is being used!")
        else:
            print(f"   ⚠️ No significant MPS GPU memory usage detected")


if __name__ == "__main__":
    main() 