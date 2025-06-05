#!/usr/bin/env python3
"""Memory usage benchmark for Fish Speech TTS on MPS vs CPU"""

import psutil
import torch
import time
import subprocess
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

from cli_tts import setup_fish_speech, synthesize_speech_cli


def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    stats = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'system_available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'system_used_percent': psutil.virtual_memory().percent
    }
    
    # Add MPS memory if available
    if torch.backends.mps.is_available():
        try:
            stats['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
            stats['mps_reserved_mb'] = torch.mps.driver_allocated_memory() / 1024 / 1024
        except:
            stats['mps_allocated_mb'] = 0
            stats['mps_reserved_mb'] = 0
    
    return stats


def benchmark_device(device_name, text, model_version="1.5"):
    """Benchmark memory usage for specific device"""
    print(f"\n🔍 Benchmarking {device_name.upper()} memory usage...")
    
    # Clear MPS cache if available
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Measure baseline memory
    baseline = get_memory_usage()
    print(f"📊 Baseline memory - RSS: {baseline['rss_mb']:.1f}MB, "
          f"System: {baseline['system_used_percent']:.1f}%")
    
    if 'mps_allocated_mb' in baseline:
        print(f"📊 MPS baseline - Allocated: {baseline['mps_allocated_mb']:.1f}MB, "
              f"Reserved: {baseline['mps_reserved_mb']:.1f}MB")
    
    # Setup model
    print(f"⚙️ Setting up Fish Speech model...")
    start_time = time.time()
    fish_speech_dir, checkpoints_dir = setup_fish_speech(model_version)
    setup_time = time.time() - start_time
    
    after_setup = get_memory_usage()
    setup_memory_delta = after_setup['rss_mb'] - baseline['rss_mb']
    print(f"📈 After model setup - RSS: +{setup_memory_delta:.1f}MB, "
          f"Total: {after_setup['rss_mb']:.1f}MB")
    
    if 'mps_allocated_mb' in after_setup:
        mps_setup_delta = after_setup['mps_allocated_mb'] - baseline['mps_allocated_mb']
        print(f"📈 MPS after setup - Allocated: +{mps_setup_delta:.1f}MB, "
              f"Total: {after_setup['mps_allocated_mb']:.1f}MB")
    
    # Perform synthesis
    print(f"🎯 Running synthesis on {device_name}...")
    synth_start = time.time()
    
    output_file = f"output/memory_test_{device_name}.wav"
    success = synthesize_speech_cli(
        text=text,
        output_path=output_file,
        device=device_name,
        checkpoints_dir=checkpoints_dir,
        use_cache=False  # No cache for fair comparison
    )
    
    synth_time = time.time() - synth_start
    
    if not success:
        print(f"❌ Synthesis failed on {device_name}")
        return None
    
    # Measure peak memory
    peak_memory = get_memory_usage()
    synthesis_memory_delta = peak_memory['rss_mb'] - after_setup['rss_mb']
    total_memory_delta = peak_memory['rss_mb'] - baseline['rss_mb']
    
    print(f"📊 Peak memory usage:")
    print(f"   Setup time: {setup_time:.2f}s")
    print(f"   Synthesis time: {synth_time:.2f}s")
    print(f"   RSS memory - Setup: +{setup_memory_delta:.1f}MB, "
          f"Synthesis: +{synthesis_memory_delta:.1f}MB, "
          f"Total: +{total_memory_delta:.1f}MB")
    print(f"   Peak RSS: {peak_memory['rss_mb']:.1f}MB")
    print(f"   System memory: {peak_memory['system_used_percent']:.1f}%")
    
    result = {
        'device': device_name,
        'setup_time': setup_time,
        'synthesis_time': synth_time,
        'baseline_rss': baseline['rss_mb'],
        'peak_rss': peak_memory['rss_mb'],
        'setup_memory_delta': setup_memory_delta,
        'synthesis_memory_delta': synthesis_memory_delta,
        'total_memory_delta': total_memory_delta,
        'system_memory_percent': peak_memory['system_used_percent'],
        'success': success
    }
    
    if 'mps_allocated_mb' in peak_memory:
        mps_synth_delta = peak_memory['mps_allocated_mb'] - after_setup['mps_allocated_mb']
        mps_total_delta = peak_memory['mps_allocated_mb'] - baseline['mps_allocated_mb']
        
        print(f"   MPS memory - Setup: +{mps_setup_delta:.1f}MB, "
              f"Synthesis: +{mps_synth_delta:.1f}MB, "
              f"Total: +{mps_total_delta:.1f}MB")
        print(f"   Peak MPS allocated: {peak_memory['mps_allocated_mb']:.1f}MB")
        print(f"   Peak MPS reserved: {peak_memory['mps_reserved_mb']:.1f}MB")
        
        result.update({
            'mps_baseline_allocated': baseline['mps_allocated_mb'],
            'mps_peak_allocated': peak_memory['mps_allocated_mb'],
            'mps_peak_reserved': peak_memory['mps_reserved_mb'],
            'mps_setup_delta': mps_setup_delta,
            'mps_synthesis_delta': mps_synth_delta,
            'mps_total_delta': mps_total_delta
        })
    
    # Clean up
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return result


def main():
    print("🧪 Fish Speech Memory Usage Benchmark")
    print("=" * 50)
    
    # Test text
    test_text = ("Современные технологии искусственного интеллекта позволяют создавать "
                "высококачественные синтезированные голоса, которые практически неотличимы "
                "от человеческой речи, что открывает новые возможности для образования, "
                "развлечений и бизнеса в цифровую эпоху.")
    
    print(f"📝 Test text: {test_text[:100]}...")
    print(f"📏 Text length: {len(test_text)} characters")
    
    # Check system info
    print(f"\n💻 System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"   MPS available: {torch.backends.mps.is_available()}")
    print(f"   PyTorch version: {torch.__version__}")
    
    results = {}
    
    # Benchmark CPU
    results['cpu'] = benchmark_device('cpu', test_text)
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Benchmark MPS (if available)
    if torch.backends.mps.is_available():
        results['mps'] = benchmark_device('mps', test_text)
    else:
        print("\n⚠️ MPS not available, skipping MPS benchmark")
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("📋 MEMORY USAGE COMPARISON")
    print("=" * 50)
    
    if results['cpu'] and results.get('mps'):
        cpu_r = results['cpu']
        mps_r = results['mps']
        
        print(f"\n⏱️ Performance:")
        print(f"   CPU synthesis time: {cpu_r['synthesis_time']:.2f}s")
        print(f"   MPS synthesis time: {mps_r['synthesis_time']:.2f}s")
        speedup = cpu_r['synthesis_time'] / mps_r['synthesis_time']
        print(f"   MPS speedup: {speedup:.2f}x")
        
        print(f"\n🧠 System Memory (RSS):")
        print(f"   CPU peak: {cpu_r['peak_rss']:.1f}MB")
        print(f"   MPS peak: {mps_r['peak_rss']:.1f}MB")
        rss_diff = mps_r['peak_rss'] - cpu_r['peak_rss']
        print(f"   MPS vs CPU: {rss_diff:+.1f}MB ({rss_diff/cpu_r['peak_rss']*100:+.1f}%)")
        
        print(f"\n📈 Memory Delta (from baseline):")
        print(f"   CPU total: +{cpu_r['total_memory_delta']:.1f}MB")
        print(f"   MPS total: +{mps_r['total_memory_delta']:.1f}MB")
        
        if 'mps_peak_allocated' in mps_r:
            print(f"\n🎯 MPS-specific Memory:")
            print(f"   MPS allocated: {mps_r['mps_peak_allocated']:.1f}MB")
            print(f"   MPS reserved: {mps_r['mps_peak_reserved']:.1f}MB")
            print(f"   MPS delta: +{mps_r['mps_total_delta']:.1f}MB")
        
        print(f"\n💡 Efficiency (Memory per second):")
        cpu_eff = cpu_r['total_memory_delta'] / cpu_r['synthesis_time']
        mps_eff = mps_r['total_memory_delta'] / mps_r['synthesis_time']
        print(f"   CPU: {cpu_eff:.1f}MB/s")
        print(f"   MPS: {mps_eff:.1f}MB/s")
        eff_ratio = cpu_eff / mps_eff if mps_eff > 0 else 0
        print(f"   MPS efficiency gain: {eff_ratio:.2f}x")
    
    print(f"\n✅ Benchmark completed! Check output files:")
    if results['cpu']:
        print(f"   CPU: output/memory_test_cpu.wav")
    if results.get('mps'):
        print(f"   MPS: output/memory_test_mps.wav")


if __name__ == "__main__":
    main() 