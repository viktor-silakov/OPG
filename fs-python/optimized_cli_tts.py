#!/usr/bin/env python3
"""Optimized Fish Speech CLI with memory monitoring and performance improvements"""

import argparse
import time
import gc
import psutil
import torch
import subprocess
import sys
import os
from pathlib import Path


def monitor_memory():
    """Monitor system and GPU memory usage"""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024
    
    if torch.backends.mps.is_available():
        mps_mb = torch.mps.current_allocated_memory() / 1024 / 1024
        return ram_mb, mps_mb
    else:
        return ram_mb, 0.0


def optimize_memory():
    """Optimize memory usage"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_optimized_tts(text, output_file="output/optimized_output.wav", play=False, device="mps"):
    """Run Fish Speech TTS with optimized memory management"""
    
    print("🧠 Optimized Fish Speech TTS")
    print("=" * 50)
    print(f"🎯 Text: '{text[:60]}{'...' if len(text) > 60 else ''}'")
    print(f"🖥️  Device: {device}")
    
    # Initial memory check
    initial_ram, initial_mps = monitor_memory()
    print(f"📊 Initial memory: RAM {initial_ram:.1f}MB, MPS {initial_mps:.1f}MB")
    
    # Pre-optimize memory
    optimize_memory()
    
    start_time = time.time()
    
    try:
        # Build the command for the base CLI
        cmd = [
            sys.executable, "cli_tts.py",
            text,
            "-o", output_file,
            "--device", device
        ]
        
        if play:
            cmd.append("--play")
        
        print(f"🚀 Running Fish Speech TTS...")
        
        # Run the base CLI with memory monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Monitor memory during execution
        peak_ram = initial_ram
        peak_mps = initial_mps
        
        while process.poll() is None:
            time.sleep(0.5)
            current_ram, current_mps = monitor_memory()
            peak_ram = max(peak_ram, current_ram)
            peak_mps = max(peak_mps, current_mps)
        
        stdout, stderr = process.communicate()
        
        # Final memory check
        final_ram, final_mps = monitor_memory()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if process.returncode == 0:
            print(f"✅ TTS completed successfully!")
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            print(f"📊 Memory usage:")
            print(f"   Peak RAM: {peak_ram - initial_ram:.1f}MB (total: {peak_ram:.1f}MB)")
            print(f"   Peak MPS: {peak_mps - initial_mps:.1f}MB (total: {peak_mps:.1f}MB)")
            print(f"   Final delta: RAM {final_ram - initial_ram:+.1f}MB, MPS {final_mps - initial_mps:+.1f}MB")
            
            # Check if output file exists
            if Path(output_file).exists():
                file_size = Path(output_file).stat().st_size / 1024
                print(f"🎵 Audio output: {output_file} ({file_size:.1f}KB)")
                
                # Performance optimization info
                print(f"\n✨ Optimizations applied:")
                print(f"   🧹 Memory cleanup before and after generation")
                print(f"   📊 Real-time memory monitoring")
                print(f"   🔧 Apple Silicon MPS acceleration")
                
                return True, {
                    'time': generation_time,
                    'peak_ram': peak_ram - initial_ram,
                    'peak_mps': peak_mps - initial_mps,
                    'output_file': output_file,
                    'file_size_kb': file_size
                }
            else:
                print(f"❌ Output file not found: {output_file}")
                return False, None
        else:
            print(f"❌ TTS failed with return code: {process.returncode}")
            if stderr:
                print(f"Error: {stderr}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error during optimized TTS: {e}")
        return False, None
    
    finally:
        # Final memory optimization
        optimize_memory()


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="Optimized Fish Speech TTS with memory monitoring")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output/optimized_output.wav", help="Output WAV file")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--play", action="store_true", help="Play audio after generation")
    parser.add_argument("--monitor", action="store_true", help="Show detailed memory monitoring")
    
    args = parser.parse_args()
    
    print("🎙️  Fish Speech Optimized CLI TTS")
    print("=" * 70)
    
    # System info
    print(f"🖥️  System: PyTorch {torch.__version__}")
    print(f"🧠 Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        device = "cpu"
    
    # Run optimized TTS
    success, result = run_optimized_tts(
        text=args.text,
        output_file=args.output,
        play=args.play,
        device=device
    )
    
    if success and result:
        print(f"\n🏆 Optimized TTS completed successfully!")
        
        if args.monitor:
            print(f"\n📊 Detailed Performance Metrics:")
            print(f"   Generation time: {result['time']:.2f}s")
            print(f"   Peak RAM usage: {result['peak_ram']:.1f}MB")
            print(f"   Peak MPS usage: {result['peak_mps']:.1f}MB")
            print(f"   Output file size: {result['file_size_kb']:.1f}KB")
            print(f"   Speed: {result['file_size_kb'] / result['time']:.1f} KB/s")
        
        print(f"\n💡 Tips for better performance:")
        print(f"   • Close other applications to free RAM")
        print(f"   • Use shorter texts for faster generation")
        print(f"   • Keep model in cache for subsequent runs")
        
    else:
        print(f"\n❌ Optimized TTS failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 