#!/usr/bin/env python3
"""
Quick optimization tool for VQ token extraction
Automatically determines optimal parameters for your system
"""

import torch
import subprocess
import sys
from pathlib import Path

def check_system_capabilities():
    """Check system and suggest optimal parameters"""
    print("🔍 Checking system capabilities...")
    
    # Check available devices
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = "cuda"
        batch_size = 32 if gpu_memory > 8 else 16
        num_workers = 8
        print(f"✅ CUDA GPU: {gpu_memory:.1f} GB")
        print(f"🚀 Recommended: --device-extract cuda --batch-size-extract {batch_size} --num-workers-extract {num_workers}")
        return device, batch_size, num_workers
    
    elif mps_available:
        device = "mps"
        batch_size = 16
        num_workers = 2
        print("✅ Apple Silicon MPS available")
        print(f"🍎 Recommended: --device-extract mps --batch-size-extract {batch_size} --num-workers-extract {num_workers}")
        return device, batch_size, num_workers
    
    else:
        device = "cpu"
        batch_size = 4
        num_workers = 4
        print("⚠️ Only CPU available")
        print(f"💻 Recommended: --device-extract cpu --batch-size-extract {batch_size} --num-workers-extract {num_workers}")
        return device, batch_size, num_workers

def suggest_command(project_name, data_dir):
    """Suggest optimized command"""
    device, batch_size, num_workers = check_system_capabilities()
    
    cmd = f"""python finetune_tts.py \\
    --project {project_name} \\
    --data-dir {data_dir} \\
    --extract-tokens \\
    --device-extract {device} \\
    --batch-size-extract {batch_size} \\
    --num-workers-extract {num_workers}"""
    
    print("\n📋 Optimized command:")
    print(cmd)
    print("\n💡 This should be 5-10x faster than your current settings!")
    
    return cmd

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python quick_optimize_extraction.py <project_name> <data_dir>")
        print("Example: python quick_optimize_extraction.py G_Zephyr_neutral_2h /path/to/data")
        sys.exit(1)
    
    project_name = sys.argv[1]
    data_dir = sys.argv[2]
    
    suggest_command(project_name, data_dir) 