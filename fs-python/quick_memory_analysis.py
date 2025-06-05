#!/usr/bin/env python3
"""Quick memory analysis for Fish Speech MPS inference"""

import gc
import psutil
import torch
import time
import sys
from pathlib import Path

# Add Fish Speech to path
sys.path.append(str(Path(__file__).parent / "fish-speech"))

def get_memory_info():
    """Get detailed memory information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    info = {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'system_used_percent': psutil.virtual_memory().percent,
    }
    
    if torch.backends.mps.is_available():
        try:
            info['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
            info['mps_reserved_mb'] = torch.mps.driver_allocated_memory() / 1024 / 1024
        except:
            info['mps_allocated_mb'] = 0
            info['mps_reserved_mb'] = 0
    
    return info

def analyze_memory_breakdown():
    """Quick analysis of memory usage breakdown"""
    print("🔬 Quick Memory Breakdown Analysis")
    print("=" * 50)
    
    baseline = get_memory_info()
    print(f"📊 Starting RAM: {baseline['rss_mb']:.1f}MB")
    
    steps = []
    
    # Step 1: Import Fish Speech
    print("\n1️⃣ Importing Fish Speech...")
    from fish_speech.models.text2semantic.llama import BaseTransformer
    from fish_speech.models.text2semantic.inference import init_model
    from huggingface_hub import snapshot_download
    
    after_import = get_memory_info()
    import_delta = after_import['rss_mb'] - baseline['rss_mb']
    print(f"   RAM after import: {after_import['rss_mb']:.1f}MB (+{import_delta:.1f}MB)")
    steps.append(("Import", import_delta))
    
    # Step 2: Get model path
    print("\n2️⃣ Getting model path...")
    checkpoints_dir = snapshot_download(
        repo_id="fishaudio/fish-speech-1.5", 
        repo_type="model"
    )
    
    after_download = get_memory_info()
    download_delta = after_download['rss_mb'] - after_import['rss_mb']
    print(f"   RAM after download: {after_download['rss_mb']:.1f}MB (+{download_delta:.1f}MB)")
    steps.append(("Download metadata", download_delta))
    
    # Step 3: Load model with different precisions
    precisions = [
        ("half (float16)", torch.half),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32)
    ]
    
    best_precision = None
    min_ram = float('inf')
    
    for name, dtype in precisions:
        print(f"\n3️⃣ Testing {name} precision...")
        
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        gc.collect()
        
        load_baseline = get_memory_info()
        
        try:
            model, decode_one_token = init_model(
                checkpoint_path=checkpoints_dir,
                device="mps",
                precision=dtype,
                compile=False
            )
            
            after_load = get_memory_info()
            load_delta = after_load['rss_mb'] - load_baseline['rss_mb']
            mps_usage = after_load['mps_allocated_mb']
            
            print(f"   RAM usage: {load_delta:.1f}MB")
            print(f"   MPS usage: {mps_usage:.1f}MB")
            print(f"   Total GPU memory reserved: {after_load['mps_reserved_mb']:.1f}MB")
            
            steps.append((f"Model load ({name})", load_delta))
            
            if load_delta < min_ram:
                min_ram = load_delta
                best_precision = name
            
            # Test cache setup
            print(f"   Setting up caches...")
            cache_baseline = get_memory_info()
            
            with torch.device("mps"):
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=model.config.max_seq_len,
                    dtype=dtype,
                )
            
            after_cache = get_memory_info()
            cache_delta = after_cache['rss_mb'] - cache_baseline['rss_mb']
            print(f"   Cache setup RAM: +{cache_delta:.1f}MB")
            print(f"   Total MPS after cache: {after_cache['mps_allocated_mb']:.1f}MB")
            
            steps.append((f"Cache setup ({name})", cache_delta))
            
            # Clean up
            del model, decode_one_token
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Final analysis
    print(f"\n📊 Memory Usage Summary:")
    print("=" * 40)
    for step_name, delta in steps:
        print(f"   {step_name:25}: {delta:6.1f}MB")
    
    total_overhead = sum(delta for _, delta in steps if "Model load" not in _)
    print(f"   {'=' * 25}   {'=' * 6}")
    print(f"   {'Base overhead':25}: {total_overhead:6.1f}MB")
    
    if best_precision:
        print(f"\n🏆 Best precision for RAM: {best_precision}")
        print(f"   Recommended: Use {best_precision} to minimize memory usage")
    
    return steps

def find_memory_issues():
    """Identify potential memory issues"""
    print("\n🕵️ Memory Issue Analysis")
    print("=" * 50)
    
    issues = []
    
    # Check system memory
    vm = psutil.virtual_memory()
    if vm.percent > 80:
        issues.append(f"High system memory usage: {vm.percent:.1f}%")
    
    # Check if MPS is using too much system RAM
    baseline = get_memory_info()
    
    print("Testing memory allocation patterns...")
    
    try:
        # Test large tensor allocation
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # Small test
        small_tensor = torch.randn(100, 100, device=device)
        after_small = get_memory_info()
        small_ram_delta = after_small['rss_mb'] - baseline['rss_mb']
        
        # Medium test  
        medium_tensor = torch.randn(1000, 1000, device=device)
        after_medium = get_memory_info()
        medium_ram_delta = after_medium['rss_mb'] - after_small['rss_mb']
        
        print(f"   Small tensor (100x100): {small_ram_delta:.1f}MB RAM")
        print(f"   Medium tensor (1000x1000): {medium_ram_delta:.1f}MB RAM")
        
        if medium_ram_delta > 100:  # If more than 100MB RAM for a simple tensor
            issues.append("Excessive RAM usage for GPU tensors - possible MPS fallback to CPU")
        
        # Clean up
        del small_tensor, medium_tensor
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
    except Exception as e:
        issues.append(f"Tensor allocation test failed: {e}")
    
    if issues:
        print(f"\n⚠️ Potential Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print(f"\n✅ No obvious memory issues detected")
    
    return issues

def create_optimized_cli():
    """Create memory-optimized CLI version"""
    print("\n⚡ Creating Memory-Optimized CLI")
    print("=" * 50)
    
    optimized_code = '''#!/usr/bin/env python3
"""Memory-optimized Fish Speech TTS CLI"""

import torch
import gc
import argparse
import sys
from pathlib import Path

def optimized_tts(text, output_path="output.wav", device="mps"):
    """Memory-optimized TTS with aggressive cleanup"""
    
    # Force cleanup before starting
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    try:
        # Import only when needed to reduce memory overhead
        sys.path.append(str(Path(__file__).parent / "fish-speech"))
        from fish_speech.models.text2semantic.inference import init_model, generate_long
        from huggingface_hub import snapshot_download
        
        print(f"🎯 Optimized synthesis: {text[:50]}...")
        
        # Get model with minimal caching
        checkpoints_dir = snapshot_download(
            repo_id="fishaudio/fish-speech-1.5",
            repo_type="model"
        )
        
        # Use half precision for MPS to reduce RAM usage
        precision = torch.half if device == "mps" else torch.bfloat16
        
        # Load model
        model, decode_one_token = init_model(
            checkpoint_path=checkpoints_dir,
            device=device,
            precision=precision,
            compile=False  # Disable compilation
        )
        
        # Smaller cache to reduce memory
        max_seq_len = min(2048, model.config.max_seq_len)
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=max_seq_len,
                dtype=precision,
            )
        
        # Generate with conservative settings
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=1024,  # Limit tokens
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            compile=False,
            iterative_prompt=True,
            max_length=1024,  # Reduce max length
            chunk_length=50,  # Smaller chunks
        )
        
        # Collect tokens
        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
        
        # Generate audio if we have tokens
        if codes:
            import numpy as np
            import subprocess
            import tempfile
            
            # Save tokens to temporary file
            codes_array = torch.cat(codes, dim=1).cpu().numpy()
            
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
                np.save(temp_file.name, codes_array)
                
                # Use VQGAN subprocess for audio generation
                vqgan_cmd = [
                    sys.executable, "-m", "fish_speech.models.vqgan.inference",
                    "--input-path", temp_file.name,
                    "--output-path", output_path,
                    "--config-name", "firefly_gan_vq",
                    "--checkpoint-path", str(Path(checkpoints_dir) / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
                    "--device", device
                ]
                
                result = subprocess.run(vqgan_cmd, capture_output=True, text=True, cwd="fish-speech")
                
                # Clean up temp file
                Path(temp_file.name).unlink()
                
                if result.returncode == 0:
                    print(f"✅ Audio saved: {output_path}")
                    return True
                else:
                    print(f"❌ VQGAN error: {result.stderr}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return False
        
    finally:
        # Aggressive cleanup
        if 'model' in locals():
            del model, decode_one_token
        if 'codes' in locals():
            del codes
        if 'codes_array' in locals():
            del codes_array
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Fish Speech TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output_optimized.wav", help="Output file")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Device")
    
    args = parser.parse_args()
    
    success = optimized_tts(args.text, args.output, args.device)
    
    if success and Path(args.output).exists():
        file_size = Path(args.output).stat().st_size
        print(f"🎵 Generated: {args.output} ({file_size} bytes)")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("optimized_cli.py", "w") as f:
        f.write(optimized_code)
    
    print("✅ Created optimized_cli.py")
    print("📝 Usage: python optimized_cli.py \"Your text here\" -o output.wav")

def main():
    print("🚀 Quick Fish Speech Memory Analysis")
    print("=" * 60)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return
    
    # Quick breakdown analysis
    memory_steps = analyze_memory_breakdown()
    
    # Find issues
    issues = find_memory_issues()
    
    # Create optimized version
    create_optimized_cli()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Calculate major memory consumers
    model_steps = [step for step in memory_steps if "Model load" in step[0]]
    cache_steps = [step for step in memory_steps if "Cache setup" in step[0]]
    
    if model_steps:
        min_model_ram = min(step[1] for step in model_steps)
        max_model_ram = max(step[1] for step in model_steps)
        print(f"🧠 Model loading RAM: {min_model_ram:.1f}MB - {max_model_ram:.1f}MB")
    
    if cache_steps:
        avg_cache_ram = sum(step[1] for step in cache_steps) / len(cache_steps)
        print(f"💾 Cache setup RAM: ~{avg_cache_ram:.1f}MB")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Use half (float16) precision for minimal RAM usage")
    print(f"   2. Use optimized_cli.py for memory-efficient synthesis")
    print(f"   3. Consider reducing max_seq_len if memory is critical")
    
    if issues:
        print(f"\n⚠️ Issues to address:")
        for issue in issues:
            print(f"   • {issue}")

if __name__ == "__main__":
    main() 