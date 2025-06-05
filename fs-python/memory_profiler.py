#!/usr/bin/env python3
"""Advanced memory profiler for Fish Speech MPS inference"""

import gc
import psutil
import torch
import time
import tracemalloc
import sys
from pathlib import Path
import subprocess
import threading
from collections import defaultdict

# Add Fish Speech to path
sys.path.append(str(Path(__file__).parent / "fish-speech"))

def get_memory_info():
    """Get detailed memory information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    info = {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'shared_mb': getattr(memory_info, 'shared', 0) / 1024 / 1024,
        'system_available_mb': psutil.virtual_memory().available / 1024 / 1024,
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

def profile_subprocess_memory(text_length="short"):
    """Profile memory usage of subprocess-based TTS"""
    print(f"\n🔍 Profiling subprocess memory usage - {text_length} text")
    print("=" * 60)
    
    # Test texts of different lengths
    test_texts = {
        "short": "Короткий тест для проверки памяти.",
        "medium": "Это средний по длине текст для тестирования использования памяти при синтезе речи. Здесь два предложения для более детального анализа.",
        "long": "Это длинный текст для тестирования использования памяти при обработке больших объемов данных в системе синтеза речи Fish Speech на Apple Silicon. Второе предложение содержит дополнительную информацию о технических характеристиках и особенностях работы нейронной сети."
    }
    
    text = test_texts[text_length]
    print(f"📝 Text length: {len(text)} characters")
    
    baseline = get_memory_info()
    print(f"📊 Baseline RSS: {baseline['rss_mb']:.1f}MB")
    
    # Clear any caches
    subprocess.run([sys.executable, "-c", 
                   "import sys; sys.path.append('./fish-speech'); "
                   "from fish_speech.models.text2semantic.inference import *; "
                   "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None"],
                  capture_output=True)
    
    # Monitor memory during subprocess execution
    memory_samples = []
    monitoring = True
    
    def memory_monitor():
        while monitoring:
            memory_samples.append(get_memory_info())
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=memory_monitor)
    monitor_thread.start()
    
    start_time = time.time()
    
    # Run TTS subprocess
    from cli_tts import setup_fish_speech, synthesize_speech_cli
    fish_speech_dir, checkpoints_dir = setup_fish_speech("1.5")
    
    success = synthesize_speech_cli(
        text=text,
        output_path=f"output/memory_test_{text_length}.wav",
        device="mps",
        checkpoints_dir=checkpoints_dir,
        use_cache=False
    )
    
    end_time = time.time()
    monitoring = False
    monitor_thread.join()
    
    if not success:
        print("❌ Synthesis failed")
        return None
    
    # Analyze memory usage
    peak_memory = max(memory_samples, key=lambda x: x['rss_mb'])
    synthesis_time = end_time - start_time
    
    print(f"\n📈 Memory Analysis:")
    print(f"   Synthesis time: {synthesis_time:.2f}s")
    print(f"   Peak RSS: {peak_memory['rss_mb']:.1f}MB")
    print(f"   Memory delta: {peak_memory['rss_mb'] - baseline['rss_mb']:.1f}MB")
    print(f"   Peak MPS: {peak_memory['mps_allocated_mb']:.1f}MB")
    print(f"   MPS delta: {peak_memory['mps_allocated_mb'] - baseline['mps_allocated_mb']:.1f}MB")
    
    return {
        'text_length': text_length,
        'char_count': len(text),
        'synthesis_time': synthesis_time,
        'baseline_rss': baseline['rss_mb'],
        'peak_rss': peak_memory['rss_mb'],
        'memory_delta': peak_memory['rss_mb'] - baseline['rss_mb'],
        'peak_mps': peak_memory['mps_allocated_mb'],
        'samples': memory_samples
    }

def analyze_memory_components():
    """Analyze what components use the most memory"""
    print("\n🔬 Memory Component Analysis")
    print("=" * 60)
    
    tracemalloc.start()
    baseline = get_memory_info()
    
    print(f"📊 Starting memory: {baseline['rss_mb']:.1f}MB")
    
    # Step 1: Import Fish Speech
    print("\n1️⃣ Importing Fish Speech modules...")
    import_start = time.time()
    
    from fish_speech.models.text2semantic.llama import BaseTransformer
    from fish_speech.models.text2semantic.inference import init_model
    from huggingface_hub import snapshot_download
    
    after_import = get_memory_info()
    import_time = time.time() - import_start
    
    print(f"   Import time: {import_time:.2f}s")
    print(f"   Memory after import: {after_import['rss_mb']:.1f}MB (+{after_import['rss_mb'] - baseline['rss_mb']:.1f}MB)")
    
    # Step 2: Download model metadata
    print("\n2️⃣ Downloading model metadata...")
    download_start = time.time()
    
    checkpoints_dir = snapshot_download(
        repo_id="fishaudio/fish-speech-1.5",
        repo_type="model"
    )
    
    after_download = get_memory_info()
    download_time = time.time() - download_start
    
    print(f"   Download time: {download_time:.2f}s")
    print(f"   Memory after download: {after_download['rss_mb']:.1f}MB (+{after_download['rss_mb'] - after_import['rss_mb']:.1f}MB)")
    
    # Step 3: Load model
    print("\n3️⃣ Loading model...")
    load_start = time.time()
    
    torch.mps.empty_cache()
    model, decode_one_token = init_model(
        checkpoint_path=checkpoints_dir,
        device="mps",
        precision=torch.bfloat16,
        compile=False
    )
    
    after_load = get_memory_info()
    load_time = time.time() - load_start
    
    print(f"   Load time: {load_time:.2f}s")
    print(f"   Memory after load: {after_load['rss_mb']:.1f}MB (+{after_load['rss_mb'] - after_download['rss_mb']:.1f}MB)")
    print(f"   MPS memory: {after_load['mps_allocated_mb']:.1f}MB")
    
    # Step 4: Setup caches
    print("\n4️⃣ Setting up caches...")
    cache_start = time.time()
    
    with torch.device("mps"):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    
    after_cache = get_memory_info()
    cache_time = time.time() - cache_start
    
    print(f"   Cache setup time: {cache_time:.2f}s")
    print(f"   Memory after cache: {after_cache['rss_mb']:.1f}MB (+{after_cache['rss_mb'] - after_load['rss_mb']:.1f}MB)")
    print(f"   MPS memory: {after_cache['mps_allocated_mb']:.1f}MB")
    
    # Get tracemalloc statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n📊 Tracemalloc Results:")
    print(f"   Current traced: {current / 1024 / 1024:.1f}MB")
    print(f"   Peak traced: {peak / 1024 / 1024:.1f}MB")
    
    # Cleanup
    del model, decode_one_token
    torch.mps.empty_cache()
    gc.collect()
    
    final = get_memory_info()
    print(f"\n🧹 After cleanup: {final['rss_mb']:.1f}MB")
    
    return {
        'baseline': baseline['rss_mb'],
        'after_import': after_import['rss_mb'],
        'after_download': after_download['rss_mb'], 
        'after_load': after_load['rss_mb'],
        'after_cache': after_cache['rss_mb'],
        'final': final['rss_mb'],
        'peak_traced': peak / 1024 / 1024
    }

def find_memory_optimizations():
    """Find potential memory optimizations"""
    print("\n🛠️ Memory Optimization Analysis")
    print("=" * 60)
    
    optimizations = []
    
    # Test different precisions
    precisions = [
        ("bfloat16", torch.bfloat16),
        ("half", torch.half),
        ("float32", torch.float32)
    ]
    
    from huggingface_hub import snapshot_download
    from fish_speech.models.text2semantic.inference import init_model
    
    checkpoints_dir = snapshot_download(
        repo_id="fishaudio/fish-speech-1.5",
        repo_type="model"
    )
    
    for name, dtype in precisions:
        print(f"\n🔬 Testing {name} precision...")
        
        torch.mps.empty_cache()
        gc.collect()
        baseline = get_memory_info()
        
        try:
            model, decode_one_token = init_model(
                checkpoint_path=checkpoints_dir,
                device="mps",
                precision=dtype,
                compile=False
            )
            
            peak = get_memory_info()
            memory_usage = peak['rss_mb'] - baseline['rss_mb']
            mps_usage = peak['mps_allocated_mb']
            
            print(f"   RAM usage: {memory_usage:.1f}MB")
            print(f"   MPS usage: {mps_usage:.1f}MB")
            
            optimizations.append({
                'precision': name,
                'ram_mb': memory_usage,
                'mps_mb': mps_usage,
                'total_mb': memory_usage + mps_usage
            })
            
            del model, decode_one_token
            torch.mps.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Find best precision
    best = min(optimizations, key=lambda x: x['ram_mb'])
    print(f"\n🏆 Best precision for RAM: {best['precision']} ({best['ram_mb']:.1f}MB RAM)")
    
    return optimizations

def create_optimized_inference():
    """Create optimized inference function"""
    print("\n⚡ Creating Optimized Inference")
    print("=" * 60)
    
    optimization_code = '''
def optimized_synthesize_speech(text, output_path="output.wav", device="mps", checkpoints_dir=None):
    """Memory-optimized speech synthesis"""
    import torch
    import gc
    import sys
    from pathlib import Path
    
    # Force garbage collection
    gc.collect()
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    # Use half precision for better memory efficiency
    precision = torch.half if device == "mps" else torch.bfloat16
    
    # Import only when needed
    from fish_speech.models.text2semantic.inference import init_model, generate_long
    
    try:
        # Load model with optimized settings
        model, decode_one_token = init_model(
            checkpoint_path=checkpoints_dir,
            device=device,
            precision=precision,
            compile=False  # Disable compilation to save memory
        )
        
        # Use smaller cache sizes
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=min(2048, model.config.max_seq_len),  # Limit cache size
                dtype=precision,
            )
        
        # Generate with memory-efficient settings
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=1024,  # Limit token generation
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            compile=False,
            iterative_prompt=True,
            max_length=2048,  # Reduce max length
            chunk_length=100,  # Smaller chunks
        )
        
        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
        
        if codes:
            import numpy as np
            codes_array = torch.cat(codes, dim=1).cpu().numpy()
            return codes_array
        
    except Exception as e:
        print(f"❌ Optimized synthesis error: {e}")
        return None
    finally:
        # Clean up
        if 'model' in locals():
            del model, decode_one_token
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        gc.collect()
'''
    
    # Save optimized function
    with open("optimized_inference.py", "w") as f:
        f.write(optimization_code)
    
    print("✅ Created optimized_inference.py")
    return optimization_code

def main():
    print("🔬 Fish Speech Memory Profiler")
    print("=" * 70)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return
    
    # Test different text lengths
    results = {}
    for length in ["short", "medium", "long"]:
        try:
            result = profile_subprocess_memory(length)
            if result:
                results[length] = result
        except Exception as e:
            print(f"❌ Error profiling {length}: {e}")
    
    # Analyze memory components
    try:
        component_analysis = analyze_memory_components()
    except Exception as e:
        print(f"❌ Component analysis error: {e}")
        component_analysis = None
    
    # Find optimizations
    try:
        optimizations = find_memory_optimizations()
    except Exception as e:
        print(f"❌ Optimization analysis error: {e}")
        optimizations = None
    
    # Create optimized version
    create_optimized_inference()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 MEMORY ANALYSIS SUMMARY")
    print("=" * 70)
    
    if results:
        print("📊 Memory usage by text length:")
        for length, data in results.items():
            print(f"   {length:6}: {data['memory_delta']:6.1f}MB RAM, {data['char_count']:5} chars, {data['synthesis_time']:5.1f}s")
        
        # Calculate memory per character
        if len(results) > 1:
            long_result = results.get('long')
            short_result = results.get('short')
            if long_result and short_result:
                char_diff = long_result['char_count'] - short_result['char_count']
                mem_diff = long_result['memory_delta'] - short_result['memory_delta']
                if char_diff > 0:
                    print(f"   Memory per extra character: {mem_diff/char_diff*1000:.3f}MB/1000 chars")
    
    if component_analysis:
        print(f"\n🏗️ Memory breakdown:")
        print(f"   Import overhead: {component_analysis['after_import'] - component_analysis['baseline']:.1f}MB")
        print(f"   Model loading: {component_analysis['after_load'] - component_analysis['after_download']:.1f}MB") 
        print(f"   Cache setup: {component_analysis['after_cache'] - component_analysis['after_load']:.1f}MB")
    
    if optimizations:
        print(f"\n⚡ Optimization recommendations:")
        best_ram = min(optimizations, key=lambda x: x['ram_mb'])
        print(f"   Use {best_ram['precision']} precision: {best_ram['ram_mb']:.1f}MB RAM vs {max(optimizations, key=lambda x: x['ram_mb'])['ram_mb']:.1f}MB")
        print(f"   Expected savings: {max(optimizations, key=lambda x: x['ram_mb'])['ram_mb'] - best_ram['ram_mb']:.1f}MB")
    
    print(f"\n✅ Analysis complete! Check optimized_inference.py for memory-efficient implementation.")

if __name__ == "__main__":
    main() 