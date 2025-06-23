#!/usr/bin/env python3
"""Fish Speech CLI with Flash Attention optimizations"""

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
    """Optimize memory usage with enhanced cleanup"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_flash_attention_optimizations():
    """Apply Flash Attention optimizations before running TTS"""
    print("🔥 Applying Flash Attention optimizations...")
    
    try:
        # Import and apply enhancements
        from enhanced_flash_attention import apply_flash_attention_enhancements
        success = apply_flash_attention_enhancements()
        
        if success:
            print("✅ Flash Attention optimizations applied successfully!")
            return True
        else:
            print("⚠️  Flash Attention optimizations could not be applied, using standard version")
            return False
            
    except ImportError:
        print("⚠️  Enhanced Flash Attention module not found, using standard version")
        return False
    except Exception as e:
        print(f"⚠️  Error applying Flash Attention: {e}")
        return False


def run_flash_optimized_tts(args):
    """Run Fish Speech TTS with Flash Attention optimizations"""
    
    # Initial memory check
    initial_ram, initial_mps = monitor_memory()
    
    # Apply Flash Attention optimizations
    flash_applied = False
    if not args.no_flash:
        flash_applied = apply_flash_attention_optimizations()
    
    # Pre-optimize memory
    optimize_memory()
    
    start_time = time.time()
    
    try:
        # Build the command for the base CLI with all original arguments
        cmd = [sys.executable, "cli_tts.py"]
        
        # Positional argument
        if args.text:
            cmd.append(args.text)
        
        # Output and device
        cmd.extend(["-o", args.output])
        cmd.extend(["--device", args.device])
        
        # Play option
        if args.play:
            cmd.append("--play")
        
        # Model options
        if args.model_version != "1.5":  # Only add if not default
            cmd.extend(["--model-version", args.model_version])
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        
        # Voice options
        if args.prompt_tokens:
            cmd.extend(["--prompt-tokens", args.prompt_tokens])
        if args.prompt_text:
            cmd.extend(["--prompt-text", args.prompt_text])
        if args.prompt_tokens_file:
            cmd.extend(["--prompt-tokens-file", args.prompt_tokens_file])
        if args.voice:
            cmd.extend(["--voice", args.voice])
        
        # Prosody parameters
        if args.speed != 1.0:
            cmd.extend(["--speed", str(args.speed)])
        if args.volume != 0:
            cmd.extend(["--volume", str(args.volume)])
        if args.pitch != 1.0:
            cmd.extend(["--pitch", str(args.pitch)])
        
        # Emotion and style parameters
        if args.emotion:
            cmd.extend(["--emotion", args.emotion])
        if args.intensity != 0.5:
            cmd.extend(["--intensity", str(args.intensity)])
        if args.style:
            cmd.extend(["--style", args.style])
        
        # Cache options
        if args.no_cache:
            cmd.append("--no-cache")
        if args.clear_cache:
            cmd.append("--clear-cache")
        if args.cache_info:
            cmd.append("--cache-info")
        
        # Create reference option
        if args.create_reference:
            cmd.extend(["--create-reference"] + list(args.create_reference))
        
        # List voices option
        if args.list_voices:
            cmd.append("--list-voices")
        
        # Run the base CLI with enhanced memory monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
            env=dict(os.environ, **{
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # More aggressive MPS memory management
                'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection'
            })
        )
        
        # Enhanced memory monitoring during execution
        peak_ram = initial_ram
        peak_mps = initial_mps
        memory_samples = []
        
        while process.poll() is None:
            time.sleep(0.3)  # More frequent monitoring
            current_ram, current_mps = monitor_memory()
            peak_ram = max(peak_ram, current_ram)
            peak_mps = max(peak_mps, current_mps)
            memory_samples.append((current_ram, current_mps))
            
            # Proactive memory cleanup during long generations
            if len(memory_samples) > 10 and len(memory_samples) % 5 == 0:
                optimize_memory()
        
        stdout, stderr = process.communicate()
        
        # Final memory check and cleanup
        optimize_memory()
        final_ram, final_mps = monitor_memory()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Pass through original output
        if stdout:
            print(stdout, end='')
        if stderr:
            print(stderr, end='', file=sys.stderr)
        
        if process.returncode == 0:
            # Check if output file exists for metrics
            if Path(args.output).exists():
                file_size = Path(args.output).stat().st_size / 1024
                avg_ram = sum(sample[0] for sample in memory_samples) / len(memory_samples) if memory_samples else 0
                avg_mps = sum(sample[1] for sample in memory_samples) / len(memory_samples) if memory_samples else 0
                
                return True, {
                    'time': generation_time,
                    'peak_ram': peak_ram - initial_ram,
                    'peak_mps': peak_mps - initial_mps,
                    'avg_ram': avg_ram - initial_ram,
                    'avg_mps': avg_mps - initial_mps,
                    'output_file': args.output,
                    'file_size_kb': file_size,
                    'flash_attention_applied': flash_applied,
                    'memory_samples': len(memory_samples)
                }
            else:
                return True, None  # Success but no output file (e.g., cache operations)
        else:
            return False, None
            
    except Exception as e:
        return False, None
    
    finally:
        # Final aggressive memory optimization
        optimize_memory()


def compare_attention_backends():
    """Compare different attention backend performance"""
    print("🔥 Comparing Flash Attention Backends...")
    
    try:
        from enhanced_flash_attention import test_flash_attention_performance
        return test_flash_attention_performance()
    except ImportError:
        print("⚠️  Enhanced Flash Attention not available for comparison")
        return False


def main():
    """Main function with enhanced CLI argument parsing matching cli_tts.py"""
    parser = argparse.ArgumentParser(description="Fish Speech TTS with Flash Attention optimizations")
    
    # Main arguments (matching cli_tts.py exactly)
    parser.add_argument("text", nargs='?', help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output/output.wav", help="Output file (default: output/output.wav)")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="Device for calculations")
    parser.add_argument("--play", action="store_true", help="Play immediately after synthesis")
    
    # Model options
    parser.add_argument("--model-version", default="1.5", choices=["1.4", "1.5", "1.6"], 
                       help="Fish Speech model version (default: 1.5)")
    parser.add_argument("--model-path", type=str, help="Path to custom fine-tuned model (overrides --model-version)")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.ckpt file or directory with model files)")
    
    # Voice options
    parser.add_argument("--prompt-tokens", type=str, help="Path to .npy file with voice reference")
    parser.add_argument("--prompt-text", type=str, help="Text corresponding to voice reference (required with --prompt-tokens)")
    parser.add_argument("--prompt-tokens-file", type=str, help="Path to .txt file with text for voice reference (alternative to --prompt-text)")
    parser.add_argument("--voice", type=str, help="Voice name from voices folder (automatically finds .npy and .txt files)")
    
    # Prosody parameters
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0, default: 1.0)")
    parser.add_argument("--volume", type=int, default=0, help="Volume in dB (-20 to +20, default: 0)")
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch (0.5-2.0, default: 1.0)")
    
    # Emotion and style parameters
    parser.add_argument("--emotion", choices=["happy", "sad", "angry", "neutral", "excited"], 
                       help="Speech emotion")
    parser.add_argument("--intensity", type=float, default=0.5, 
                       help="Emotion intensity (0.0-1.0, default: 0.5)")
    parser.add_argument("--style", choices=["formal", "casual", "dramatic"], 
                       help="Speech style")
    
    # Cache options
    parser.add_argument("--no-cache", action="store_true", help="Disable semantic tokens cache")
    parser.add_argument("--clear-cache", action="store_true", help="Clear semantic tokens cache")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information")
    
    # Option to create reference from audio
    parser.add_argument("--create-reference", nargs=2, metavar=("audio_file", "output.npy"),
                       help="Create voice reference from audio file: --create-reference input.wav reference.npy")
    
    # Voice information
    parser.add_argument("--list-voices", action="store_true", help="Show list of available voices")
    
    # Flash Attention specific options
    parser.add_argument("--monitor", action="store_true", help="Show detailed memory monitoring")
    parser.add_argument("--no-flash", action="store_true", help="Disable Flash Attention optimizations")
    parser.add_argument("--benchmark", action="store_true", help="Run Flash Attention benchmark")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        args.device = device
    
    # Run benchmark if requested
    if args.benchmark:
        compare_attention_backends()
        return 0
    
    # For operations that don't require text, pass through to original CLI
    if args.clear_cache or args.cache_info or args.list_voices or args.create_reference:
        success, result = run_flash_optimized_tts(args)
        return 0 if success else 1
    
    # Check if text is provided for TTS
    if not args.text:
        parser.error("Text is required for TTS generation")
    
    # Validate prosody parameters (matching cli_tts.py validation)
    if not (0.5 <= args.speed <= 2.0):
        parser.error("Speech speed must be between 0.5 and 2.0")
    
    if not (-20 <= args.volume <= 20):
        parser.error("Volume must be between -20 and +20 dB")
    
    if not (0.5 <= args.pitch <= 2.0):
        parser.error("Pitch must be between 0.5 and 2.0")
    
    if args.intensity is not None and not (0.0 <= args.intensity <= 1.0):
        parser.error("Emotion intensity must be between 0.0 and 1.0")
    
    # Check validity of options for voice references
    if args.prompt_tokens and not args.prompt_text and not args.prompt_tokens_file:
        parser.error("When using --prompt-tokens, --prompt-text or --prompt-tokens-file must be specified")
    
    # Check conflict between --voice and other voice options
    if args.voice and (args.prompt_tokens or args.prompt_text or args.prompt_tokens_file):
        parser.error("Parameter --voice cannot be used with --prompt-tokens, --prompt-text or --prompt-tokens-file")
    
    # Validate checkpoint path if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            parser.error(f"Checkpoint file not found: {args.checkpoint}")
        
        # Support both .ckpt files and directories with model files
        if checkpoint_path.is_file():
            if not args.checkpoint.endswith('.ckpt'):
                parser.error("Checkpoint file must have .ckpt extension")
        elif checkpoint_path.is_dir():
            # Check if directory contains model files
            required_files = ["config.json"]
            missing_files = [f for f in required_files if not (checkpoint_path / f).exists()]
            if missing_files:
                parser.error(f"Checkpoint directory missing required files: {missing_files}")
    
    # Run Flash-Optimized TTS
    success, result = run_flash_optimized_tts(args)
    
    if success and result and args.monitor:
        print(f"\n📊 Detailed Performance Metrics:")
        print(f"   Generation time: {result['time']:.2f}s")
        print(f"   Peak RAM usage: {result['peak_ram']:.1f}MB")
        print(f"   Peak MPS usage: {result['peak_mps']:.1f}MB")
        print(f"   Average RAM usage: {result['avg_ram']:.1f}MB")
        print(f"   Average MPS usage: {result['avg_mps']:.1f}MB")
        print(f"   Output file size: {result['file_size_kb']:.1f}KB")
        print(f"   Generation speed: {result['file_size_kb'] / result['time']:.1f} KB/s")
        print(f"   Flash Attention: {'✅ Applied' if result['flash_attention_applied'] else '❌ Not applied'}")
        print(f"   Memory samples: {result['memory_samples']}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 