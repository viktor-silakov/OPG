#!/usr/bin/env python3
"""Memory-optimized Fish Speech TTS"""

import torch
import gc
import argparse
import sys
from pathlib import Path
from cli_tts import setup_fish_speech, synthesize_speech_cli

def optimized_synthesize(text, output_path="output_optimized.wav", device="mps", **kwargs):
    """Memory-optimized synthesis with aggressive cleanup"""
    
    print(f"🚀 Optimized TTS: {text[:50]}...")
    
    # Force garbage collection before starting
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    try:
        # Setup with minimal memory footprint
        model_version = kwargs.get('model_version', '1.5')
        custom_model_path = kwargs.get('model_path')
        fish_speech_dir, checkpoints_dir = setup_fish_speech(model_version, custom_model_path)
        
        # Use existing optimized function but force no caching unless explicitly requested
        # Remove use_cache from kwargs to avoid conflict
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'use_cache'}
        
        success = synthesize_speech_cli(
            text=text,
            output_path=output_path,
            device=device,
            checkpoints_dir=checkpoints_dir,
            use_cache=kwargs.get('use_cache', False),  # Disable caching by default to save memory
            **filtered_kwargs
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Optimized synthesis error: {e}")
        return False
        
    finally:
        # Aggressive cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def create_memory_optimized_inference():
    """Create memory-optimized inference version by modifying Fish Speech params"""
    print("🔧 Creating memory optimizations...")
    
    # Create a modified inference.py with optimized parameters
    fish_speech_dir = Path("fish-speech")
    inference_file = fish_speech_dir / "fish_speech/models/text2semantic/inference.py"
    
    if inference_file.exists():
        # Read original file
        with open(inference_file, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_file = inference_file.with_suffix('.py.backup')
        if not backup_file.exists():
            with open(backup_file, 'w') as f:
                f.write(content)
        
        # Apply memory optimizations
        optimizations = [
            # Force half precision for MPS
            ('precision = torch.half if half else torch.bfloat16', 
             'precision = torch.half if device == "mps" else (torch.half if half else torch.bfloat16)'),
            
            # Reduce default max_seq_len
            ('max_seq_len=model.config.max_seq_len,', 
             'max_seq_len=min(2048, model.config.max_seq_len),'),
            
            # Smaller chunk length by default
            ('chunk_length: int = 150,', 
             'chunk_length: int = 100,'),
            
            # Reduce max_new_tokens default
            ('max_new_tokens: int = 0,', 
             'max_new_tokens: int = 1024,')
        ]
        
        modified_content = content
        for old, new in optimizations:
            if old in modified_content:
                modified_content = modified_content.replace(old, new)
                print(f"   ✅ Applied: {old[:30]}...")
        
        # Write optimized version
        optimized_file = inference_file.with_suffix('.py.optimized')
        with open(optimized_file, 'w') as f:
            f.write(modified_content)
        
        print(f"✅ Created optimized inference: {optimized_file}")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Fish Speech TTS")
    parser.add_argument("text", nargs='?', help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output/optimized.wav", help="Output file (default: output/optimized.wav)")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="Device for calculations")
    parser.add_argument("--play", action="store_true", help="Play immediately after synthesis")
    
    # Model options
    parser.add_argument("--model-version", default="1.5", choices=["1.4", "1.5", "1.6"], 
                       help="Fish Speech model version (default: 1.5)")
    parser.add_argument("--model-path", type=str, help="Path to custom fine-tuned model (overrides --model-version)")
    
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
    
    # Memory optimization option
    parser.add_argument("--optimize", action="store_true", help="Create optimized inference version")
    
    args = parser.parse_args()
    
    # Import additional dependencies for full functionality
    import shutil
    import subprocess
    from cli_tts import clear_semantic_cache, get_directory_size, create_voice_reference
    
    if args.optimize:
        create_memory_optimized_inference()
        return
    
    # Clear cache if requested
    if args.clear_cache:
        clear_semantic_cache()
        return
    
    # Show cache information if requested
    if args.cache_info:
        cache_dir = Path("cache/semantic_tokens")
        _, checkpoints_dir = setup_fish_speech(args.model_version, args.model_path)
        model_cache_size = get_directory_size(checkpoints_dir)
        
        print(f"📁 Model cache: {checkpoints_dir}")
        print(f"💾 Model size: {model_cache_size:.1f} MB")
        
        if cache_dir.exists():
            semantic_cache_size = get_directory_size(cache_dir)
            files_count = len(list(cache_dir.glob("*.npy")))
            print(f"📁 Semantic tokens cache: {cache_dir}")
            print(f"💾 Cache size: {semantic_cache_size:.1f} MB")
            print(f"📄 Files in cache: {files_count}")
        else:
            print("📭 Semantic tokens cache is empty")
        
        print(f"🗂️ Model files:")
        for file in checkpoints_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {file.name}: {size_mb:.1f} MB")
        return
    
    # Show voice list if requested
    if args.list_voices:
        voices_dir = Path("voices")
        if not voices_dir.exists():
            print("❌ Voices folder not found")
            return
        
        npy_files = list(voices_dir.glob("*.npy"))
        if not npy_files:
            print("❌ Voice reference files not found in voices folder")
            return
        
        print(f"🎭 Available voices ({len(npy_files)}):")
        for npy_file in sorted(npy_files):
            voice_name = npy_file.stem
            txt_file = voices_dir / f"{voice_name}.txt"
            
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        preview = first_line[:50] + "..." if len(first_line) > 50 else first_line
                    print(f"   ✅ {voice_name} - {preview}")
                except:
                    print(f"   ✅ {voice_name} - (has txt file)")
            else:
                print(f"   ⚠️  {voice_name} - (no txt file)")
        
        print(f"\n💡 Usage: --voice VOICE_NAME")
        print(f"   Example: --voice RU_Male_Deadpool")
        return
    
    # If reference creation is needed
    if args.create_reference:
        audio_file, reference_file = args.create_reference
        _, checkpoints_dir = setup_fish_speech(args.model_version, args.model_path)
        success = create_voice_reference(audio_file, reference_file, checkpoints_dir, args.device)
        if success:
            print(f"🎭 Referencer created! Now use: --prompt-tokens {reference_file}")
        return
    
    # Check that text is provided for synthesis
    if not args.text:
        parser.error("Text is required for synthesis or --create-reference option")
    
    print("🧠 Memory-Optimized Fish Speech TTS")
    print("=" * 50)
    print(f"💻 Device: {args.device}")
    if args.model_path:
        print(f"🎤 Custom model: {args.model_path}")
    else:
        print(f"🤖 Model: Fish Speech {args.model_version}")
    
    # Validate prosody parameters
    if not (0.5 <= args.speed <= 2.0):
        parser.error("Speech speed must be between 0.5 and 2.0")
    
    if not (-20 <= args.volume <= 20):
        parser.error("Volume must be between -20 and +20 dB")
    
    if not (0.5 <= args.pitch <= 2.0):
        parser.error("Pitch must be between 0.5 and 2.0")
    
    if args.intensity is not None and not (0.0 <= args.intensity <= 1.0):
        parser.error("Emotion intensity must be between 0.0 and 1.0")
    
    # Check MPS availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS not available, switching to CPU")
        args.device = "cpu"
    
    # Check validity of options for voice references
    if args.prompt_tokens and not args.prompt_text and not args.prompt_tokens_file:
        parser.error("When using --prompt-tokens, --prompt-text or --prompt-tokens-file must be specified")
    
    # Check conflict between --voice and other voice options
    if args.voice and (args.prompt_tokens or args.prompt_text or args.prompt_tokens_file):
        parser.error("Parameter --voice cannot be used with --prompt-tokens, --prompt-text or --prompt-tokens-file")
    
    # Handle --voice parameter
    if args.voice:
        voices_dir = Path("voices")
        voice_npy = voices_dir / f"{args.voice}.npy"
        voice_txt = voices_dir / f"{args.voice}.txt"
        
        if not voice_npy.exists():
            parser.error(f"Voice file not found: {voice_npy}")
        
        args.prompt_tokens = str(voice_npy)
        if voice_txt.exists():
            args.prompt_tokens_file = str(voice_txt)
            print(f"🎭 Using voice: {args.voice}")
            print(f"📄 Reference: {voice_npy}")
            print(f"📝 Text: {voice_txt}")
        else:
            print(f"⚠️ Text file not found: {voice_txt}")
            print(f"🎭 Using only reference: {voice_npy}")
    
    # Read prompt text from file if specified
    prompt_text = args.prompt_text
    if args.prompt_tokens_file and not args.prompt_text:
        try:
            with open(args.prompt_tokens_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            print(f"📄 Loaded text from file: {args.prompt_tokens_file} ({len(prompt_text)} characters)")
        except FileNotFoundError:
            parser.error(f"Text file not found: {args.prompt_tokens_file}")
        except Exception as e:
            parser.error(f"Text file reading error: {e}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show memory before
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"📊 Initial RAM: {initial_memory:.1f}MB")
    
    if torch.backends.mps.is_available():
        initial_mps = torch.mps.current_allocated_memory() / 1024 / 1024
        print(f"📊 Initial MPS: {initial_mps:.1f}MB")
    
    # Run optimized synthesis with all parameters
    success = optimized_synthesize(
        args.text,
        args.output,
        args.device,
        model_version=args.model_version,
        model_path=args.model_path,
        prompt_tokens=args.prompt_tokens,
        prompt_text=prompt_text,
        use_cache=not args.no_cache,
        speed=args.speed,
        volume=args.volume,
        pitch=args.pitch,
        emotion=args.emotion,
        intensity=args.intensity,
        style=args.style
    )
    
    # Show memory after
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"📊 Final RAM: {final_memory:.1f}MB (Δ{final_memory - initial_memory:+.1f}MB)")
    
    if torch.backends.mps.is_available():
        final_mps = torch.mps.current_allocated_memory() / 1024 / 1024
        print(f"📊 Final MPS: {final_mps:.1f}MB (Δ{final_mps - initial_mps:+.1f}MB)")
    
    if success and Path(args.output).exists():
        file_size = Path(args.output).stat().st_size
        print(f"✅ Generated: {args.output} ({file_size} bytes)")
        
        if args.play and shutil.which("afplay"):
            print("🔊 Playing...")
            subprocess.run(["afplay", args.output])
    else:
        print("❌ Synthesis failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 