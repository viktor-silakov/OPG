#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import torch
import soundfile as sf
import subprocess
import shutil
import tempfile
import hashlib
import json
from huggingface_hub import snapshot_download

# Global cache for VQGAN model path to avoid repeated downloads
_vqgan_model_cache = {}

def get_directory_size(directory):
    """Returns the size of the directory in MB"""
    total_size = 0
    for file in Path(directory).rglob('*'):
        if file.is_file():
            try:
                total_size += file.stat().st_size
            except (OSError, FileNotFoundError):
                pass
    return total_size / (1024 * 1024)

def get_cache_key(text, prompt_tokens=None, prompt_text=None, **kwargs):
    """Generates a cache key based on input parameters"""
    # Create a dictionary with parameters for hashing
    cache_params = {
        'text': text,
        'prompt_text': prompt_text or '',
        'chunk_length': kwargs.get('chunk_length', '200'),
        'max_new_tokens': kwargs.get('max_new_tokens', '2048'),
        'top_p': kwargs.get('top_p', '0.7'),
        'temperature': kwargs.get('temperature', '0.7'),
        'repetition_penalty': kwargs.get('repetition_penalty', '1.2'),
        # New parameters for prosody and emotions
        'speed': kwargs.get('speed', 1.0),
        'volume': kwargs.get('volume', 0),
        'pitch': kwargs.get('pitch', 1.0),
        'emotion': kwargs.get('emotion', ''),
        'intensity': kwargs.get('intensity', 0.5),
        'style': kwargs.get('style', ''),
        # Checkpoint parameter
        'checkpoint': kwargs.get('checkpoint', '')
    }
    
    # Add the hash of the prompt_tokens file if specified
    if prompt_tokens and Path(prompt_tokens).exists():
        with open(prompt_tokens, 'rb') as f:
            prompt_hash = hashlib.md5(f.read()).hexdigest()[:8]
        cache_params['prompt_tokens_hash'] = prompt_hash
    
    # Create a JSON string and hash it
    cache_string = json.dumps(cache_params, sort_keys=True, ensure_ascii=False)
    cache_hash = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
    
    return cache_hash

def setup_cache_dir():
    """Creates and returns the cache directory"""
    cache_dir = Path("cache/semantic_tokens")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cached_tokens(cache_key, cache_dir):
    """Returns the path to cached tokens if they exist"""
    cache_file = cache_dir / f"{cache_key}.npy"
    if cache_file.exists():
        return cache_file
    return None

def save_tokens_to_cache(tokens_file, cache_key, cache_dir):
    """Saves tokens to cache"""
    cache_file = cache_dir / f"{cache_key}.npy"
    try:
        shutil.copy2(tokens_file, cache_file)
        return cache_file
    except Exception as e:
        print(f"⚠️ Failed to save to cache: {e}")
        return None

def clear_semantic_cache():
    """Clear semantic token cache"""
    cache_dir = setup_cache_dir()
    if cache_dir and cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"🗑️ Semantic cache cleared: {cache_dir}")
            return True
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    return True

def get_vqgan_model_path(checkpoints_dir):
    """Get VQGAN model path with caching to avoid repeated downloads"""
    global _vqgan_model_cache
    
    checkpoints_dir = Path(checkpoints_dir)
    cache_key = str(checkpoints_dir.resolve())
    
    # Check if already cached
    if cache_key in _vqgan_model_cache:
        return _vqgan_model_cache[cache_key]
    
    # Check if VQGAN exists in custom model
    vqgan_file = checkpoints_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    if vqgan_file.exists():
        _vqgan_model_cache[cache_key] = checkpoints_dir
        return checkpoints_dir
    
    # Need to get VQGAN from base model - do this only once
    print("🔄 Custom model: getting VQGAN from base model (one-time setup)...")
    try:
        from huggingface_hub import snapshot_download
        base_model_path = snapshot_download(
            repo_id="fishaudio/fish-speech-1.5",
            repo_type="model"
        )
        vqgan_model_path = Path(base_model_path)
        print(f"✅ VQGAN model cached: {vqgan_model_path}")
        
        # Cache the result
        _vqgan_model_cache[cache_key] = vqgan_model_path
        return vqgan_model_path
        
    except Exception as e:
        print(f"❌ Base model retrieval error for VQGAN: {e}")
        return None

def clear_vqgan_cache():
    """Clear VQGAN model path cache"""
    global _vqgan_model_cache
    _vqgan_model_cache.clear()
    print("🗑️ VQGAN model cache cleared")

def setup_fish_speech(model_version="1.5", custom_model_path=None):
    """Checks and sets up Fish Speech"""
    fish_speech_dir = Path("fish-speech")
    if not fish_speech_dir.exists():
        print(f"❌ Fish Speech not found in {fish_speech_dir}")
        print("📝 Fish Speech is already included in this project.")
        print("🔧 Make sure it's installed: pip install -e ./fish-speech")
        return None
    
    # If a custom model path is specified
    if custom_model_path:
        custom_path = Path(custom_model_path)
        if not custom_path.exists():
            print(f"❌ Custom model not found: {custom_path}")
            sys.exit(1)
            
        # Check for the presence of necessary files
        required_files = ["model.pth", "config.json"]
        missing_files = [f for f in required_files if not (custom_path / f).exists()]
        
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            sys.exit(1)
            
        print(f"🎤 Using custom model: {custom_path}")
        return fish_speech_dir, custom_path
    
    # Determine repo_id based on model version
    version_mapping = {
        "1.4": "fishaudio/fish-speech-1.4",
        "1.5": "fishaudio/fish-speech-1.5", 
        "1.6": "fishaudio/fish-speech-1.6"
    }
    
    repo_id = version_mapping.get(model_version, "fishaudio/fish-speech-1.5")
    print(f"🐟 Using Fish Speech model {model_version}")
    
    # Use huggingface cache instead of local folder
    print("📥 Checking Fish Speech model in cache...")
    try:
        # snapshot_download automatically uses ~/.cache/huggingface/hub/
        checkpoints_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Model found in cache: {checkpoints_dir}")
        
    except Exception as e:
        print(f"❌ Model retrieval error: {e}")
        sys.exit(1)
    
    return fish_speech_dir, Path(checkpoints_dir)

def create_voice_reference(audio_path, output_npy_path, checkpoints_dir, device="mps"):
    """Creates a reference npy file from audio"""
    print(f"🎙️  Creating voice reference from {audio_path}")
    
    fish_speech_dir = Path("fish-speech")
    
    abs_audio_path = Path(audio_path).resolve()
    abs_output_path = Path(output_npy_path).resolve()
    
    if not abs_audio_path.exists():
        print(f"❌ Audio file not found: {abs_audio_path}")
        return False
    
    try:
        vqgan_cmd = [
            sys.executable, "-m", "fish_speech.models.vqgan.inference",
            "--input-path", str(abs_audio_path),
            "--output-path", str(abs_output_path.with_suffix('.wav')),  # Temporary wav
            "--config-name", "firefly_gan_vq",
            "--checkpoint-path", str(checkpoints_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            "--device", device
        ]
        
        result = subprocess.run(
            vqgan_cmd,
            capture_output=True,
            text=True,
            cwd=fish_speech_dir
        )
        
        if result.returncode != 0:
            print(f"❌ Reference creation error: {result.stderr}")
            return False
        
        # VQGAN creates npy file automatically
        npy_file = abs_output_path.with_suffix('.npy')
        if npy_file.exists():
            # Move to the correct location if needed
            if npy_file != abs_output_path:
                shutil.move(str(npy_file), str(abs_output_path))
            print(f"✅ Voice reference created: {abs_output_path}")
            return True
        else:
            print(f"❌ Reference creation failed: {npy_file}")
            return False
            
    except Exception as e:
        print(f"❌ Reference creation error: {e}")
        return False

def apply_prosody_effects(audio_path, speed=1.0, volume_db=0, pitch=1.0):
    """Applies prosody effects to an audio file"""
    try:
        import numpy as np
        from scipy.signal import resample
        
        # Read audio
        audio_data, sample_rate = sf.read(audio_path)
        
        # Speed change through resampling
        if speed != 1.0:
            # Resampling for speed change
            new_length = int(len(audio_data) / speed)
            audio_data = resample(audio_data, new_length)
        
        # Volume change (dB to linear coefficient)
        if volume_db != 0:
            # Convert dB to linear multiplier: dB = 20 * log10(amplitude)
            volume_factor = 10 ** (volume_db / 20.0)
            audio_data = audio_data * volume_factor
            
            # Prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
        
        # Pitch change (pitch shifting)
        if pitch != 1.0:
            # Simple pitch shifting through changing sample rate during playback
            # For higher quality pitch shifting, more complex algorithms are needed
            new_sample_rate = int(sample_rate * pitch)
            # Write with changed sample rate, but read as usual
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, new_sample_rate)
                # Read back with original sample rate
                audio_data, _ = sf.read(temp_file.name)
                Path(temp_file.name).unlink()  # Delete temporary file
        
        # Save processed audio
        sf.write(audio_path, audio_data, sample_rate)
        return True
        
    except ImportError as e:
        print(f"⚠️ Prosody requires scipy: pip install scipy")
        return False
    except Exception as e:
        print(f"⚠️ Prosody processing error: {e}")
        return False

def synthesize_speech_cli(text, output_path="output.wav", device="mps", prompt_tokens=None, prompt_text=None, checkpoints_dir=None, use_cache=True, **kwargs):
    """Synthesizes speech using Fish Speech CLI commands"""
    print(f"🎯 Synthesizing: {text}")
    
    if prompt_tokens:
        print(f"🎭 Using voice reference: {prompt_tokens}")
    
    # Extract additional parameters
    speed = kwargs.get('speed', 1.0)
    volume = kwargs.get('volume', 0)
    pitch = kwargs.get('pitch', 1.0)
    emotion = kwargs.get('emotion')
    intensity = kwargs.get('intensity', 0.5)
    style = kwargs.get('style')
    checkpoint = kwargs.get('checkpoint')
    
    # Show active prosody parameters
    prosody_params = []
    if speed != 1.0:
        prosody_params.append(f"speed: {speed}x")
    if volume != 0:
        prosody_params.append(f"volume: {volume:+d}dB")
    if pitch != 1.0:
        prosody_params.append(f"pitch: {pitch}x")
    if emotion:
        prosody_params.append(f"emotion: {emotion}")
        if intensity != 0.5:
            prosody_params.append(f"intensity: {intensity}")
    if style:
        prosody_params.append(f"style: {style}")
    
    if prosody_params:
        print(f"🎵 Prosody parameters: {', '.join(prosody_params)}")
    
    fish_speech_dir = Path("fish-speech")
    
    # Define paths for semantic and VQGAN models
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        
        if checkpoint_path.is_file() and checkpoint_path.suffix == '.ckpt':
            # This is a Lightning checkpoint (likely LoRA fine-tuned)
            print(f"📦 Detected Lightning checkpoint: {checkpoint_path}")
            
            # Check for hydra config to determine base model
            hydra_config = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
            if hydra_config.exists():
                print(f"📋 Found training config: {hydra_config}")
                
                # Try to convert LoRA checkpoint to inference format
                converted_model_dir = convert_lora_checkpoint(checkpoint_path, hydra_config)
                
                if converted_model_dir and converted_model_dir.exists():
                    # Use converted model
                    semantic_model_path = converted_model_dir
                    checkpoint_file = None
                    print(f"🎯 Using converted LoRA model: {semantic_model_path}")
                else:
                    # Fallback to base model
                    try:
                        import yaml
                        with open(hydra_config, 'r') as f:
                            config = yaml.safe_load(f)
                        
                        base_model_path = config.get('pretrained_ckpt_path')
                        if base_model_path:
                            semantic_model_path = Path(base_model_path)
                            print(f"⚠️  Conversion failed, using base model: {semantic_model_path}")
                        else:
                            semantic_model_path = checkpoints_dir
                            print(f"⚠️  Could not find base model, using default")
                        checkpoint_file = None
                    except Exception as e:
                        print(f"⚠️  Error parsing config: {e}, using default model")
                        semantic_model_path = checkpoints_dir
                        checkpoint_file = None
            else:
                print(f"⚠️  No hydra config found, treating as regular checkpoint")
                semantic_model_path = checkpoints_dir
                checkpoint_file = checkpoint_path
                
        elif checkpoint_path.is_dir():
            # Regular checkpoint directory
            semantic_model_path = checkpoint_path
            checkpoint_file = None
            print(f"📁 Using checkpoint directory: {semantic_model_path}")
        else:
            print(f"❌ Unsupported checkpoint format: {checkpoint_path}")
            semantic_model_path = checkpoints_dir
            checkpoint_file = None
    else:
        semantic_model_path = checkpoints_dir
        checkpoint_file = None
    
    # Get VQGAN model path with caching (avoids repeated downloads)
    vqgan_model_path = get_vqgan_model_path(checkpoints_dir)
    if vqgan_model_path is None:
        print("❌ Failed to obtain VQGAN model")
        return False
    
    # Cache setup
    cache_dir = setup_cache_dir() if use_cache else None
    cache_key = None
    cached_tokens = None
    
    if use_cache:
        # Include prosody parameters and checkpoint in cache key
        cache_key = get_cache_key(text, prompt_tokens, prompt_text, 
                                 speed=speed, volume=volume, pitch=pitch,
                                 emotion=emotion, intensity=intensity, style=style,
                                 checkpoint=checkpoint)
        cached_tokens = get_cached_tokens(cache_key, cache_dir)
        
        if cached_tokens:
            print(f"💾 Found cached tokens: {cache_key[:8]}...")
    
    try:
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            codes_file = None
            
            if cached_tokens:
                # Use cached tokens
                codes_file = cached_tokens
                print("⚡ Using cached semantic tokens")
            else:
                # Generate new tokens
                print("📊 Generating semantic tokens...")
                
                semantic_cmd = [
                    sys.executable, "-m", "fish_speech.models.text2semantic.inference",
                    "--text", text,
                    "--checkpoint-path", str(semantic_model_path.resolve()),
                    "--num-samples", "1",
                    "--device", device,
                    "--output-dir", str(temp_path),
                    "--iterative-prompt",
                    "--chunk-length", "200",
                    "--max-new-tokens", "2048",
                    "--top-p", "0.7",
                    "--temperature", "0.7",
                    "--repetition-penalty", "1.2"
                ]
                
                # Note about checkpoint loading
                if checkpoint_file:
                    print(f"🔧 Note: Using Lightning checkpoint directly (not converted)")
                    print(f"💡 For better performance, consider converting to inference format")
                
                # Add emotion and style parameters if supported
                # Note: emotion parameters are not supported in the base version of Fish Speech
                # if emotion:
                #     semantic_cmd.extend(["--emotion", emotion])
                #     if intensity != 0.5:
                #         semantic_cmd.extend(["--emotion-intensity", str(intensity)])
                # 
                # if style:
                #     semantic_cmd.extend(["--style", style])
                
                # Add prompt tokens if specified
                if prompt_tokens:
                    abs_prompt_path = Path(prompt_tokens).resolve()
                    if abs_prompt_path.exists():
                        semantic_cmd.extend(["--prompt-tokens", str(abs_prompt_path)])
                        if prompt_text:
                            semantic_cmd.extend(["--prompt-text", prompt_text])
                    else:
                        print(f"⚠️ Reference file not found: {abs_prompt_path}, continuing without it")
                
                result = subprocess.run(
                    semantic_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=fish_speech_dir
                )
                
                if result.returncode != 0:
                    print(f"❌ Token generation error: {result.stderr}")
                    print(f"stdout: {result.stdout}")
                    return False
                
                print("✅ Semantic tokens generated")
                
                # Find the generated tokens file
                codes_files = list(temp_path.glob("codes_*.npy"))
                if not codes_files:
                    print("❌ Tokens file not found")
                    return False
                
                codes_file = codes_files[0]
                print(f"📄 Found tokens file: {codes_file}")
                
                # Save to cache if enabled
                if use_cache and cache_key:
                    cached_file = save_tokens_to_cache(codes_file, cache_key, cache_dir)
                    if cached_file:
                        print(f"💾 Tokens saved to cache: {cache_key[:8]}...")
            
            # Step 2: Generate audio from tokens
            print("🎵 Generating audio from tokens...")
            
            # Create full output file path relative to fish-speech
            abs_output_path = (Path.cwd() / output_path).resolve()
            
            # Use absolute path for tokens input file
            abs_codes_path = Path(codes_file).resolve()
            
            vqgan_cmd = [
                sys.executable, "-m", "fish_speech.models.vqgan.inference",
                "--input-path", str(abs_codes_path),
                "--output-path", str(abs_output_path),
                "--config-name", "firefly_gan_vq",
                "--checkpoint-path", str(vqgan_model_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
                "--device", device
            ]
            
            # Note: VQGAN does not support prosody parameters directly
            # They will be applied after generation through audio post-processing
            
            result = subprocess.run(
                vqgan_cmd,
                capture_output=True,
                text=True,
                cwd=fish_speech_dir
            )
            
            if result.returncode != 0:
                print(f"❌ Audio generation error: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return False
            
            print("✅ Audio generated")
            
            # Apply post-processing for prosody parameters
            if speed != 1.0 or volume != 0 or pitch != 1.0:
                print("🎛️ Applying prosody parameters...")
                success = apply_prosody_effects(abs_output_path, speed, volume, pitch)
                if not success:
                    print("⚠️ Prosody application error, leaving original audio")
            
            # Check result
            if abs_output_path.exists():
                file_size = abs_output_path.stat().st_size
                print(f"✅ TTS completed! File: {output_path} ({file_size} bytes)")
                print(f"🔊 For playback: afplay {output_path}")
                return True
            else:
                print("❌ Output file not created")
                print(f"Checked path: {abs_output_path}")
                return False
                
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_lora_checkpoint(checkpoint_path, hydra_config_path, output_dir=None):
    """Convert Lightning LoRA checkpoint to inference format"""
    checkpoint_path = Path(checkpoint_path)
    hydra_config_path = Path(hydra_config_path)
    
    if output_dir is None:
        # Create unique directory for each checkpoint
        checkpoint_name = checkpoint_path.stem  # e.g., "step_000002100"
        output_dir = checkpoint_path.parent / f"converted_inference_{checkpoint_name}"
    else:
        output_dir = Path(output_dir)
    
    # Check if already converted
    if (output_dir / "model.pth").exists() and (output_dir / "config.json").exists():
        print(f"✅ Converted model already exists: {output_dir}")
        return output_dir
    
    print(f"🔄 Converting LoRA checkpoint to inference format...")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"📋 Config: {hydra_config_path}")
    print(f"📁 Output: {output_dir}")
    
    try:
        import yaml
        
        # Read hydra config
        with open(hydra_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_model_path = config.get('pretrained_ckpt_path')
        lora_config = config.get('model', {}).get('model', {}).get('lora_config')
        
        if not base_model_path or not lora_config:
            print(f"❌ Could not extract required info from config")
            return None
        
        # Determine LoRA config name
        r = lora_config.get('r', 8)
        alpha = lora_config.get('lora_alpha', 16)
        lora_config_name = f"r_{r}_alpha_{alpha}"
        
        print(f"🔧 LoRA config: {lora_config_name} (r={r}, alpha={alpha})")
        print(f"🎯 Base model: {base_model_path}")
        
        # Build merge_lora command
        fish_speech_dir = Path("fish-speech")
        
        merge_cmd = [
            sys.executable, "-m", "tools.llama.merge_lora",
            "--lora-config", lora_config_name,
            "--base-weight", str(base_model_path),
            "--lora-weight", str(checkpoint_path.resolve()),
            "--output", str(output_dir.resolve())
        ]
        
        print(f"🚀 Running merge command...")
        result = subprocess.run(
            merge_cmd,
            capture_output=True,
            text=True,
            cwd=fish_speech_dir
        )
        
        if result.returncode == 0:
            print(f"✅ LoRA checkpoint converted successfully!")
            print(f"📁 Converted model: {output_dir}")
            return output_dir
        else:
            print(f"❌ LoRA conversion failed:")
            print(f"stderr: {result.stderr}")
            print(f"stdout: {result.stdout}")
            return None
            
    except Exception as e:
        print(f"❌ Error during LoRA conversion: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fish Speech CLI TTS")
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
    
    args = parser.parse_args()
    
    print("🐟 Fish Speech CLI TTS")
    print(f"💻 Device: {args.device}")
    if args.checkpoint:
        print(f"📦 Checkpoint: {args.checkpoint}")
    elif args.model_path:
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
    
    # Check MPS availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS not available, switching to CPU")
        args.device = "cpu"
    
    # Fish Speech setup
    fish_speech_dir, checkpoints_dir = setup_fish_speech(args.model_version, args.model_path)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_semantic_cache()
        return
    
    # Show cache information if requested
    if args.cache_info:
        cache_dir = Path("cache/semantic_tokens")
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
                # Read first line of text for preview
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
        success = create_voice_reference(audio_file, reference_file, checkpoints_dir, args.device)
        if success:
            print(f"🎭 Referencer created! Now use: --prompt-tokens {reference_file}")
        return
    
    # Check that text is provided for synthesis
    if not args.text:
        parser.error("Text is required for synthesis or --create-reference option")
    
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
        
        # Use found files
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
    
    # Speech synthesis with prosody and emotion parameters
    success = synthesize_speech_cli(
        args.text, 
        args.output, 
        args.device, 
        args.prompt_tokens,
        prompt_text,
        checkpoints_dir,
        use_cache=not args.no_cache,
        speed=args.speed,
        volume=args.volume,
        pitch=args.pitch,
        emotion=args.emotion,
        intensity=args.intensity,
        style=args.style,
        checkpoint=args.checkpoint
    )
    
    if success and args.play and shutil.which("afplay"):
        print("🔊 Playing...")
        subprocess.run(["afplay", args.output])

if __name__ == "__main__":
    main() 