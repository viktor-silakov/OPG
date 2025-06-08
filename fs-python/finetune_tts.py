#!/usr/bin/env python3

"""
Fish Speech Fine-tuning Pipeline

Features:
- Prepare dataset structure
- Extract semantic tokens
- Create protobuf datasets
- Fine-tune models with LoRA
- Resume training from checkpoints
- Merge LoRA weights
- List available checkpoints

Resume Training Options:
--resume-from-checkpoint path/to/checkpoint.ckpt  # Resume from specific checkpoint
--resume-latest                                   # Resume from latest checkpoint automatically
--list-checkpoints                               # Show available checkpoints
"""

import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
import time
import json

def print_status(message, emoji="ℹ️"):
    """Prints status with emoji"""
    print(f"{emoji} {message}")

def check_requirements():
    """Checks system requirements"""
    print_status("Checking system requirements", "🔍")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print_status("❌ Python 3.10 or higher is required", "❌")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_status(f"✅ CUDA GPU is available: {gpu_memory:.1f} GB of memory", "✅")
        if gpu_memory < 8:
            print_status("⚠️ Minimum 8GB GPU memory recommended for fine-tuning", "⚠️")
    elif torch.backends.mps.is_available():
        print_status("✅ MPS (Apple Silicon) is available", "✅")
    else:
        print_status("⚠️ GPU not found, CPU will be used (slow)", "⚠️")
    
    # Check fish-speech availability
    FISH_SPEECH_DIR = Path("fish-speech")
    if not FISH_SPEECH_DIR.exists():
        print_status("❌ Fish Speech not found", "🚫")
        print_status("📝 Fish Speech is already included in this project.", "📁")
        print_status("🔧 Make sure it's installed: pip install -e ./fish-speech", "💡")
        return False
    
    print_status("✅ All requirements are met", "✅")
    return True

def prepare_dataset_structure(input_dir, output_dir):
    """Prepares dataset structure in Fish Speech format"""
    print_status(f"Preparing dataset: {input_dir} -> {output_dir}", "📁")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_audio = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
    supported_text = {'.txt', '.lab'}
    
    files_processed = 0
    speakers_found = set()
    
    # Iterate through all files in the input directory
    for item in input_path.rglob("*"):
        if item.is_file():
            suffix = item.suffix.lower()
            
            if suffix in supported_audio:
                # Determine speaker from folder structure
                relative_path = item.relative_to(input_path)
                if len(relative_path.parts) > 1:
                    speaker = relative_path.parts[0]
                else:
                    speaker = "SPK1"  # Default speaker
                
                speakers_found.add(speaker)
                
                # Create speaker folder
                speaker_dir = output_path / speaker
                speaker_dir.mkdir(exist_ok=True)
                
                # Copy audio file
                audio_stem = item.stem
                target_audio = speaker_dir / f"{audio_stem}{suffix}"
                shutil.copy2(item, target_audio)
                
                # Find corresponding text file
                text_file = None
                for text_suffix in supported_text:
                    potential_text = item.with_suffix(text_suffix)
                    if potential_text.exists():
                        text_file = potential_text
                        break
                
                if text_file:
                    # Copy text file as .lab
                    target_text = speaker_dir / f"{audio_stem}.lab"
                    shutil.copy2(text_file, target_text)
                    files_processed += 1
                else:
                    print_status(f"⚠️ Text not found for {item.name}", "⚠️")
    
    print_status(f"✅ Prepared {files_processed} files for {len(speakers_found)} speakers", "✅")
    print_status(f"Speakers: {', '.join(sorted(speakers_found))}", "🎤")
    
    return files_processed > 0

def extract_semantic_tokens(data_dir, checkpoints_dir, batch_size=16, num_workers=1, device="auto"):
    """Extracts semantic tokens from audio"""
    print_status("Extracting semantic tokens", "🔬")
    
    fish_speech_dir = Path("fish-speech")
    data_path = Path(data_dir).resolve()  # Make absolute path
    
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print_status(f"Using device: {device}", "💻")
    
    # Command for extracting VQ tokens
    cmd = [
        sys.executable, "tools/vqgan/extract_vq.py", str(data_path),
        "--num-workers", str(num_workers),
        "--batch-size", str(batch_size),
        "--config-name", "firefly_gan_vq",
        "--checkpoint-path", str(checkpoints_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
        "--device", device
    ]
    
    print_status(f"Command: {' '.join(cmd)}", "🖥️")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=fish_speech_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print_status(f"❌ Extraction error:\n{result.stderr}", "❌")
            return False
        
        print_status("✅ Semantic tokens extracted", "✅")
        
        # Check result
        npy_files = list(data_path.rglob("*.npy"))
        print_status(f"Created {len(npy_files)} .npy files", "📊")
        
        return True
        
    except subprocess.TimeoutExpired:
        print_status("❌ Extraction timeout", "❌")
        return False
    except Exception as e:
        print_status(f"❌ Error: {e}", "❌")
        return False

def build_protobuf_dataset(data_dir, output_dir="data/protos", num_workers=16):
    """Creates protobuf dataset for training"""
    print_status("Creating protobuf dataset", "📦")
    
    fish_speech_dir = Path("fish-speech")
    data_path = Path(data_dir).resolve()  # Make absolute path
    output_path = fish_speech_dir / output_dir  # Path relative to fish-speech
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "fish-speech/tools/llama/build_dataset.py",
        "--input", str(data_path),
        "--output", str(output_path),
        "--text-extension", ".lab",
        "--num-workers", str(num_workers)
    ]
    
    print_status(f"Command: {' '.join(cmd)}", "🖥️")
    
    try:
        result = subprocess.run(
            cmd,
            # Run from main directory, not entering fish-speech
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            print_status(f"❌ Creation error:\n{result.stderr}", "❌")
            return False
        
        # Check result
        proto_files = list(output_path.glob("*.protos"))
        if proto_files:
            print_status(f"✅ Created protobuf dataset: {proto_files[0]}", "✅")
            return True
        else:
            print_status("❌ Protobuf file not created", "❌")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("❌ Creation timeout", "❌")
        return False
    except Exception as e:
        print_status(f"❌ Error: {e}", "❌")
        return False

def start_finetuning(project_name, checkpoints_dir, 
                    lora_config="r_8_alpha_16", 
                    max_steps=1000,
                    batch_size=4,
                    learning_rate=1e-4,
                    device="auto",
                    force_mps=False,
                    resume_from_checkpoint=None):
    """Starts fine-tuning process with LoRA"""
    if resume_from_checkpoint:
        print_status(f"Resuming fine-tuning project: {project_name} from checkpoint", "🔄")
    else:
        print_status(f"Starting fine-tuning project: {project_name}", "🚀")
    
    fish_speech_dir = Path("fish-speech")
    
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Determine path for initial model (base model or checkpoint for resume)
    if resume_from_checkpoint:
        # Use existing checkpoint as starting point
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            print_status(f"❌ Checkpoint not found: {resume_from_checkpoint}", "❌")
            return False
        pretrained_path = checkpoint_path.parent  # Use parent directory for checkpoint
        print_status(f"📂 Resuming from: {checkpoint_path}", "📂")
    else:
        # Use base model
        # Check standard HuggingFace cache location
        hf_cache_path = Path.home() / ".cache/huggingface/hub/models--fishaudio--fish-speech-1.5/snapshots"
        if hf_cache_path.exists():
            snapshots = list(hf_cache_path.iterdir())
            if snapshots:
                pretrained_path = snapshots[0]  # Take first available snapshot
            else:
                pretrained_path = checkpoints_dir
        else:
            pretrained_path = checkpoints_dir
    
    # Command for training with optimized settings
    cmd = [
        sys.executable, "fish_speech/train.py",
        "--config-name", "text2semantic_finetune",
        f"project={project_name}",
        f"+lora@model.model.lora_config={lora_config}",
        f"trainer.max_steps={max_steps}",
        f"data.batch_size={min(batch_size, 16)}",  # Limit batch size for memory economy
        f"model.optimizer.lr={learning_rate}",
        "trainer.accelerator=cpu",  # Force using CPU for stability
        "trainer.devices=1",
        "trainer.strategy=auto",
        "data.num_workers=0",  # Set 0 workers to avoid memory issues
        "data.max_length=512",  # Limit sequence length
    ]
    
    # Add checkpoint path based on mode
    if resume_from_checkpoint:
        cmd.append(f"ckpt_path={resume_from_checkpoint}")
        print_status(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}", "🔄")
    else:
        cmd.append(f"pretrained_ckpt_path={pretrained_path}")
        print_status(f"🆕 Starting from base model: {pretrained_path}", "🆕")
    
    # Additional optimizations for memory economy
    if max_steps > 100:
        # For long training, add additional constraints
        cmd.append("trainer.limit_train_batches=10")  # Limit number of batches per epoch
    
    print_status("🔧 Using optimized settings for memory economy:", "🔧")
    if resume_from_checkpoint:
        print_status(f"   • Mode: Resume training from checkpoint", "🔄")
    else:
        print_status(f"   • Mode: New training from base model", "🆕")
    print_status(f"   • Device: CPU (for stability)", "💻")
    print_status(f"   • Batch size: {min(batch_size, 16)}", "📊")
    print_status(f"   • Data workers: 0", "👷")
    print_status(f"   • Maximum length: 512 tokens", "📏")
    print_status(f"   • Maximum steps: {max_steps}", "📈")
    print_status(f"   • Learning rate: {learning_rate}", "🎯")
    print_status(f"   • LoRA config: {lora_config}", "🔧")
    print_status("💡 All our BFloat16→Float32 conversions are already applied in the code", "💡")
    
    print_status("Training command:", "🖥️")
    print(" ".join(cmd))
    print()
    
    try:
        print_status("▶️ Starting training...", "▶️")
        print_status("⚠️ If process is killed (-9), try reducing batch_size", "⚠️")
        print_status("Press Ctrl+C to stop", "⚠️")
        
        # Start training in real time
        process = subprocess.Popen(
            cmd,
            cwd=fish_speech_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Output logs in real time
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print_status("✅ Training completed successfully!", "✅")
            
            # Find checkpoints
            results_dir = fish_speech_dir / "results" / project_name / "checkpoints"
            if results_dir.exists():
                checkpoints = list(results_dir.glob("*.ckpt"))
                if checkpoints:
                    print_status(f"📁 Checkpoints saved in: {results_dir}", "📁")
                    for ckpt in sorted(checkpoints):
                        print_status(f"   📄 {ckpt.name}", "📄")
            
            return True
        elif process.returncode == -9:
            print_status("❌ ❌ Training failed with error (code: -9)", "❌")
            print_status("❌ ❌ Training error", "❌")
            print_status("💡 Try:", "💡")
            print_status("   • Reduce batch-size: --batch-size 1", "💡")
            print_status("   • Reduce max-steps: --max-steps 10", "💡")
            print_status("   • Close other applications to free memory", "💡")
            return False
        else:
            print_status(f"❌ Training failed with error (code: {process.returncode})", "❌")
            return False
            
    except KeyboardInterrupt:
        print_status("⏹️ Training interrupted by user", "⏹️")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print_status(f"❌ Training error: {e}", "❌")
        return False

def merge_lora_weights(project_name, checkpoints_dir, lora_config="r_8_alpha_16", checkpoint_step=None):
    """Merges LoRA weights with base model"""
    print_status("Merging LoRA weights with base model", "🔗")
    
    fish_speech_dir = Path("fish-speech")
    results_dir = fish_speech_dir / "results" / project_name / "checkpoints"
    
    if not results_dir.exists():
        print_status(f"❌ Checkpoint folder not found: {results_dir}", "❌")
        return False
    
    # Find suitable checkpoint
    checkpoints = list(results_dir.glob("*.ckpt"))
    if not checkpoints:
        print_status("❌ Checkpoints not found", "❌")
        return False
    
    if checkpoint_step:
        # Find specific step
        target_ckpt = None
        for ckpt in checkpoints:
            if f"step_{checkpoint_step:09d}" in ckpt.name:
                target_ckpt = ckpt
                break
        if not target_ckpt:
            print_status(f"❌ Checkpoint for step {checkpoint_step} not found", "❌")
            return False
    else:
        # Take last checkpoint
        target_ckpt = sorted(checkpoints)[-1]
    
    print_status(f"Using checkpoint: {target_ckpt.name}", "📄")
    
    # Output folder for merged model (absolute path)
    output_dir = Path.cwd() / "checkpoints" / f"{project_name}-merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use absolute paths for all files
    target_ckpt_abs = target_ckpt.resolve()
    output_dir_abs = output_dir.resolve()
    
    cmd = [
        sys.executable, "tools/llama/merge_lora.py",
        "--lora-config", lora_config,
        "--base-weight", str(checkpoints_dir),
        "--lora-weight", str(target_ckpt_abs),  # Use absolute path
        "--output", str(output_dir_abs)         # Use absolute path
    ]
    
    print_status(f"Command: {' '.join(cmd)}", "🖥️")
    print_status(f"Checkpoint path: {target_ckpt_abs}", "📄")
    print_status(f"Output folder: {output_dir_abs}", "📁")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=fish_speech_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            print_status(f"❌ Merging weights error:\n{result.stderr}", "❌")
            return False
        
        print_status(f"✅ Model merged and saved in: {output_dir_abs}", "✅")
        
        # Check result
        model_files = list(output_dir_abs.glob("*.pth"))
        tokenizer_files = list(output_dir_abs.glob("*.tiktoken"))
        
        if model_files and tokenizer_files:
            print_status("📄 Merged model files:", "📄")
            for file in sorted(output_dir_abs.iterdir()):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print_status(f"   {file.name}: {size_mb:.1f} MB", "📄")
            
            return str(output_dir_abs)
        else:
            print_status("❌ Not all model files created", "❌")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("❌ Merging weights timeout", "❌")
        return False
    except Exception as e:
        print_status(f"❌ Error: {e}", "❌")
        return False

def download_base_model(model_version="1.5"):
    """Downloads base Fish Speech model"""
    print_status(f"Downloading base Fish Speech model {model_version}", "📥")
    
    version_mapping = {
        "1.4": "fishaudio/fish-speech-1.4",
        "1.5": "fishaudio/fish-speech-1.5", 
        "1.6": "fishaudio/fish-speech-1.6"
    }
    
    repo_id = version_mapping.get(model_version, "fishaudio/fish-speech-1.5")
    
    try:
        checkpoints_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model"
        )
        print_status(f"✅ Model downloaded: {checkpoints_dir}", "✅")
        return Path(checkpoints_dir)
    except Exception as e:
        print_status(f"❌ Model download error: {e}", "❌")
        return None

def list_checkpoints(project_name):
    """Lists available checkpoints for a project"""
    fish_speech_dir = Path("fish-speech")
    results_dir = fish_speech_dir / "results" / project_name / "checkpoints"
    
    if not results_dir.exists():
        print_status(f"❌ No results directory found for project: {project_name}", "❌")
        return False
    
    checkpoints = list(results_dir.glob("*.ckpt"))
    if not checkpoints:
        print_status(f"❌ No checkpoints found for project: {project_name}", "❌")
        return False
    
    print_status(f"📂 Available checkpoints for project '{project_name}':", "📂")
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ckpt.stat().st_mtime))
        print_status(f"   📄 {ckpt.name} ({size_mb:.1f} MB, {modified_time})", "📄")
    
    latest = sorted(checkpoints)[-1]
    print_status(f"🎯 Latest: {latest.name}", "🎯")
    return True

def find_latest_checkpoint(project_name):
    """Finds latest checkpoint for a project"""
    fish_speech_dir = Path("fish-speech")
    results_dir = fish_speech_dir / "results" / project_name / "checkpoints"
    
    if not results_dir.exists():
        return None
    
    checkpoints = list(results_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # Sort by step number and return latest
    latest_checkpoint = sorted(checkpoints)[-1]
    return str(latest_checkpoint.resolve())

def create_training_config(project_name, max_steps, batch_size, learning_rate, lora_config):
    """Creates training configuration file"""
    config = {
        "project_name": project_name,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_config": lora_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    config_file = Path("training_configs") / f"{project_name}_config.json"
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print_status(f"📝 Configuration saved: {config_file}", "📝")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Fish Speech Fine-tuning Pipeline")
    
    # Main parameters
    parser.add_argument("--project", required=True, help="Project name for fine-tuning")
    parser.add_argument("--data-dir", required=True, help="Folder with audio and text files")
    parser.add_argument("--model-version", default="1.5", choices=["1.4", "1.5", "1.6"], 
                       help="Base model version of Fish Speech")
    
    # Dataset parameters
    parser.add_argument("--prepare-data", action="store_true", 
                       help="Prepare dataset structure")
    parser.add_argument("--prepared-data-dir", help="Path to prepared dataset")
    
    # Token extraction parameters
    parser.add_argument("--extract-tokens", action="store_true", 
                       help="Extract semantic tokens")
    parser.add_argument("--batch-size-extract", type=int, default=16, 
                       help="Batch size for token extraction")
    parser.add_argument("--num-workers-extract", type=int, default=1, 
                       help="Number of worker's for token extraction")
    parser.add_argument("--device-extract", default="cpu", 
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device for token extraction (recommended cpu)")
    
    # Training parameters
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--max-steps", type=int, default=1000, 
                       help="Maximum number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--lora-config", default="r_8_alpha_16", 
                       help="LoRA configuration")
    parser.add_argument("--resume-from-checkpoint", 
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--resume-latest", action="store_true",
                       help="Automatically resume from latest checkpoint of the same project")
    parser.add_argument("--list-checkpoints", action="store_true",
                       help="List available checkpoints for the project")
    
    # Weights merging parameters
    parser.add_argument("--merge-weights", action="store_true", 
                       help="Merge LoRA weights with base model")
    parser.add_argument("--checkpoint-step", type=int, 
                       help="Specific checkpoint step for merging")
    
    # System parameters
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], 
                       help="Device for training")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip checking requirements")
    parser.add_argument("--force-mps", action="store_true", 
                       help="Force using MPS for training (experimentally)")
    
    # Pipeline
    parser.add_argument("--full-pipeline", action="store_true", 
                       help="Perform full pipeline: preparation -> tokens -> protobuf -> training -> merging")
    
    args = parser.parse_args()
    
    print("🐟 Fish Speech Fine-tuning Pipeline")
    print("="*50)
    
    # Handle list checkpoints command
    if args.list_checkpoints:
        if list_checkpoints(args.project):
            return 0
        else:
            return 1
    
    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            return 1
    
    # Define working directories
    project_name = args.project
    raw_data_dir = Path(args.data_dir)
    prepared_data_dir = Path(args.prepared_data_dir) if args.prepared_data_dir else Path("data") / project_name
    
    # Download base model
    print_status("Preparing base model", "📦")
    checkpoints_dir = download_base_model(args.model_version)
    if not checkpoints_dir:
        return 1
    
    success = True
    
    # Perform pipeline steps
    if args.full_pipeline or args.prepare_data:
        print_status("Step 1: Preparing dataset", "1️⃣")
        if not prepare_dataset_structure(raw_data_dir, prepared_data_dir):
            print_status("❌ Dataset preparation error", "❌")
            return 1
    
    if args.full_pipeline or args.extract_tokens:
        print_status("Step 2: Extracting semantic tokens", "2️⃣")
        data_dir_for_extraction = prepared_data_dir if args.prepare_data or args.full_pipeline else raw_data_dir
        if not extract_semantic_tokens(
            data_dir_for_extraction, 
            checkpoints_dir, 
            batch_size=args.batch_size_extract,
            num_workers=args.num_workers_extract,
            device=args.device_extract
        ):
            print_status("❌ Token extraction error", "❌")
            return 1
    
    if args.full_pipeline or (args.extract_tokens and args.train):
        print_status("Step 3: Creating protobuf dataset", "3️⃣")
        data_dir_for_protobuf = prepared_data_dir if args.prepare_data or args.full_pipeline else raw_data_dir
        if not build_protobuf_dataset(data_dir_for_protobuf):
            print_status("❌ Protobuf creation error", "❌")
            return 1
    
    if args.full_pipeline or args.train:
        print_status("Step 4: Fine-tuning model", "4️⃣")
        
        # Determine checkpoint for resuming
        resume_checkpoint = None
        if args.resume_latest:
            resume_checkpoint = find_latest_checkpoint(project_name)
            if resume_checkpoint:
                print_status(f"🔍 Found latest checkpoint: {resume_checkpoint}", "🔍")
            else:
                print_status(f"⚠️ No checkpoints found for project {project_name}, starting from scratch", "⚠️")
        elif args.resume_from_checkpoint:
            resume_checkpoint = args.resume_from_checkpoint
            print_status(f"📂 Using specified checkpoint: {resume_checkpoint}", "📂")
        
        # Validate resume parameters
        if args.resume_latest and args.resume_from_checkpoint:
            print_status("❌ Cannot use both --resume-latest and --resume-from-checkpoint", "❌")
            return 1
        
        # Create training config
        create_training_config(
            project_name, 
            args.max_steps, 
            args.batch_size, 
            args.learning_rate, 
            args.lora_config
        )
        
        if not start_finetuning(
            project_name,
            checkpoints_dir,
            args.lora_config,
            args.max_steps,
            args.batch_size,
            args.learning_rate,
            args.device,
            args.force_mps,
            resume_checkpoint
        ):
            print_status("❌ Training error", "❌")
            return 1
    
    if args.full_pipeline or args.merge_weights:
        print_status("Step 5: Merging LoRA weights", "5️⃣")
        merged_model_path = merge_lora_weights(
            project_name,
            checkpoints_dir,
            args.lora_config,
            args.checkpoint_step
        )
        if not merged_model_path:
            print_status("❌ Merging weights error", "❌")
            return 1
        
        print_status("🎉 Fine-tuning completed successfully!", "🎉")
        print_status(f"📁 Final model: {merged_model_path}", "📁")
        print_status("💡 For use, specify model path in cli_tts.py", "💡")
    
    return 0

if __name__ == "__main__":
    exit(main()) 