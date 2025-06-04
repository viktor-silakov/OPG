#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import shutil
import os

# Check if the cloned repository exists
if not os.path.exists("fish-speech"):
    print("❌ Fish Speech not found in the current directory.")
    print("📝 Fish Speech is already included in this project.")
    print("🔧 Make sure it's installed: pip install -e ./fish-speech")
    exit(1)

# Change to the fish-speech directory
original_cwd = Path.cwd()
checkpoints_dir = Path("fish-speech") / "checkpoints" / "fish-speech-1.5"

print("🐟 Fish Speech TTS on Apple Silicon")
print(f"Working directory: {checkpoints_dir}")

# Download the model if it doesn't exist
if not checkpoints_dir.exists():
    print("📥 Downloading Fish Speech model...")
    cmd = [
        "huggingface-cli", "download", "fishaudio/fish-speech-1.5",
        "--local-dir", str(checkpoints_dir),
        "--local-dir-use-symlinks", "False"
    ]
    
    # Check if huggingface-cli is available
    if not shutil.which("huggingface-cli"):
        print("❌ huggingface-cli not found. Install via: pip install huggingface_hub")
        sys.exit(1)
    
    try:
        subprocess.run(cmd, check=True, cwd=checkpoints_dir)
        print("✅ Model downloaded")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)

# Change to the fish-speech directory for running
try:
    # Text to synthesize
    text = "Hello! Fish talks on Mac M3 through MPS"
    print(f"🎯 Synthesizing: {text}")
    
    # Create a simple reference audio example
    reference_dir = checkpoints_dir / "references" / "example"
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    reference_audio = reference_dir / "reference.wav"
    if not reference_audio.exists():
        print("🎵 Creating test audio...")
        import soundfile as sf
        import numpy as np
        
        sample_rate = 44100
        duration = 3.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(frequency * 2 * np.pi * t) * 0.3
        sf.write(reference_audio, wave, sample_rate)
    
    # Create a file with reference text
    reference_text = reference_dir / "reference.txt" 
    reference_text.write_text(text)
    
    # Use tools/api_server approach or WebUI
    print("🚀 Starting synthesis through Fish Speech WebUI...")
    print("This will start the web interface. Open your browser at http://localhost:7860")
    
    webui_cmd = [
        sys.executable, "-m", "tools.run_webui",
        "--llama-checkpoint-path", str(checkpoints_dir),
        "--decoder-checkpoint-path", str(checkpoints_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
        "--decoder-config-name", "firefly_gan_vq"
    ]
    
    print(f"Command: {' '.join(webui_cmd)}")
    print("Press Ctrl+C to stop")
    
    subprocess.run(webui_cmd, cwd=checkpoints_dir)
    
except KeyboardInterrupt:
    print("\n👋 Stopped by user request")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Return to the original directory
    pass

print("🎉 Done! Check the result in the web interface.") 