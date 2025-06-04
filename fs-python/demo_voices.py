#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def run_demo():
    """Demonstration of working with Fish Speech voices"""
    
    print("🎭 Demonstration of working with reference voices Fish Speech")
    print("=" * 60)
    
    # Check if reference exists
    reference_file = Path("voices/test_voice_reference.npy")
    
    if not reference_file.exists():
        print("❌ Reference file not found. Creating...")
        
        # Create test audio if it doesn't exist
        test_audio = Path("voices/test_voice.wav")
        if not test_audio.exists():
            print("📢 Creating test audio...")
            subprocess.run([sys.executable, "create_test_audio.py"])
        
        # Create reference
        print("🎙️ Creating reference voice...")
        cmd = [
            "poetry", "run", "python", "cli_tts.py",
            "--create-reference", str(test_audio), str(reference_file)
        ]
        subprocess.run(cmd)
    
    # Test phrases
    test_phrases = [
        "Welcome to Fish Speech!",
        "This is a demonstration of voice cloning.",
        "Technology allows you to produce different voices.",
        "Fish Speech works on Apple Silicon through MPS."
    ]
    
    print("\n🔊 Generating speech samples:")
    print("-" * 40)
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n{i}. Phrase: '{phrase}'")
        
        # Normal voice (without reference)
        normal_file = f"demo_normal_{i}.wav"
        print(f"   📝 Normal voice → {normal_file}")
        cmd = [
            "poetry", "run", "python", "cli_tts.py",
            phrase, "-o", normal_file
        ]
        subprocess.run(cmd, capture_output=True)
        
        # With reference
        ref_file = f"demo_reference_{i}.wav"
        print(f"   🎭 With reference → {ref_file}")
        cmd = [
            "poetry", "run", "python", "cli_tts.py",
            phrase,
            "--prompt-tokens", str(reference_file),
            "--prompt-text", "Test voice for demonstration",
            "-o", ref_file
        ]
        subprocess.run(cmd, capture_output=True)
    
    print("\n✅ Demonstration completed!")
    print("\nCreated files:")
    print("📁 Normal voices: demo_normal_*.wav")
    print("📁 With reference: demo_reference_*.wav")
    print("\nTo play:")
    print("🔊 afplay demo_normal_1.wav")
    print("🔊 afplay demo_reference_1.wav")
    
    # Show file sizes
    print("\n📊 File statistics:")
    demo_files = list(Path(".").glob("demo_*.wav"))
    for file in sorted(demo_files):
        size = file.stat().st_size
        print(f"   {file.name}: {size:,} bytes")

if __name__ == "__main__":
    run_demo() 