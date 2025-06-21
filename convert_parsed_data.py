#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def print_status(message, emoji="ℹ️"):
    """Prints status with emoji"""
    print(f"{emoji} {message}")

def convert_parsed_data():
    """Converts data from parsed_data to training format"""
    
    parsed_data_dir = Path("parsed_data")
    output_dir = Path("prepared_training_data/joyful")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all wav files
    wav_files = list(parsed_data_dir.glob("*.wav"))
    
    print_status(f"Found {len(wav_files)} audio files in parsed_data", "📁")
    
    converted_count = 0
    missing_txt_count = 0
    
    for wav_file in tqdm(wav_files, desc="Converting files"):
        # Find corresponding txt file
        txt_file = wav_file.with_suffix('.txt')
        
        if not txt_file.exists():
            print_status(f"⚠️ Missing txt file for {wav_file.name}", "⚠️")
            missing_txt_count += 1
            continue
        
        # Generate new filename with sequential numbering
        base_name = f"neutral_{converted_count:06d}"
        
        # Copy wav file
        output_wav = output_dir / f"{base_name}.wav"
        shutil.copy2(wav_file, output_wav)
        
        # Copy txt content to lab file
        output_lab = output_dir / f"{base_name}.lab"
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        with open(output_lab, 'w', encoding='utf-8') as f:
            f.write(content)
        
        converted_count += 1
    
    print_status(f"✅ Successfully converted {converted_count} file pairs", "✅")
    
    if missing_txt_count > 0:
        print_status(f"⚠️ Skipped {missing_txt_count} files due to missing txt files", "⚠️")
    
    print_status(f"📁 Files saved to: {output_dir}", "📁")
    print_status(f"🎉 Conversion completed!", "🎉")
    
    return converted_count

def main():
    print("🔄 Converting parsed_data to training format")
    print("="*50)
    
    # Check if parsed_data directory exists
    if not Path("parsed_data").exists():
        print_status("❌ parsed_data directory not found", "❌")
        return 1
    
    try:
        converted_count = convert_parsed_data()
        
        if converted_count > 0:
            print_status("💡 Now you can run fine-tuning with these prepared files", "💡")
            return 0
        else:
            print_status("❌ No files were converted", "❌")
            return 1
            
    except Exception as e:
        print_status(f"❌ Error during conversion: {e}", "❌")
        return 1

if __name__ == "__main__":
    exit(main()) 