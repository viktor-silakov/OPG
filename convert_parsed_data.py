#!/usr/bin/env python3

import os
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm

emotion = "angry"
parsed_data_dir = Path(f"./parsed_data/{emotion}")
output_dir = Path(f"./prepared_training_data/{emotion}")
    
def print_status(message, emoji="ℹ️"):
    """Prints status with emoji"""
    print(f"{emoji} {message}")

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_and_remove_duplicates(wav_files):
    """Find and remove duplicate wav files based on hash"""
    hash_to_file = {}
    duplicates = []
    
    print_status("Checking for duplicate files...", "🔍")
    
    for wav_file in tqdm(wav_files, desc="Calculating hashes"):
        try:
            file_hash = calculate_file_hash(wav_file)
            
            if file_hash in hash_to_file:
                # Found duplicate
                duplicates.append(wav_file)
                print_status(f"Duplicate found: {wav_file.name} (same as {hash_to_file[file_hash].name})", "🔄")
            else:
                hash_to_file[file_hash] = wav_file
                
        except Exception as e:
            print_status(f"Error processing {wav_file.name}: {e}", "❌")
    
    # Remove duplicates
    removed_count = 0
    for duplicate_wav in duplicates:
        try:
            # Remove wav file
            duplicate_wav.unlink()
            print_status(f"Removed duplicate wav: {duplicate_wav.name}", "🗑️")
            
            # Remove corresponding txt file if exists
            duplicate_txt = duplicate_wav.with_suffix('.txt')
            if duplicate_txt.exists():
                duplicate_txt.unlink()
                print_status(f"Removed corresponding txt: {duplicate_txt.name}", "🗑️")
            
            removed_count += 1
            
        except Exception as e:
            print_status(f"Error removing {duplicate_wav.name}: {e}", "❌")
    
    if removed_count > 0:
        print_status(f"Removed {removed_count} duplicate file pairs", "✅")
    else:
        print_status("No duplicates found", "✅")
    
    # Return remaining files
    remaining_files = [f for f in wav_files if f not in duplicates]
    return remaining_files

def convert_parsed_data():
    """Converts data from parsed_data to training format"""
    

    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all wav files
    wav_files = list(parsed_data_dir.glob("*.wav"))
    
    print_status(f"Found {len(wav_files)} audio files in parsed_data", "📁")
    
    # Remove duplicates first
    wav_files = find_and_remove_duplicates(wav_files)
    print_status(f"Processing {len(wav_files)} unique audio files", "📁")
    
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
        base_name = f"{emotion}_{converted_count:06d}"
        
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
    # if not Path("parsed_data").exists():
    #     print_status("❌ parsed_data directory not found", "❌")
    #     return 1
    
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