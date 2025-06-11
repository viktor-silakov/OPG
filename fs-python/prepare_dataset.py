#!/usr/bin/env python3

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
import json
import librosa
import soundfile as sf
from tqdm import tqdm
try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence, split_on_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub not installed. Install: pip install pydub")
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        USING_FASTER_WHISPER = False
    except ImportError:
        WHISPER_AVAILABLE = False
        print("⚠️ Whisper not installed. Install: pip install openai-whisper or pip install faster-whisper")
    else:
        USING_FASTER_WHISPER = False
else:
    USING_FASTER_WHISPER = True

def print_status(message, emoji="ℹ️"):
    """Prints status with emoji"""
    print(f"{emoji} {message}")

def get_audio_duration(audio_path):
    """Gets audio file duration"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print_status(f"⚠️ Unable to get duration of {audio_path}: {e}", "⚠️")
        return 0

def normalize_audio(input_path, output_path, target_loudness=-23.0, sample_rate=44100):
    """Normalizes audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        
        # Normalize volume
        # Simple normalization by peak value
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            # Normalize to 90% of the maximum to avoid clipping
            audio = audio * (0.9 / max_val)
        
        # Save
        sf.write(output_path, audio, sample_rate)
        return True
        
    except Exception as e:
        print_status(f"❌ Normalization error {input_path}: {e}", "❌")
        return False

def segment_audio_by_silence(
    audio_path, 
    text_content, 
    output_dir, 
    min_silence_len=500, 
    silence_thresh=-32, 
    min_segment_len=1.5, 
    max_segment_len=10.0, 
    max_total_duration=None
):
    """Segments audio by silence detection using pydub"""
    print_status(f"Segmenting audio by silence: {audio_path}", "🔇")
    
    if not PYDUB_AVAILABLE:
        print_status("⚠️ pydub not available, falling back to time-based splitting", "⚠️")
        return split_long_audio(audio_path, text_content, output_dir, max_segment_len, 1.0, max_total_duration)
    
    try:
        # Load audio with pydub
        print_status(f"Loading audio with pydub: {audio_path}", "📂")
        audio = AudioSegment.from_file(audio_path)
        
        # Normalize audio
        audio = audio.normalize()
        
        # Apply duration limit if specified
        if max_total_duration and max_total_duration > 0:
            max_duration_ms = int(max_total_duration * 60 * 1000)  # convert minutes to ms
            if len(audio) > max_duration_ms:
                audio = audio[:max_duration_ms]
                print_status(f"⏱️ Limited to {max_total_duration} minutes", "⏱️")
        
        # Detect silence
        print_status("Detecting silence periods...", "🔍")
        silences = detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        print_status(f"Found {len(silences)} silence periods", "📊")
        
        # Create segments based on silence
        segments = []
        start_ms = 0
        min_segment_ms = int(min_segment_len * 1000)
        max_segment_ms = int(max_segment_len * 1000)
        
        for silence_start, silence_end in silences:
            # Check if current segment is long enough
            segment_len = silence_start - start_ms
            
            if segment_len >= min_segment_ms:
                # If segment is too long, split it further
                if segment_len > max_segment_ms:
                    # Split long segment into equal parts
                    num_parts = int(segment_len // max_segment_ms) + 1
                    part_duration = segment_len // num_parts
                    
                    for i in range(num_parts):
                        part_start = start_ms + (i * part_duration)
                        part_end = part_start + part_duration if i < num_parts - 1 else silence_start
                        if part_end - part_start >= min_segment_ms:
                            segments.append((part_start, part_end))
                else:
                    segments.append((start_ms, silence_start))
            
            start_ms = silence_end
        
        # Add final segment if it's long enough
        if len(audio) - start_ms >= min_segment_ms:
            final_len = len(audio) - start_ms
            if final_len > max_segment_ms:
                # Split final long segment
                num_parts = int(final_len // max_segment_ms) + 1
                part_duration = final_len // num_parts
                
                for i in range(num_parts):
                    part_start = start_ms + (i * part_duration)
                    part_end = part_start + part_duration if i < num_parts - 1 else len(audio)
                    if part_end - part_start >= min_segment_ms:
                        segments.append((part_start, part_end))
            else:
                segments.append((start_ms, len(audio)))
        
        print_status(f"Created {len(segments)} segments", "✅")
        
        # Save segments
        output_dir = Path(output_dir)
        segment_files = []
        
        # Split text evenly among segments
        words = text_content.split() if text_content else []
        words_per_segment = len(words) // len(segments) if segments and words else 0
        
        for i, (start_ms, end_ms) in enumerate(segments):
            # Extract audio segment
            segment = audio[start_ms:end_ms]
            
            # Save segment
            segment_name = f"{Path(audio_path).stem}_{i+1:04d}.wav"
            segment_path = output_dir / segment_name
            segment.export(str(segment_path), format="wav")
            
            # Extract corresponding text
            if words:
                start_word = i * words_per_segment
                end_word = (i + 1) * words_per_segment if i < len(segments) - 1 else len(words)
                segment_text = " ".join(words[start_word:end_word])
            else:
                segment_text = f"Silence-based segment {i+1} from {Path(audio_path).stem}"
            
            segment_files.append((segment_path, segment_text))
        
        print_status(f"✅ Created {len(segment_files)} silence-based segments", "✅")
        return segment_files
        
    except Exception as e:
        print_status(f"❌ Silence segmentation error: {e}", "❌")
        print_status("Falling back to time-based splitting", "⏪")
        return split_long_audio(audio_path, text_content, output_dir, max_segment_len, 1.0, max_total_duration)

def split_long_audio(audio_path, text_content, output_dir, max_duration=10.0, overlap=1.0, max_total_duration=None):
    """Splits long audio into segments"""
    print_status(f"Splitting audio into {max_duration}s segments: {audio_path}", "✂️")
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100, mono=True)
        total_duration = len(audio) / sr
        
        # Limit duration if specified
        if max_total_duration and max_total_duration > 0:
            max_samples = int(max_total_duration * 60 * sr)  # convert minutes to samples
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                total_duration = max_total_duration * 60
                print_status(f"⏱️ Limited to {max_total_duration} minutes ({total_duration/60:.1f} min)", "⏱️")
        
        if total_duration <= max_duration:
            # If file is shorter than max duration, return as is
            return [(audio_path, text_content)]
        
        segments = []
        segment_duration = max_duration
        overlap_samples = int(overlap * sr)
        segment_samples = int(segment_duration * sr)
        
        # Calculate number of segments
        step_samples = segment_samples - overlap_samples
        num_segments = int((len(audio) - overlap_samples) / step_samples) + 1
        
        print_status(f"Creating {num_segments} segments of {max_duration}s", "📊")
        
        # Split text evenly among segments
        words = text_content.split() if text_content else []
        words_per_segment = len(words) // num_segments if words else 0
        
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = min(start_sample + segment_samples, len(audio))
            
            # Check minimum segment length (at least 3 seconds)
            segment_duration_actual = (end_sample - start_sample) / sr
            if segment_duration_actual < 3.0 and i > 0:
                # Too short segment at the end - skip
                continue
            
            # Extract audio segment
            audio_segment = audio[start_sample:end_sample]
            
            # Save segment with correct naming
            segment_name = f"{Path(audio_path).stem}_{i+1:04d}.wav"
            segment_path = output_dir / segment_name
            sf.write(segment_path, audio_segment, sr)
            
            # Extract corresponding text
            if words:
                start_word = i * words_per_segment
                end_word = (i + 1) * words_per_segment if i < num_segments - 1 else len(words)
                segment_text = " ".join(words[start_word:end_word])
            else:
                segment_text = f"Segment {i+1} from {Path(audio_path).stem}"
            
            segments.append((segment_path, segment_text))
        
        print_status(f"✅ Created {len(segments)} segments", "✅")
        return segments
        
    except Exception as e:
        print_status(f"❌ Splitting audio error: {e}", "❌")
        return [(audio_path, text_content)]

def transcribe_audio_whisper(audio_path, model_size="base", language="auto"):
    """Transcribes audio using Whisper"""
    print_status(f"Transcribing with Whisper ({model_size}): {audio_path}", "🎤")
    
    if not WHISPER_AVAILABLE:
        print_status("❌ Whisper not available", "❌")
        return None
    
    try:
        if USING_FASTER_WHISPER:
            # Use faster-whisper
            model = WhisperModel(model_size)
            
            # Transcribe
            if language == "auto":
                segments, info = model.transcribe(str(audio_path))
            else:
                segments, info = model.transcribe(str(audio_path), language=language)
            
            # Combine all segments into one text
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        else:
            # Use original whisper
            model = whisper.load_model(model_size)
            
            # Transcribe
            if language == "auto":
                result = model.transcribe(str(audio_path))
            else:
                result = model.transcribe(str(audio_path), language=language)
            
            return result["text"].strip()
        
    except Exception as e:
        print_status(f"❌ Transcribing error: {e}", "❌")
        return None

def process_youtube_audio(url, output_dir, speaker_name="SPEAKER", segment_duration=10.0, max_duration_minutes=None, use_silence_detection=False, min_silence_len=500, silence_thresh=-32, min_segment_len=1.5):
    """Processes audio from YouTube and automatically cuts it into segments"""
    print_status(f"Loading audio from YouTube: {url}", "📺")
    
    if max_duration_minutes:
        print_status(f"⏱️ Limitation: first {max_duration_minutes} minutes", "⏱️")
    
    if use_silence_detection and not PYDUB_AVAILABLE:
        print_status("⚠️ pydub not available, disabling silence detection", "⚠️")
        use_silence_detection = False
    
    try:
        # Check for yt-dlp
        if not shutil.which("yt-dlp"):
            print_status("❌ Requires yt-dlp: pip install yt-dlp", "❌")
            return []
        
        # Create temporary directory
        temp_dir = output_dir / "temp_youtube"
        temp_dir.mkdir(exist_ok=True)
        
        # Download audio
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", str(temp_dir / "%(title)s.%(ext)s"),
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_status(f"❌ Download error: {result.stderr}", "❌")
            return []
        
        # Find downloaded files
        audio_files = list(temp_dir.glob("*.wav"))
        if not audio_files:
            print_status("❌ Audio files not found", "❌")
            return []
        
        processed_files = []
        
        # Create speaker folder
        speaker_dir = output_dir / speaker_name
        speaker_dir.mkdir(exist_ok=True)
        
        for audio_file in audio_files:
            print_status(f"Processing: {audio_file.name}", "🎵")
            
            # Normalize filename
            clean_name = "".join(c for c in audio_file.stem if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')
            
            # Normalize audio to temporary file
            normalized_audio = temp_dir / f"{clean_name}_normalized.wav"
            if not normalize_audio(audio_file, normalized_audio):
                print_status(f"⚠️ Unable to normalize {audio_file.name}", "⚠️")
                normalized_audio = audio_file
            
            # Get audio duration
            duration = get_audio_duration(normalized_audio)
            original_duration_minutes = duration / 60
            print_status(f"Original duration: {original_duration_minutes:.1f} minutes", "⏱️")
            
            # Calculate final duration
            if max_duration_minutes and max_duration_minutes < original_duration_minutes:
                final_duration_minutes = max_duration_minutes
                print_status(f"Will be processed: {final_duration_minutes:.1f} minutes", "✂️")
            else:
                final_duration_minutes = original_duration_minutes
                print_status(f"Will be processed: {final_duration_minutes:.1f} minutes (fully)", "✅")
            
            # Segment audio
            placeholder_text = f"Auto transcription for {clean_name}"
            
            if use_silence_detection:
                print_status(f"Using silence detection for segmentation", "🔇")
                segments = segment_audio_by_silence(
                    normalized_audio, 
                    placeholder_text, 
                    speaker_dir,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_thresh,
                    min_segment_len=min_segment_len,
                    max_segment_len=segment_duration,
                    max_total_duration=max_duration_minutes
                )
            else:
                segments = split_long_audio(
                    normalized_audio, 
                    placeholder_text, 
                    speaker_dir, 
                    max_duration=segment_duration,
                    overlap=1.0,
                    max_total_duration=max_duration_minutes
                )
            
            # Add all segments to processed files list
            for segment_path, segment_text in segments:
                processed_files.append(segment_path)
        
        # Delete temporary directory
        shutil.rmtree(temp_dir)
        
        estimated_duration = len(processed_files) * segment_duration / 60
        print_status(f"✅ Created {len(processed_files)} audio segments of {segment_duration}s", "✅")
        print_status(f"📊 Total dataset duration: ~{estimated_duration:.1f} minutes", "📊")
        print_status(f"📁 Saved in: {speaker_dir}", "📁")
        
        return processed_files
        
    except Exception as e:
        print_status(f"❌ YouTube processing error: {e}", "❌")
        return []

def process_directory(input_dir, output_dir, normalize=True, split_long=True, auto_transcribe=False, whisper_model="base", language="auto", use_silence_detection=False, min_silence_len=500, silence_thresh=-32, min_segment_len=1.5):
    """Processes directory with audio files"""
    print_status(f"Processing directory: {input_dir}", "📁")
    
    if use_silence_detection and not PYDUB_AVAILABLE:
        print_status("⚠️ pydub not available, disabling silence detection", "⚠️")
        use_silence_detection = False
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_audio = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
    supported_text = {'.txt', '.lab', '.srt'}
    
    total_files = 0
    total_duration = 0
    processed_speakers = set()
    
    # Collect all audio files
    audio_files = []
    for ext in supported_audio:
        audio_files.extend(input_path.rglob(f"*{ext}"))
    
    print_status(f"Found {len(audio_files)} audio files", "📊")
    
    if use_silence_detection:
        print_status(f"Using silence detection (thresh: {silence_thresh}dB, min silence: {min_silence_len}ms)", "🔇")
    
    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            # Determine speaker from folder structure
            relative_path = audio_file.relative_to(input_path)
            if len(relative_path.parts) > 1:
                speaker = relative_path.parts[0]
            else:
                speaker = "SPEAKER_1"
            
            processed_speakers.add(speaker)
            
            # Create speaker folder
            speaker_dir = output_path / speaker
            speaker_dir.mkdir(exist_ok=True)
            
            # Get duration
            duration = get_audio_duration(audio_file)
            total_duration += duration
            
            # Find text file
            text_content = ""
            text_file = None
            for text_ext in supported_text:
                potential_text = audio_file.with_suffix(text_ext)
                if potential_text.exists():
                    text_file = potential_text
                    break
            
            if text_file:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            elif auto_transcribe:
                print_status(f"🎤 Auto transcription: {audio_file.name}", "🎤")
                text_content = transcribe_audio_whisper(audio_file, whisper_model, language)
                if not text_content:
                    text_content = f"Auto transcription for {audio_file.name}"
            
            # Process long files
            if split_long and duration > 30:
                if use_silence_detection:
                    segments = segment_audio_by_silence(
                        audio_file, 
                        text_content, 
                        speaker_dir,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        min_segment_len=min_segment_len
                    )
                else:
                    segments = split_long_audio(audio_file, text_content, speaker_dir)
                    
                for seg_audio, seg_text in segments:
                    # Create .lab file
                    lab_file = seg_audio.with_suffix('.lab')
                    with open(lab_file, 'w', encoding='utf-8') as f:
                        f.write(seg_text)
                    total_files += 1
            else:
                # Regular processing
                final_audio = speaker_dir / f"{audio_file.stem}.wav"
                
                if normalize:
                    if not normalize_audio(audio_file, final_audio):
                        # If normalization failed, just copy
                        shutil.copy2(audio_file, final_audio)
                else:
                    # Convert to WAV if needed
                    if audio_file.suffix.lower() != '.wav':
                        audio, sr = librosa.load(audio_file, sr=44100)
                        sf.write(final_audio, audio, sr)
                    else:
                        shutil.copy2(audio_file, final_audio)
                
                # Create .lab file
                lab_file = final_audio.with_suffix('.lab')
                with open(lab_file, 'w', encoding='utf-8') as f:
                    f.write(text_content or f"Audio from {audio_file.name}")
                
                total_files += 1
                
        except Exception as e:
            print_status(f"⚠️ Processing error {audio_file}: {e}", "⚠️")
            continue
    
    # Statistics
    print_status(f"✅ Processed {total_files} files", "✅")
    print_status(f"🎤 Speakers: {', '.join(sorted(processed_speakers))}", "🎤")
    print_status(f"⏱️ Total duration: {total_duration/60:.1f} minutes", "⏱️")
    
    # Create summary
    summary = {
        "total_files": total_files,
        "total_duration_minutes": round(total_duration/60, 1),
        "speakers": sorted(list(processed_speakers)),
        "processing_options": {
            "normalized": normalize,
            "split_long": split_long,
            "auto_transcribed": auto_transcribe,
            "whisper_model": whisper_model,
            "language": language,
            "silence_detection": use_silence_detection,
            "min_silence_len": min_silence_len,
            "silence_thresh": silence_thresh,
            "min_segment_len": min_segment_len
        }
    }
    
    summary_file = output_path / "dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print_status(f"📊 Summary saved: {summary_file}", "📊")
    return total_files > 0

def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation for Fish Speech Fine-tuning")
    
    # Main parameters
    parser.add_argument("--input", required=True, help="Input directory or YouTube URL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--speaker", default="SPEAKER_1", help="Speaker name (for YouTube)")
    
    # Processing options
    parser.add_argument("--normalize", action="store_true", default=True, 
                       help="Normalize audio volume")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                       help="Do not normalize audio")
    parser.add_argument("--split-long", action="store_true", default=True,
                       help="Split long audio (>30 sec)")
    parser.add_argument("--no-split", action="store_false", dest="split_long",
                       help="Do not split long audio")
    parser.add_argument("--auto-transcribe", action="store_true",
                       help="Automatic transcription with Whisper")
    
    # Silence detection options
    parser.add_argument("--use-silence-detection", action="store_true",
                       help="Use silence detection for smarter audio segmentation")
    parser.add_argument("--min-silence-len", type=int, default=500,
                       help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--silence-thresh", type=int, default=-32,
                       help="Silence threshold in dB (default: -32)")
    parser.add_argument("--min-segment-len", type=float, default=1.5,
                       help="Minimum segment length in seconds (default: 1.5)")
    
    # Segmentation parameters
    parser.add_argument("--segment-duration", type=float, default=10.0,
                       help="Segment duration in seconds (default: 10)")
    parser.add_argument("--max-duration", type=float, default=None,
                       help="Maximum duration for processing in minutes (e.g., 20 for first 20 minutes)")
    
    # Whisper parameters
    parser.add_argument("--whisper-model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--language", default="auto",
                       help="Language for transcription (auto for auto-detection)")
    
    # YouTube options
    parser.add_argument("--youtube", action="store_true",
                       help="Input parameter is YouTube URL")
    
    args = parser.parse_args()
    
    print("📁 Dataset Preparation for Fish Speech")
    print("="*50)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.youtube:
        # Processing YouTube with automatic segmentation
        if args.use_silence_detection:
            print_status(f"Using silence-based segmentation", "🔇")
        else:
            print_status(f"Using time-based segmentation into {args.segment_duration}s segments", "✂️")
            
        processed_files = process_youtube_audio(
            args.input, 
            output_path, 
            args.speaker,
            segment_duration=args.segment_duration,
            max_duration_minutes=args.max_duration,
            use_silence_detection=args.use_silence_detection,
            min_silence_len=args.min_silence_len,
            silence_thresh=args.silence_thresh,
            min_segment_len=args.min_segment_len
        )
        
        if processed_files and args.auto_transcribe:
            print_status("YouTube segments transcription", "🎤")
            
            # Counter for progress
            total_files = len(processed_files)
            
            for i, audio_file in enumerate(processed_files, 1):
                print_status(f"Transcription {i}/{total_files}: {audio_file.name}", "🎤")
                
                text_content = transcribe_audio_whisper(
                    audio_file, 
                    args.whisper_model, 
                    args.language
                )
                
                if text_content:
                    # Create .lab file for each segment
                    lab_file = audio_file.with_suffix('.lab')
                    with open(lab_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    print_status(f"📝 Saved text: {lab_file.name}", "📝")
                else:
                    # Create placeholder if transcription failed
                    lab_file = audio_file.with_suffix('.lab')
                    with open(lab_file, 'w', encoding='utf-8') as f:
                        f.write(f"Segment from {args.input}")
                    print_status(f"⚠️ Created placeholder: {lab_file.name}", "⚠️")
            
            # Create summary for YouTube dataset
            summary = {
                "total_files": len(processed_files),
                "total_duration_minutes": round(len(processed_files) * args.segment_duration / 60, 1),
                "speakers": [args.speaker],
                "segment_duration": args.segment_duration,
                "max_duration_limit": args.max_duration,
                "source": "youtube",
                "url": args.input,
                "processing_options": {
                    "normalized": True,
                    "segmented": True,
                    "auto_transcribed": args.auto_transcribe,
                    "duration_limited": args.max_duration is not None
                }
            }
            
            summary_file = output_path / "dataset_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print_status(f"📊 Summary saved: {summary_file}", "📊")
            
    else:
        # Processing local directory
        success = process_directory(
            args.input,
            args.output,
            normalize=args.normalize,
            split_long=args.split_long,
            auto_transcribe=args.auto_transcribe,
            whisper_model=args.whisper_model,
            language=args.language,
            use_silence_detection=args.use_silence_detection,
            min_silence_len=args.min_silence_len,
            silence_thresh=args.silence_thresh,
            min_segment_len=args.min_segment_len
        )
        
        if not success:
            print_status("❌ Processing directory error", "❌")
            return 1
    
    print_status("🎉 Dataset preparation completed!", "🎉")
    print_status(f"📁 Result in: {output_path}", "📁")
    print_status("💡 Now you can run fine-tuning with finetune_tts.py", "💡")
    
    return 0

if __name__ == "__main__":
    exit(main()) 