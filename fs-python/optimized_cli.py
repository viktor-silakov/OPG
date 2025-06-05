#!/usr/bin/env python3
"""Memory-optimized Fish Speech TTS CLI"""

import torch
import gc
import argparse
import sys
from pathlib import Path

def optimized_tts(text, output_path="output.wav", device="mps"):
    """Memory-optimized TTS with aggressive cleanup"""
    
    # Force cleanup before starting
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    try:
        # Import only when needed to reduce memory overhead
        sys.path.append(str(Path(__file__).parent / "fish-speech"))
        from fish_speech.models.text2semantic.inference import init_model, generate_long
        from huggingface_hub import snapshot_download
        
        print(f"🎯 Optimized synthesis: {text[:50]}...")
        
        # Get model with minimal caching
        checkpoints_dir = snapshot_download(
            repo_id="fishaudio/fish-speech-1.5",
            repo_type="model"
        )
        
        # Use half precision for MPS to reduce RAM usage
        precision = torch.half if device == "mps" else torch.bfloat16
        
        # Load model
        model, decode_one_token = init_model(
            checkpoint_path=checkpoints_dir,
            device=device,
            precision=precision,
            compile=False  # Disable compilation
        )
        
        # Smaller cache to reduce memory
        max_seq_len = min(2048, model.config.max_seq_len)
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=max_seq_len,
                dtype=precision,
            )
        
        # Generate with conservative settings
        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=1024,  # Limit tokens
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            compile=False,
            iterative_prompt=True,
            chunk_length=50,  # Smaller chunks
        )
        
        # Collect tokens
        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
        
        # Generate audio if we have tokens
        if codes:
            import numpy as np
            import subprocess
            import tempfile
            
            # Save tokens to temporary file
            codes_array = torch.cat(codes, dim=1).cpu().numpy()
            
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
                np.save(temp_file.name, codes_array)
                
                # Use VQGAN subprocess for audio generation
                vqgan_cmd = [
                    sys.executable, "-m", "fish_speech.models.vqgan.inference",
                    "--input-path", temp_file.name,
                    "--output-path", output_path,
                    "--config-name", "firefly_gan_vq",
                    "--checkpoint-path", str(Path(checkpoints_dir) / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
                    "--device", device
                ]
                
                result = subprocess.run(vqgan_cmd, capture_output=True, text=True, cwd="fish-speech")
                
                # Clean up temp file
                Path(temp_file.name).unlink()
                
                if result.returncode == 0:
                    print(f"✅ Audio saved: {output_path}")
                    return True
                else:
                    print(f"❌ VQGAN error: {result.stderr}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return False
        
    finally:
        # Aggressive cleanup
        if 'model' in locals():
            del model, decode_one_token
        if 'codes' in locals():
            del codes
        if 'codes_array' in locals():
            del codes_array
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Fish Speech TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output_optimized.wav", help="Output file")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Device")
    
    args = parser.parse_args()
    
    success = optimized_tts(args.text, args.output, args.device)
    
    if success and Path(args.output).exists():
        file_size = Path(args.output).stat().st_size
        print(f"🎵 Generated: {args.output} ({file_size} bytes)")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
