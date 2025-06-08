#!/usr/bin/env python3
"""Audio Worker - Long-running process for audio generation"""

import json
import sys
import time
import gc
import psutil
import torch
from pathlib import Path
import signal
import traceback
import os
import re

# Import functions from cli_tts.py
from cli_tts import (
    setup_fish_speech, synthesize_speech_cli, apply_prosody_effects,
    setup_cache_dir, clear_semantic_cache
)

class ProgressFilter:
    """Filter to redirect progress messages to stdout instead of stderr"""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.progress_patterns = [
            re.compile(r'Fetching \d+ files:\s+\d+%'),
            re.compile(r'\d+%\|[█▏▎▍▌▋▊▉ ]*\|\s*\d+/\d+'),
            re.compile(r'\d+/\d+ \[\d+:\d+<\d+:\d+'),
            re.compile(r'\[\d+:\d+<\?\?\?\?'),
            re.compile(r'it/s\]'),
            re.compile(r'files:\s+\d+%'),
            re.compile(r'Downloading'),
            re.compile(r'Git LFS'),
        ]
    
    def write(self, text):
        # Check if text contains progress information
        is_progress = any(pattern.search(text) for pattern in self.progress_patterns)
        
        if is_progress:
            # Redirect progress to stdout (but don't print to avoid JSON parsing issues)
            pass  # Silently filter out progress messages
        else:
            # Keep actual errors in stderr
            if text.strip() and any(keyword in text.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

class AudioWorker:
    def __init__(self, worker_id: int, device: str = "mps"):
        self.worker_id = worker_id
        self.device = device
        self.fish_speech_dir = None
        self.checkpoints_dir = None
        self.is_running = True
        self.model_loaded = False
        
        # Memory monitoring
        self.process = psutil.Process()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)
        
        # Install progress filter for stderr
        self.original_stderr = sys.stderr
        sys.stderr = ProgressFilter(self.original_stderr)
        
        print(f"🤖 Audio Worker {worker_id} initialized on device: {device}")
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"🛑 Worker {self.worker_id} received shutdown signal")
        self.is_running = False
    
    def _monitor_memory(self):
        """Monitor current memory usage"""
        ram_mb = self.process.memory_info().rss / 1024 / 1024
        mps_mb = 0
        if torch.backends.mps.is_available():
            mps_mb = torch.mps.current_allocated_memory() / 1024 / 1024
        return ram_mb, mps_mb
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_model(self, model_version: str, model_path: str = None):
        """Load the TTS model once"""
        try:
            if self.model_loaded and self.checkpoints_dir:
                print(f"📦 Worker {self.worker_id}: Model already loaded")
                return True
            
            print(f"🔄 Worker {self.worker_id}: Loading model {model_version}...")
            
            # Apply Flash Attention optimizations if available
            try:
                from enhanced_flash_attention import apply_flash_attention_enhancements
                apply_flash_attention_enhancements()
                print(f"⚡ Worker {self.worker_id}: Flash Attention enabled")
            except ImportError:
                print(f"⚠️  Worker {self.worker_id}: Flash Attention not available")
            except Exception as e:
                print(f"⚠️  Worker {self.worker_id}: Flash Attention error: {e}")
            
            # Load model using setup_fish_speech
            initial_ram, initial_mps = self._monitor_memory()
            
            # Temporarily suppress progress bars during model loading
            print(f"📥 Worker {self.worker_id}: Downloading model (progress hidden for clean logs)...")
            
            result = setup_fish_speech(model_version, model_path)
            if result:
                self.fish_speech_dir, self.checkpoints_dir = result
                self.model_loaded = True
                
                final_ram, final_mps = self._monitor_memory()
                print(f"📊 Worker {self.worker_id}: Model loaded. RAM: {final_ram - initial_ram:.1f}MB, MPS: {final_mps - initial_mps:.1f}MB")
                
                # Setup cache directory
                setup_cache_dir()
                print(f"💾 Worker {self.worker_id}: Cache directory ready")
                
                return True
            else:
                print(f"❌ Worker {self.worker_id}: Failed to setup Fish Speech")
                return False
            
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def generate_audio(self, request_data: dict) -> dict:
        """Generate audio for a single request"""
        try:
            text = request_data.get("text", "")
            output_path = request_data.get("output_path", "")
            voice_settings = request_data.get("voice_settings", {})
            request_id = request_data.get("id", 0)
            use_semantic_cache = request_data.get("use_semantic_cache", True)
            
            if not text or not output_path:
                return {
                    "success": False,
                    "error": "Missing text or output_path",
                    "id": request_id
                }
            
            if not self.model_loaded:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "id": request_id
                }
            
            print(f"🎵 Worker {self.worker_id}: Generating audio for request {request_id}")
            if not use_semantic_cache:
                print(f"🚫 Worker {self.worker_id}: Semantic token cache disabled for request {request_id}")
            
            start_time = time.time()
            initial_ram, initial_mps = self._monitor_memory()
            
            # Create output directory
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine which checkpoints directory to use
            # Custom checkpointPath from voice settings overrides default
            checkpoints_dir = self.checkpoints_dir  # Default
            custom_checkpoint_path = voice_settings.get("checkpointPath")
            
            if custom_checkpoint_path:
                # Convert relative path to absolute
                custom_path = Path(custom_checkpoint_path)
                if not custom_path.is_absolute():
                    # Relative to current working directory
                    custom_path = Path.cwd() / custom_path
                
                if custom_path.exists():
                    checkpoints_dir = custom_path
                    print(f"🎛️ Worker {self.worker_id}: Using custom checkpoint: {custom_checkpoint_path}")
                else:
                    print(f"⚠️  Worker {self.worker_id}: Custom checkpoint not found: {custom_path}, using default")
            
            # Prepare voice reference if specified
            prompt_tokens = None
            prompt_text = None
            
            if voice_settings.get("voice"):
                voice_name = voice_settings["voice"]
                voice_file = Path(f"voices/{voice_name}.npy")
                voice_text_file = Path(f"voices/{voice_name}.txt")
                
                if voice_file.exists():
                    prompt_tokens = str(voice_file)
                    if voice_text_file.exists():
                        with open(voice_text_file, 'r', encoding='utf-8') as f:
                            prompt_text = f.read().strip()
                    print(f"🎭 Worker {self.worker_id}: Using voice reference: {voice_name}")
                else:
                    print(f"⚠️  Worker {self.worker_id}: Voice file not found: {voice_file}")
            
            # Generate audio using synthesize_speech_cli
            success = synthesize_speech_cli(
                text=text,
                output_path=output_path,
                device=self.device,
                prompt_tokens=prompt_tokens,
                prompt_text=prompt_text,
                checkpoints_dir=checkpoints_dir,
                use_cache=use_semantic_cache,
                speed=voice_settings.get("speed", 1.0),
                volume=voice_settings.get("volume", 0),
                pitch=voice_settings.get("pitch", 1.0),
                emotion=voice_settings.get("emotion"),
                intensity=voice_settings.get("intensity", 0.5),
                style=voice_settings.get("style")
            )
            
            if success and Path(output_path).exists():
                # Apply prosody effects (speed, volume, pitch)
                prosody_success = apply_prosody_effects(
                    output_path,
                    speed=voice_settings.get("speed", 1.0),
                    volume_db=voice_settings.get("volume", 0),
                    pitch=voice_settings.get("pitch", 1.0)
                )
                
                if not prosody_success:
                    print(f"⚠️  Worker {self.worker_id}: Prosody effects failed for request {request_id}")
                
                file_size = Path(output_path).stat().st_size / 1024
                generation_time = time.time() - start_time
                final_ram, final_mps = self._monitor_memory()
                
                cache_status = "cached" if use_semantic_cache else "no-cache"
                checkpoint_info = f"custom:{custom_checkpoint_path}" if custom_checkpoint_path else "default"
                print(f"✅ Worker {self.worker_id}: Generated {request_id} in {generation_time:.2f}s ({file_size:.1f}KB) [{cache_status}, {checkpoint_info}]")
                
                # Optimize memory after generation
                self._optimize_memory()
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "generation_time": generation_time,
                    "file_size_kb": file_size,
                    "memory_usage": {
                        "ram_mb": final_ram - initial_ram,
                        "mps_mb": final_mps - initial_mps
                    },
                    "use_semantic_cache": use_semantic_cache,
                    "checkpoint_path": str(checkpoints_dir),
                    "id": request_id
                }
            else:
                return {
                    "success": False,
                    "error": "Audio generation failed" if not success else "Audio file was not created",
                    "id": request_id
                }
                
        except Exception as e:
            print(f"❌ Worker {self.worker_id}: Generation failed for request {request_id}: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "id": request_id
            }
    
    def run(self):
        """Main worker loop - reads from stdin and processes requests"""
        print(f"🚀 Worker {self.worker_id}: Started and waiting for requests...")
        
        try:
            while self.is_running:
                try:
                    # Read request from stdin
                    line = sys.stdin.readline()
                    if not line:
                        # EOF - parent process closed
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse JSON request
                    request = json.loads(line)
                    
                    # Handle different request types
                    if request.get("type") == "load_model":
                        success = self.load_model(
                            request.get("model_version", "1.5"),
                            request.get("model_path")
                        )
                        response = {
                            "type": "model_loaded",
                            "success": success,
                            "worker_id": self.worker_id
                        }
                        print(json.dumps(response), flush=True)
                        
                    elif request.get("type") == "generate":
                        result = self.generate_audio(request)
                        result["type"] = "generation_result"
                        result["worker_id"] = self.worker_id
                        print(json.dumps(result), flush=True)
                        
                    elif request.get("type") == "shutdown":
                        print(f"🛑 Worker {self.worker_id}: Shutdown requested")
                        break
                        
                    elif request.get("type") == "ping":
                        # Health check
                        ram, mps = self._monitor_memory()
                        response = {
                            "type": "pong",
                            "worker_id": self.worker_id,
                            "memory": {"ram_mb": ram, "mps_mb": mps},
                            "model_loaded": self.model_loaded
                        }
                        print(json.dumps(response), flush=True)
                        
                    elif request.get("type") == "clear_cache":
                        # Clear semantic cache
                        try:
                            clear_semantic_cache()
                            response = {
                                "type": "cache_cleared",
                                "success": True,
                                "worker_id": self.worker_id
                            }
                        except Exception as e:
                            response = {
                                "type": "cache_cleared",
                                "success": False,
                                "error": str(e),
                                "worker_id": self.worker_id
                            }
                        print(json.dumps(response), flush=True)
                        
                except json.JSONDecodeError as e:
                    error_response = {
                        "type": "error",
                        "worker_id": self.worker_id,
                        "error": f"Invalid JSON: {e}"
                    }
                    print(json.dumps(error_response), flush=True)
                    
                except Exception as e:
                    error_response = {
                        "type": "error",
                        "worker_id": self.worker_id,
                        "error": str(e)
                    }
                    print(json.dumps(error_response), flush=True)
                    
        except KeyboardInterrupt:
            print(f"🛑 Worker {self.worker_id}: Interrupted")
        
        finally:
            print(f"👋 Worker {self.worker_id}: Shutting down")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python audio_worker.py <worker_id> [device]")
        sys.exit(1)
    
    worker_id = int(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "mps"
    
    # Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        print(f"⚠️  MPS not available, falling back to CPU")
        device = "cpu"
    
    worker = AudioWorker(worker_id, device)
    worker.run()


if __name__ == "__main__":
    main() 