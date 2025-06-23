#!/usr/bin/env python3
"""
E2E Test for Fish Speech Fine-tuning with Emotional Tokens
Tests the complete workflow: training from scratch + resume from checkpoint
"""

import os
import sys
import pytest
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
import json
import time

# Add parent directory to path to import finetune_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_tts import (
    check_requirements,
    prepare_dataset_structure,
    start_finetuning
)

class TestConfig:
    """Test configuration constants"""
    # Test projects
    INITIAL_PROJECT = "e2e_test_initial"
    RESUME_PROJECT = "e2e_test_resume"
    
    # Training parameters
    INITIAL_STEPS = 5
    RESUME_STEPS = 8
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    SAVE_EVERY_N_STEPS = 3
    
    # Real Russian voice data from main project
    VOICES_DIR = Path(__file__).parent.parent / "voices"
    
    # Selected Russian voices for testing
    RUSSIAN_VOICES = [
        "RU_Google_Female_Zephyr",
        "RU_Google_Male_Achird",
        "RU_Google_Female_Zephyr",
    ]
    
    # Test data preparation
    TEST_DATA_DIR = Path(__file__).parent / "data" 
    REAL_DATA_DIR = TEST_DATA_DIR / "real_russian"
    
    # Inference testing
    INFERENCE_TEXTS = [
        "Привет! Это тест обученной модели.",
        "Неожиданно, но это работает!"
        "Жаль никто не пришел!"
    ]
    INFERENCE_OUTPUT_DIR = TEST_DATA_DIR / "inference_outputs"

class E2ETestRunner:
    """Main E2E test runner"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.checkpoints_dir = Path("/Users/a1/Project/OPG/checkpoints")
        self.initial_checkpoint_path: Optional[Path] = None
        self.resume_checkpoint_path: Optional[Path] = None
        self.inference_results = []
        
    def setup_test_data(self):
        """Setup real Russian voice data for training"""
        print("🔧 Setting up real Russian voice data...")
        
        # Ensure test data directory exists
        TestConfig.REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        voices_copied = 0
        for voice_name in TestConfig.RUSSIAN_VOICES:
            voice_dir = TestConfig.REAL_DATA_DIR / voice_name
            voice_dir.mkdir(exist_ok=True)
            
            # Copy .npy file (semantic tokens)
            npy_source = TestConfig.VOICES_DIR / f"{voice_name}.npy"
            npy_target = voice_dir / f"{voice_name}.npy"
            
            # Copy .txt file (transcriptions)
            txt_source = TestConfig.VOICES_DIR / f"{voice_name}.txt"
            txt_target = voice_dir / f"{voice_name}.lab"  # Fish Speech expects .lab files
            
            if npy_source.exists() and txt_source.exists():
                import shutil
                
                # Copy semantic tokens
                shutil.copy2(npy_source, npy_target)
                
                # Copy and enhance text with emotional tokens
                with open(txt_source, "r", encoding="utf-8") as f:
                    original_text = f.read().strip()
                
                # Add emotional tokens to make text more interesting for testing
                enhanced_text = self._add_emotional_tokens(original_text, voice_name)
                
                with open(txt_target, "w", encoding="utf-8") as f:
                    f.write(enhanced_text)
                
                voices_copied += 1
                print(f"✅ Copied voice: {voice_name} (npy: {npy_source.stat().st_size//1024}KB)")
            else:
                print(f"⚠️ Missing files for voice: {voice_name}")
        
        print(f"✅ Real Russian voice data prepared: {voices_copied}/{len(TestConfig.RUSSIAN_VOICES)} voices")
        
        # Verify we have enough data for training
        if voices_copied == 0:
            raise RuntimeError("No voice data was copied! Check VOICES_DIR path.")
    
    def _add_emotional_tokens(self, text: str, voice_name: str) -> str:
        """Add emotional tokens to text based on voice character"""
        
        # Simple approach: just add one emotion token at the beginning
        voice_emotions = {
            "RU_Google_Female_Zephyr": "(joyful)",
            "RU_Male_Goblin_Puchkov": "(angry)",
            "RU_Google_Male_Achird": "(joyful)"
        }
        
        emotion = voice_emotions.get(voice_name, "(joyful)")
        return f"{emotion} {text}"
        
    def run_system_checks(self):
        """Run system requirement checks"""
        print("🔍 Running system checks...")
        
        result = check_requirements()
        assert result, "System requirements check failed"
        
        print("✅ System checks passed")
        
    def run_initial_training(self):
        """Run initial training from scratch"""
        print(f"🚀 Starting initial training: {TestConfig.INITIAL_PROJECT}")
        
        # Prepare dataset with semantic tokens (skip audio preparation)
        target_dir = Path(TestConfig.TEST_DATA_DIR / "prepared")
        print(f"📁 Using prepared semantic tokens directly: {target_dir}")
        
        # Copy semantic tokens directly to prepared directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        prepared_files = 0
        for voice_name in TestConfig.RUSSIAN_VOICES:
            voice_source_dir = TestConfig.REAL_DATA_DIR / voice_name
            voice_target_dir = target_dir / voice_name
            voice_target_dir.mkdir(exist_ok=True)
            
            # Copy .npy and .lab files to prepared directory
            npy_source = voice_source_dir / f"{voice_name}.npy"
            lab_source = voice_source_dir / f"{voice_name}.lab"
            
            if npy_source.exists() and lab_source.exists():
                import shutil
                shutil.copy2(npy_source, voice_target_dir / f"{voice_name}.npy")
                shutil.copy2(lab_source, voice_target_dir / f"{voice_name}.lab")
                prepared_files += 1
                print(f"✅ Prepared voice: {voice_name}")
        
        print(f"✅ Prepared {prepared_files}/{len(TestConfig.RUSSIAN_VOICES)} voices with semantic tokens")
        assert prepared_files > 0, "No voice data was prepared for training"
        
        # Run training
        training_result = start_finetuning(
            project_name=TestConfig.INITIAL_PROJECT,
            checkpoints_dir=str(TestConfig.TEST_DATA_DIR / "prepared"),
            max_steps=TestConfig.INITIAL_STEPS,
            batch_size=TestConfig.BATCH_SIZE,
            learning_rate=TestConfig.LEARNING_RATE,
            save_every_n_steps=TestConfig.SAVE_EVERY_N_STEPS,
            resume_from_checkpoint=None
        )
        
        assert training_result, "Initial training failed"
        
        # Find created checkpoint
        checkpoint_dir = self.checkpoints_dir / TestConfig.INITIAL_PROJECT / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        assert len(checkpoints) > 0, "No checkpoints were created"
        
        # Use the latest checkpoint
        self.initial_checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        print(f"✅ Initial training completed, checkpoint: {self.initial_checkpoint_path.name}")
        
    def run_resume_training(self):
        """Run resume training from checkpoint"""
        print(f"🔄 Starting resume training: {TestConfig.RESUME_PROJECT}")
        
        assert self.initial_checkpoint_path, "No initial checkpoint available"
        
        # Run resume training
        training_result = start_finetuning(
            project_name=TestConfig.RESUME_PROJECT,
            checkpoints_dir=str(TestConfig.TEST_DATA_DIR / "prepared"),
            max_steps=TestConfig.RESUME_STEPS,
            batch_size=TestConfig.BATCH_SIZE,
            learning_rate=TestConfig.LEARNING_RATE,
            save_every_n_steps=TestConfig.SAVE_EVERY_N_STEPS,
            resume_from_checkpoint=str(self.initial_checkpoint_path)
        )
        
        assert training_result, "Resume training failed"
        
        # Find created checkpoint
        checkpoint_dir = self.checkpoints_dir / TestConfig.RESUME_PROJECT / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        assert len(checkpoints) > 0, "No checkpoints were created in resume training"
        
        # Use the latest checkpoint
        self.resume_checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        print(f"✅ Resume training completed, checkpoint: {self.resume_checkpoint_path.name}")
        
    def verify_checkpoints(self):
        """Verify that checkpoints were created and contain expected data"""
        print("🔍 Verifying checkpoints...")
        
        checkpoints_to_verify = [
            (self.initial_checkpoint_path, "Initial"),
            (self.resume_checkpoint_path, "Resume")
        ]
        
        for checkpoint_path, name in checkpoints_to_verify:
            if checkpoint_path and checkpoint_path.exists():
                # Check file size (should be reasonable)
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                assert size_mb > 1, f"{name} checkpoint too small: {size_mb:.1f}MB"
                assert size_mb < 500, f"{name} checkpoint too large: {size_mb:.1f}MB"
                
                print(f"✅ {name} checkpoint verified: {size_mb:.1f}MB")
            else:
                raise AssertionError(f"{name} checkpoint not found or doesn't exist")
        
    def verify_emotional_tokens(self):
        """Verify that emotional tokens are properly handled"""
        print("🎭 Verifying emotional tokens...")
        
        # Import tokenizer to check tokens
        sys.path.insert(0, str(Path("fish-speech").resolve()))
        from fish_speech.tokenizer import FishTokenizer, EMOTION_TOKENS
        
        # Check that emotional tokens are defined
        expected_emotions = ["(joyful)", "(sad)", "(angry)", "(scared)", "(surprised)"]
        
        for emotion in expected_emotions:
            assert emotion in EMOTION_TOKENS, f"Emotion token {emotion} not found in EMOTION_TOKENS"
        
        print(f"✅ All {len(expected_emotions)} emotional tokens verified")
    
    def run_inference_tests(self):
        """Run inference tests with trained checkpoints"""
        print("🎤 Running inference tests with trained models...")
        
        # Create output directory
        TestConfig.INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Test both initial and resume checkpoints
        checkpoints_to_test = [
            (self.initial_checkpoint_path, "initial", TestConfig.INITIAL_PROJECT),
            (self.resume_checkpoint_path, "resume", TestConfig.RESUME_PROJECT)
        ]
        
        for checkpoint_path, checkpoint_type, project_name in checkpoints_to_test:
            if not checkpoint_path or not checkpoint_path.exists():
                print(f"⚠️ Skipping {checkpoint_type} checkpoint - not found")
                continue
                
            print(f"🧪 Testing {checkpoint_type} checkpoint: {checkpoint_path.name}")
            
            for i, test_text in enumerate(TestConfig.INFERENCE_TEXTS):
                # Select voice for this test
                voice_name = TestConfig.RUSSIAN_VOICES[i % len(TestConfig.RUSSIAN_VOICES)]
                
                # Setup output file
                output_file = TestConfig.INFERENCE_OUTPUT_DIR / f"{checkpoint_type}_test_{i+1}_{voice_name}.wav"
                
                # Run inference
                result = self._run_single_inference(
                    text=test_text,
                    checkpoint_path=checkpoint_path,
                    voice_name=voice_name,
                    output_file=output_file,
                    play_audio=True
                )
                
                # Store result for reporting
                result['checkpoint_type'] = checkpoint_type
                result['project_name'] = project_name
                result['test_text'] = test_text
                result['voice_name'] = voice_name
                self.inference_results.append(result)
                
                if result['success']:
                    print(f"✅ Inference {i+1}/{len(TestConfig.INFERENCE_TEXTS)} passed ({result['duration']:.1f}s)")
                else:
                    print(f"❌ Inference {i+1}/{len(TestConfig.INFERENCE_TEXTS)} failed: {result['error']}")
        
        # Verify results
        total_tests = len(checkpoints_to_test) * len(TestConfig.INFERENCE_TEXTS)
        passed_tests = sum(1 for r in self.inference_results if r['success'])
        
        print(f"🎤 Inference testing completed: {passed_tests}/{total_tests} tests passed")
        
        # At least 50% should pass for a successful test
        assert passed_tests >= total_tests * 0.5, f"Too many inference tests failed: {passed_tests}/{total_tests}"
    
    def _run_single_inference(self, text: str, checkpoint_path: Path, voice_name: str, 
                             output_file: Path, play_audio: bool = False) -> dict:
        """Run a single inference test"""
        import subprocess
        import time
        
        start_time = time.time()
        
        try:
            # Build command for flash_optimized_cli.py
            cmd = [
                sys.executable, "flash_optimized_cli.py",
                text,
                "-o", str(output_file),
                "--device", "mps",
                "--checkpoint", str(checkpoint_path),
                "--voice", voice_name,
                "--emotion", "neutral",  # Let text tokens control emotion
                "--monitor"  # Enable memory monitoring
            ]
            
            if play_audio:
                cmd.append("--play")
            
            print(f"🔊 Running inference: {' '.join(cmd[-6:])}")  # Show last few args
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=Path(__file__).parent.parent  # fs-python directory
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Check output file
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    
                    # Basic audio file validation
                    if file_size > 1000:  # At least 1KB
                        return {
                            'success': True,
                            'duration': duration,
                            'file_size': file_size,
                            'output_file': str(output_file),
                            'error': None
                        }
                    else:
                        return {
                            'success': False,
                            'duration': duration,
                            'file_size': file_size,
                            'output_file': str(output_file),
                            'error': f"Output file too small: {file_size} bytes"
                        }
                else:
                    return {
                        'success': False,
                        'duration': duration,
                        'file_size': 0,
                        'output_file': str(output_file),
                        'error': "Output file was not created"
                    }
            else:
                return {
                    'success': False,
                    'duration': duration,
                    'file_size': 0,
                    'output_file': str(output_file),
                    'error': f"Command failed (exit {result.returncode}): {result.stderr.strip()}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'file_size': 0,
                'output_file': str(output_file),
                'error': "Inference timeout (>120s)"
            }
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'file_size': 0,
                'output_file': str(output_file),
                'error': f"Unexpected error: {str(e)}"
            }
        
    def generate_test_report(self):
        """Generate a test report with results"""
        print("📊 Generating test report...")
        
        # Calculate inference statistics
        total_inference_tests = len(self.inference_results)
        passed_inference_tests = sum(1 for r in self.inference_results if r['success'])
        avg_inference_time = sum(r['duration'] for r in self.inference_results) / total_inference_tests if total_inference_tests > 0 else 0
        total_audio_files = sum(1 for r in self.inference_results if r['success'] and r['file_size'] > 0)
        
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": {
                "initial_project": TestConfig.INITIAL_PROJECT,
                "resume_project": TestConfig.RESUME_PROJECT,
                "initial_steps": TestConfig.INITIAL_STEPS,
                "resume_steps": TestConfig.RESUME_STEPS,
                "batch_size": TestConfig.BATCH_SIZE,
                "learning_rate": TestConfig.LEARNING_RATE,
                "voices_tested": TestConfig.RUSSIAN_VOICES,
                "inference_texts": TestConfig.INFERENCE_TEXTS
            },
            "training_results": {
                "initial_checkpoint": str(self.initial_checkpoint_path) if self.initial_checkpoint_path else None,
                "resume_checkpoint": str(self.resume_checkpoint_path) if self.resume_checkpoint_path else None,
                "initial_checkpoint_size_mb": round(self.initial_checkpoint_path.stat().st_size / (1024 * 1024), 2) if self.initial_checkpoint_path else None,
                "resume_checkpoint_size_mb": round(self.resume_checkpoint_path.stat().st_size / (1024 * 1024), 2) if self.resume_checkpoint_path else None
            },
            "inference_results": {
                "total_tests": total_inference_tests,
                "passed_tests": passed_inference_tests,
                "success_rate": round(passed_inference_tests / total_inference_tests * 100, 1) if total_inference_tests > 0 else 0,
                "average_inference_time_sec": round(avg_inference_time, 2),
                "audio_files_generated": total_audio_files,
                "detailed_results": self.inference_results
            },
            "status": "PASSED"
        }
        
        report_file = self.test_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Test report saved: {report_file}")
        
    def run_full_test(self):
        """Run the complete E2E test"""
        print("🐟 Starting Fish Speech E2E Training Test")
        print("=" * 50)
        
        try:
            self.setup_test_data()
            self.run_system_checks()
            self.verify_emotional_tokens()
            self.run_initial_training()
            self.run_resume_training()
            self.verify_checkpoints()
            self.run_inference_tests()
            self.generate_test_report()
            
            # Calculate stats for final report
            inference_passed = sum(1 for r in self.inference_results if r['success'])
            inference_total = len(self.inference_results)
            
            print("=" * 50)
            print("🎉 E2E Test PASSED!")
            print(f"✅ Initial training: {TestConfig.INITIAL_STEPS} steps")
            print(f"✅ Resume training: {TestConfig.RESUME_STEPS} steps")
            print(f"✅ Checkpoints created and verified")
            print(f"✅ Emotional tokens working")
            print(f"✅ Inference tests: {inference_passed}/{inference_total} passed")
            if inference_passed > 0:
                avg_time = sum(r['duration'] for r in self.inference_results if r['success']) / inference_passed
                print(f"✅ Average inference time: {avg_time:.1f}s")
            return True
            
        except Exception as e:
            print("=" * 50)
            print(f"❌ E2E Test FAILED: {e}")
            print("Check logs above for details")
            return False

# Pytest integration
class TestE2ETraining:
    """Pytest test class for E2E training"""
    
    def test_full_e2e_workflow(self):
        """Test the complete E2E workflow"""
        runner = E2ETestRunner()
        result = runner.run_full_test()
        assert result, "E2E test failed"

def main():
    """Main entry point for running the test standalone"""
    runner = E2ETestRunner()
    success = runner.run_full_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 