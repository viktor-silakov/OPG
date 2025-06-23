#!/usr/bin/env python3
"""
Cleanup utility for Fish Speech E2E tests
Removes test checkpoints and prepared data
"""

import shutil
from pathlib import Path

def cleanup_test_data():
    """Clean up all test-related data"""
    print("🧹 Cleaning up E2E test data...")
    
    # Paths to clean
    test_dir = Path(__file__).parent
    checkpoints_dir = Path("/Users/a1/Project/OPG/checkpoints")
    
    cleanup_items = [
        # Test checkpoints
        checkpoints_dir / "e2e_test_initial",
        checkpoints_dir / "e2e_test_resume", 
        # Prepared data
        test_dir / "data" / "prepared",
        # Real Russian voice copies
        test_dir / "data" / "real_russian",
        # Inference outputs
        test_dir / "data" / "inference_outputs",
        # Test reports
        test_dir / "test_report.json",
        test_dir / "report.html",
        test_dir / "report.json",
        # Pytest cache
        test_dir / ".pytest_cache"
    ]
    
    removed_count = 0
    for item in cleanup_items:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
                print(f"🗂️  Removed directory: {item}")
            else:
                item.unlink()
                print(f"📄 Removed file: {item}")
            removed_count += 1
        else:
            print(f"⏭️  Skipped (not found): {item}")
    
    print(f"✅ Cleanup completed! Removed {removed_count} items.")

if __name__ == "__main__":
    cleanup_test_data() 