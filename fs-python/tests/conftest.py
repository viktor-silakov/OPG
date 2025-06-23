#!/usr/bin/env python3
"""
Pytest configuration for Fish Speech E2E tests
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "initial_steps": 5,
        "resume_steps": 8,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "save_every_n_steps": 3
    }

@pytest.fixture(scope="session") 
def test_data_dir():
    """Test data directory fixture"""
    return Path(__file__).parent / "data"

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow) 