"""
Pytest configuration and shared fixtures for eco2ai tests.
"""
import pytest
import os
import glob


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test emission files after each test."""
    yield
    # Cleanup any emission files created during tests
    for pattern in ['*emission*.csv', '*emission*.json', 'encoded_*.csv']:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except Exception:
                pass


@pytest.fixture
def test_file_name():
    """Provide a unique test file name for each test."""
    import uuid
    return f"test_emission_{uuid.uuid4().hex[:8]}.csv"
