"""
Basic import tests to verify the package is installable and importable.
"""
import pytest


def test_import_eco2ai():
    """Test that eco2ai can be imported"""
    import eco2ai
    assert eco2ai is not None


def test_version_available():
    """Test that version is available"""
    import eco2ai
    assert hasattr(eco2ai, '__version__')
    assert isinstance(eco2ai.__version__, str)


def test_tracker_importable():
    """Test that Tracker class is importable"""
    from eco2ai import Tracker
    assert Tracker is not None


def test_track_decorator_importable():
    """Test that track decorator is importable"""
    from eco2ai import track
    assert track is not None


def test_utils_importable():
    """Test that utility functions are importable"""
    from eco2ai import set_params, get_params, summary, available_devices
    assert set_params is not None
    assert get_params is not None
    assert summary is not None
    assert available_devices is not None


def test_tools_importable():
    """Test that hardware tools are importable"""
    from eco2ai import CPU, GPU, RAM
    assert CPU is not None
    assert GPU is not None
    assert RAM is not None
