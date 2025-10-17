import pytest
import time
import os
from eco2ai import Tracker, track


class TestTracker:
    """Test suite for the Tracker class"""

    def test_tracker_initialization(self):
        """Test that Tracker initializes with default parameters"""
        tracker = Tracker(
            project_name="test_project",
            file_name="emission.csv",  # Explicitly set to avoid config interference
            ignore_warnings=True
        )
        assert tracker.project_name == "test_project"
        assert tracker.file_name == "emission.csv"
        assert tracker._measure_period == 10
        assert tracker._pue == 1

    def test_tracker_custom_parameters(self):
        """Test Tracker initialization with custom parameters"""
        tracker = Tracker(
            project_name="custom_project",
            experiment_description="Test experiment",
            file_name="test_emission.csv",
            measure_period=5,
            pue=1.5,
            ignore_warnings=True
        )
        assert tracker.project_name == "custom_project"
        assert tracker.experiment_description == "Test experiment"
        assert tracker.file_name == "test_emission.csv"
        assert tracker._measure_period == 5
        assert tracker._pue == 1.5

    def test_tracker_start_stop(self, test_file_name):
        """Test basic start/stop functionality"""
        tracker = Tracker(
            project_name="test_project",
            file_name=test_file_name,
            ignore_warnings=True
        )
        tracker.start()
        time.sleep(2)
        tracker.stop()

        # Check that the output file was created
        assert os.path.exists(test_file_name)

    def test_tracker_consumption(self, test_file_name):
        """Test that consumption is calculated"""
        tracker = Tracker(
            project_name="test_project",
            file_name=test_file_name,
            ignore_warnings=True
        )
        tracker.start()
        time.sleep(2)
        tracker.stop()

        consumption = tracker.consumption()
        assert consumption >= 0

    def test_tracker_id_generation(self, test_file_name):
        """Test that each tracker gets a unique ID"""
        tracker = Tracker(
            project_name="test_project",
            file_name=test_file_name,
            ignore_warnings=True
        )
        tracker.start()
        tracker_id = tracker.id()
        tracker.stop()

        assert tracker_id is not None
        assert isinstance(tracker_id, str)

    def test_invalid_measure_period(self):
        """Test that invalid measure_period raises ValueError"""
        with pytest.raises(ValueError):
            Tracker(project_name="test", measure_period=-1, ignore_warnings=True)

        with pytest.raises(ValueError):
            Tracker(project_name="test", measure_period=0, ignore_warnings=True)

    @pytest.mark.skip(reason="Decorator test can interfere with other tests")
    def test_decorator_tracking(self):
        """Test the @track decorator"""
        @track
        def simple_computation():
            time.sleep(1)
            return 42

        result = simple_computation()
        assert result == 42


class TestTrackerTraining:
    """Test suite for training-specific Tracker functionality"""

    def test_start_training(self, test_file_name):
        """Test start_training method"""
        tracker = Tracker(
            project_name="training_test",
            file_name=test_file_name,
            ignore_warnings=True
        )
        tracker.start_training(start_epoch=1)
        assert tracker._mode == "training"
        assert tracker._current_epoch == 1

        tracker.stop_training()

    def test_new_epoch(self, test_file_name):
        """Test new_epoch method"""
        tracker = Tracker(
            project_name="training_test",
            file_name=test_file_name,
            ignore_warnings=True
        )
        tracker.start_training(start_epoch=1)
        time.sleep(1)

        tracker.new_epoch({'loss': 0.5, 'accuracy': 0.85})
        assert tracker._current_epoch == 2

        tracker.stop_training()

    def test_invalid_start_epoch(self):
        """Test that non-integer start_epoch raises TypeError"""
        tracker = Tracker(project_name="test", ignore_warnings=True)
        with pytest.raises(TypeError):
            tracker.start_training(start_epoch="1")
