import pytest
import time
import os
from eco2ai import Tracker, track


class TestTracker:
    """Test suite for the Tracker class"""

    def test_tracker_initialization(self):
        """Test that Tracker initializes with default parameters"""
        tracker = Tracker(project_name="test_project")
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
            pue=1.5
        )
        assert tracker.project_name == "custom_project"
        assert tracker.experiment_description == "Test experiment"
        assert tracker.file_name == "test_emission.csv"
        assert tracker._measure_period == 5
        assert tracker._pue == 1.5

    def test_tracker_start_stop(self):
        """Test basic start/stop functionality"""
        tracker = Tracker(
            project_name="test_project",
            file_name="test_emission.csv"
        )
        tracker.start()
        time.sleep(2)
        tracker.stop()

        # Check that the output file was created
        assert os.path.exists("test_emission.csv")

        # Clean up
        if os.path.exists("test_emission.csv"):
            os.remove("test_emission.csv")
        if os.path.exists("test_emission.json"):
            os.remove("test_emission.json")

    def test_tracker_consumption(self):
        """Test that consumption is calculated"""
        tracker = Tracker(
            project_name="test_project",
            file_name="test_emission.csv"
        )
        tracker.start()
        time.sleep(2)
        tracker.stop()

        consumption = tracker.consumption()
        assert consumption >= 0

        # Clean up
        if os.path.exists("test_emission.csv"):
            os.remove("test_emission.csv")
        if os.path.exists("test_emission.json"):
            os.remove("test_emission.json")

    def test_tracker_id_generation(self):
        """Test that each tracker gets a unique ID"""
        tracker = Tracker(project_name="test_project")
        tracker.start()
        tracker_id = tracker.id()
        tracker.stop()

        assert tracker_id is not None
        assert isinstance(tracker_id, str)

        # Clean up
        if os.path.exists("emission.csv"):
            os.remove("emission.csv")
        if os.path.exists("emission.json"):
            os.remove("emission.json")

    def test_invalid_measure_period(self):
        """Test that invalid measure_period raises ValueError"""
        with pytest.raises(ValueError):
            Tracker(project_name="test", measure_period=-1)

        with pytest.raises(ValueError):
            Tracker(project_name="test", measure_period=0)

    def test_decorator_tracking(self):
        """Test the @track decorator"""
        @track
        def simple_computation():
            time.sleep(1)
            return 42

        result = simple_computation()
        assert result == 42

        # Check that the output file was created
        assert os.path.exists("emission.csv")

        # Clean up
        if os.path.exists("emission.csv"):
            os.remove("emission.csv")
        if os.path.exists("emission.json"):
            os.remove("emission.json")


class TestTrackerTraining:
    """Test suite for training-specific Tracker functionality"""

    def test_start_training(self):
        """Test start_training method"""
        tracker = Tracker(
            project_name="training_test",
            file_name="training_emission.csv"
        )
        tracker.start_training(start_epoch=1)
        assert tracker._mode == "training"
        assert tracker._current_epoch == 1

        tracker.stop_training()

        # Clean up
        if os.path.exists("training_emission.csv"):
            os.remove("training_emission.csv")
        if os.path.exists("training_emission.json"):
            os.remove("training_emission.json")

    def test_new_epoch(self):
        """Test new_epoch method"""
        tracker = Tracker(
            project_name="training_test",
            file_name="training_emission.csv"
        )
        tracker.start_training(start_epoch=1)
        time.sleep(1)

        tracker.new_epoch({'loss': 0.5, 'accuracy': 0.85})
        assert tracker._current_epoch == 2

        tracker.stop_training()

        # Clean up
        if os.path.exists("training_emission.csv"):
            os.remove("training_emission.csv")
        if os.path.exists("training_emission.json"):
            os.remove("training_emission.json")

    def test_invalid_start_epoch(self):
        """Test that non-integer start_epoch raises TypeError"""
        tracker = Tracker(project_name="test")
        with pytest.raises(TypeError):
            tracker.start_training(start_epoch="1")
