import pytest
from eco2ai.utils import (
    get_params,
    set_params,
    electricity_pricing_check,
)


class TestUtils:
    """Test suite for utility functions"""

    def test_get_set_params(self):
        """Test parameter getting and setting"""
        set_params(
            project_name="test_project",
            file_name="test.csv",
            measure_period=15
        )

        params = get_params()
        assert params["project_name"] == "test_project"
        assert params["file_name"] == "test.csv"
        assert params["measure_period"] == 15

    def test_electricity_pricing_valid(self):
        """Test valid electricity pricing configuration"""
        pricing = {
            "8:30-19:00": 0.15,
            "19:00-6:00": 0.08,
            "6:00-8:30": 0.12
        }
        # Should not raise any exception
        electricity_pricing_check(pricing)

    def test_electricity_pricing_none(self):
        """Test that None pricing is valid"""
        # Should not raise any exception
        electricity_pricing_check(None)
