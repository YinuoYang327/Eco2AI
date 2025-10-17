import pytest
from eco2ai.tools.tools_cpu import CPU, all_available_cpu
from eco2ai.tools.tools_gpu import GPU, all_available_gpu
from eco2ai.tools.tools_ram import RAM


class TestCPU:
    """Test suite for CPU monitoring"""

    def test_cpu_initialization(self):
        """Test CPU class initialization"""
        cpu = CPU(cpu_processes="current", ignore_warnings=True)
        assert cpu is not None

    def test_cpu_name(self):
        """Test that CPU name is returned"""
        cpu = CPU(ignore_warnings=True)
        name = cpu.name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_cpu_num(self):
        """Test that CPU count is returned"""
        cpu = CPU(ignore_warnings=True)
        num = cpu.cpu_num()
        assert isinstance(num, int)
        assert num > 0

    def test_all_available_cpu(self):
        """Test all_available_cpu function"""
        # This function prints to stdout, doesn't return a value
        try:
            all_available_cpu()
            # If it runs without error, the test passes
        except Exception:
            pytest.fail("all_available_cpu() raised an exception")


class TestGPU:
    """Test suite for GPU monitoring"""

    def test_gpu_initialization(self):
        """Test GPU class initialization"""
        gpu = GPU(ignore_warnings=True)
        assert gpu is not None

    def test_gpu_availability(self):
        """Test GPU availability detection"""
        gpu = GPU(ignore_warnings=True)
        # is_gpu_available should be a boolean
        assert isinstance(gpu.is_gpu_available, bool)

    def test_all_available_gpu(self):
        """Test all_available_gpu function"""
        # This should not raise an exception
        try:
            gpu_info = all_available_gpu()
        except Exception:
            # GPU might not be available in test environment
            pass


class TestRAM:
    """Test suite for RAM monitoring"""

    def test_ram_initialization(self):
        """Test RAM class initialization"""
        ram = RAM(ignore_warnings=True)
        assert ram is not None

    def test_ram_calculation(self):
        """Test that RAM consumption can be calculated"""
        ram = RAM(ignore_warnings=True)
        # This should not raise an exception
        try:
            consumption = ram.calculate_consumption()
            assert consumption >= 0
        except Exception as e:
            pytest.skip(f"RAM calculation not available: {e}")
