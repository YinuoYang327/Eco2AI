# Testing Guide

## Running Tests Locally

### Option 1: Using Isolated Virtual Environment (Recommended)

This approach keeps test dependencies isolated and can be easily cleaned up:

```bash
# 1. Create isolated test environment
py -m venv test_env_temp

# 2. Install package and dependencies
test_env_temp\Scripts\pip install -e .
test_env_temp\Scripts\pip install pytest pytest-cov

# 3. Run tests
test_env_temp\Scripts\pytest tests/ -v

# 4. Clean up when done (removes ALL test dependencies)
rmdir /s /q test_env_temp
```

### Option 2: Using System Python

```bash
# Install dependencies
pip install -e .
pip install pytest pytest-cov

# Run tests
pytest tests/ -v
```

## Test Coverage

View detailed coverage report:

```bash
pytest tests/ --cov=eco2ai --cov-report=html
# Open htmlcov/index.html in browser
```

## Running Specific Tests

```bash
# Run only import tests
pytest tests/test_import.py -v

# Run only tracker tests
pytest tests/test_tracker.py -v

# Run only tool tests
pytest tests/test_tools.py -v

# Run only utility tests
pytest tests/test_utils.py -v
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- Ubuntu, Windows, macOS

See `.github/workflows/ci.yml` for details.

## Cleanup Test Files

The test suite automatically cleans up test files using pytest fixtures.
If you need to manually clean up:

```bash
# Remove test emission files
del *emission*.csv *emission*.json encoded_*.csv 2>nul
```

## Quick Cleanup Script

To remove the isolated test environment:

**Windows:**
```cmd
rmdir /s /q test_env_temp
```

**Linux/macOS:**
```bash
rm -rf test_env_temp
```
