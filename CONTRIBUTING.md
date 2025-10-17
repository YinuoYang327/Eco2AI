# Contributing to Eco2AI

Thank you for your interest in contributing to Eco2AI! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/YinuoYang327/eco2ai.git
   cd eco2ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install the package in editable mode**
   ```bash
   pip install -e .
   ```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Your Changes

- Write clear, readable code
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Update documentation if needed

### 3. Code Quality

Before committing, ensure your code meets quality standards:

#### Format your code with Black
```bash
black .
```

#### Check code style with flake8
```bash
flake8 eco2ai/ tests/
```

#### Sort imports with isort
```bash
isort .
```

#### Type checking with mypy
```bash
mypy eco2ai/
```

### 4. Write Tests

All new features and bug fixes should include tests:

- Add tests to the appropriate file in `tests/`
- Ensure all tests pass:
  ```bash
  pytest tests/ -v
  ```
- Check test coverage:
  ```bash
  pytest tests/ --cov=eco2ai --cov-report=html
  ```

Aim for at least 80% code coverage for new features.

### 5. Update Documentation

- Update the README.md if you're adding new features
- Add docstrings following NumPy/Google style
- Update CHANGELOG.md with your changes

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of the feature"
```

Commit message format:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests when relevant

Examples:
- `Add electricity pricing validation`
- `Fix version inconsistency in setup.py`
- `Update README with new examples`

### 7. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub:
- Provide a clear title and description
- Reference any related issues
- Explain what changed and why
- Include screenshots for UI changes

## Types of Contributions

### Bug Reports

When filing a bug report, include:
- Python version
- Operating system
- Eco2AI version
- Minimal code to reproduce the issue
- Expected behavior
- Actual behavior
- Error messages and stack traces

### Feature Requests

When proposing a new feature:
- Explain the use case
- Describe the desired behavior
- Provide examples if possible
- Consider backward compatibility

### Code Contributions

We welcome:
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tracker.py

# Run with coverage
pytest tests/ --cov=eco2ai --cov-report=html

# Run specific test
pytest tests/test_tracker.py::TestTracker::test_tracker_initialization
```

### Writing Tests

- Use pytest framework
- Follow naming convention: `test_*.py` for files, `test_*` for functions
- Use fixtures for common setup
- Mock external dependencies
- Test edge cases and error conditions

Example:
```python
def test_tracker_initialization():
    """Test that Tracker initializes with default parameters"""
    tracker = Tracker(project_name="test_project")
    assert tracker.project_name == "test_project"
    assert tracker.file_name == "emission.csv"
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 120 characters
- Use meaningful variable and function names
- Add docstrings to all public functions and classes

### Docstring Format

Use NumPy style docstrings:

```python
def calculate_emissions(consumption, emission_level):
    """
    Calculate CO2 emissions based on power consumption.

    Parameters
    ----------
    consumption : float
        Power consumption in kWh
    emission_level : float
        Carbon intensity in kg CO2/MWh

    Returns
    -------
    float
        CO2 emissions in kg

    Examples
    --------
    >>> calculate_emissions(1.0, 500)
    0.5
    """
    return consumption * emission_level / 1000
```

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Update documentation
   - Add entry to CHANGELOG.md
   - Rebase on latest main branch

2. **PR Review:**
   - Maintainers will review your PR
   - Address any feedback
   - Keep the PR focused on a single feature/fix

3. **After Approval:**
   - PR will be merged by a maintainer
   - Delete your feature branch

## Release Process

Maintainers will:
1. Update version in `setup.py` and `eco2ai/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag
4. Publish to PyPI via GitHub Actions

## Questions?

If you have questions:
- Open a GitHub issue
- Check existing issues and PRs
- Review the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors page
- Future documentation

Thank you for contributing to Eco2AI!
