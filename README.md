# Eco2AI

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YOUR ML/AI CODE                                  │
│                                                                             │
│    import eco2ai                                                            │
│    tracker = eco2ai.Tracker(project_name="...")                            │
│    tracker.start()  ──────────────┐                                        │
│    # Your training/computation    │                                        │
│    tracker.stop()   ◄──────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ECO2AI TRACKER ENGINE                              │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │   CPU Tool   │    │   GPU Tool   │    │   RAM Tool   │                │
│  │  Power (W)   │    │  Power (W)   │    │  Power (W)   │                │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             ▼                                               │
│                  ┌─────────────────────┐                                   │
│                  │ Emission Calculator │                                   │
│                  │  • Regional Carbon  │                                   │
│                  │    Intensity (CI)   │                                   │
│                  │  • PUE Factor       │                                   │
│                  │  • Cost Pricing     │                                   │
│                  └──────────┬──────────┘                                   │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT & EXPORT                                     │
│                                                                             │
│   ┌─────────┐      ┌─────────┐      ┌──────────────┐                      │
│   │   CSV   │      │  JSON   │      │  JSONBin.io  │                      │
│   │  Local  │      │  Local  │      │    Cloud     │                      │
│   └─────────┘      └─────────┘      └──────┬───────┘                      │
│                                             │                               │
│   📊 Metrics: CO2(kg) • kWh • Duration • Cost • Hardware Info              │
└─────────────────────────────────────────────┼───────────────────────────────┘
                                              │
                                              │ API Access
                                              ▼
                                ┌──────────────────────────┐
                                │   FRONTEND UI/DASHBOARD  │
                                │                          │
                                │  • Real-time Monitoring  │
                                │  • Historical Analysis   │
                                │  • Emission Reports      │
                                │  • Cost Visualization    │
                                │  • Multi-Project Compare │
                                └──────────────────────────┘
```

A Python package for tracking CO2 emissions of AI/ML experiments and computational processes.

## Overview

Eco2AI is a carbon emission tracker designed to monitor and record the environmental impact of your AI/ML experiments and computational tasks. It tracks power consumption from CPU, GPU, and RAM, calculates the corresponding CO2 emissions based on your location's carbon intensity, and exports the results in multiple formats.

## Features

- **Real-time tracking** of CPU, GPU, and RAM power consumption
- **CO2 emission calculations** based on regional carbon intensity
- **Cross-platform support** (Windows, Linux, macOS)
- **Multiple export formats** (CSV, JSON)
- **Cloud integration** with JSONBin.io for storing results
- **Training process tracking** with per-epoch monitoring
- **Decorator support** for easy function tracking
- **Electricity cost estimation** based on time-of-day pricing
- **PUE (Power Usage Effectiveness)** support for data center calculations

## Installation

### From PyPI (when published)

```bash
pip install eco2ai
```

### From source

```bash
git clone https://github.com/YinuoYang327/eco2ai.git
cd eco2ai
pip install -e .
```

### For conda

```bash
conda install eco2ai
```

## Requirements

- Python >= 3.7
- APScheduler
- pynvml >= 5.6.2 (for NVIDIA GPU support)
- psutil
- py-cpuinfo
- numpy
- pandas
- tzlocal
- requests

## Quick Start

### Basic Usage

```python
import eco2ai

# Create a tracker
tracker = eco2ai.Tracker(
    project_name="My ML Project",
    experiment_description="Training a CNN model"
)

# Start tracking
tracker.start()

# Your computationally intensive code here
# ... model training, data processing, etc. ...

# Stop tracking and save results
tracker.stop()
```

Results will be saved to `emission.csv` by default.

### Using the Decorator

```python
from eco2ai import track

@track
def train_model():
    # Your training code here
    pass

train_model()  # Automatically tracked
```

### Tracking Training Process

```python
import eco2ai

tracker = eco2ai.Tracker(
    project_name="Deep Learning Experiment",
    experiment_description="ResNet-50 training"
)

tracker.start_training(start_epoch=1)

for epoch in range(1, num_epochs + 1):
    # Training code
    loss, accuracy = train_epoch(model, data_loader)

    # Record this epoch's metrics
    tracker.new_epoch({
        'loss': loss,
        'accuracy': accuracy
    })

tracker.stop_training()
```

## Advanced Configuration

### Custom File Output

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    file_name="my_emissions.csv"
)
```

### Specify Location for Accurate Carbon Intensity

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    alpha_2_code="US",  # ISO Alpha-2 country code
    region="California"
)
```

Find your country code at: https://www.iban.com/country-codes

### Custom Measurement Period

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    measure_period=5  # Measure every 5 seconds (default: 10)
)
```

### Data Center Usage (PUE)

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    pue=1.5  # Power Usage Effectiveness (default: 1.0)
)
```

### Electricity Cost Tracking

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    electricity_pricing={
        "8:30-19:00": 0.15,    # Peak hours: $0.15/kWh
        "19:00-6:00": 0.08,    # Night: $0.08/kWh
        "6:00-8:30": 0.12      # Morning: $0.12/kWh
    }
)

tracker.start()
# ... your code ...
tracker.stop()

print(f"Total cost: ${tracker.price()}")
```

### JSONBin.io Cloud Integration

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    jsonbin_api_key="your_api_key_here",
    jsonbin_bin_id="optional_existing_bin_id"
)
```

Results will be automatically uploaded to JSONBin.io when you call `tracker.stop()`.

### CPU Process Tracking

```python
tracker = eco2ai.Tracker(
    project_name="My Project",
    cpu_processes="current"  # Track only current process (default)
    # cpu_processes="all"    # Track all CPU processes
)
```

## Output Data

The tracker saves data with the following columns:

- `id`: Unique identifier for each tracking session
- `project_name`: Your project name
- `experiment_description`: Description of the experiment
- `epoch`: Epoch information (for training tracking)
- `start_time`: When tracking started
- `duration(s)`: Duration in seconds
- `power_consumption(kWh)`: Total power consumed
- `CO2_emissions(kg)`: CO2 emissions in kilograms
- `CPU_name`: CPU model and TDP information
- `GPU_name`: GPU model information
- `OS`: Operating system
- `region/country`: Location used for carbon intensity
- `cost`: Electricity cost (if pricing configured)

## API Reference

### Tracker Class

#### Constructor Parameters

- `project_name` (str): Name of your project
- `experiment_description` (str): Description of the experiment
- `file_name` (str): Output CSV file name (default: "emission.csv")
- `measure_period` (float): Measurement interval in seconds (default: 10)
- `emission_level` (float): Manual CO2 emission factor (kg CO2/MWh)
- `alpha_2_code` (str): ISO Alpha-2 country code
- `region` (str): Specific region/state within country
- `cpu_processes` (str): "current" or "all" (default: "current")
- `pue` (float): Power Usage Effectiveness (default: 1.0)
- `encode_file` (str): Encoded output file name
- `electricity_pricing` (dict): Time-based electricity pricing
- `ignore_warnings` (bool): Suppress warnings (default: False)
- `timezone` (str): Manual timezone setting
- `jsonbin_api_key` (str): JSONBin.io API key
- `jsonbin_bin_id` (str): Existing JSONBin.io bin ID

#### Methods

- `start()`: Begin tracking
- `stop()`: Stop tracking and save results
- `start_training(start_epoch=1)`: Begin training process tracking
- `new_epoch(parameters_dict)`: Record a new training epoch
- `stop_training()`: Stop training tracking
- `consumption()`: Get current power consumption
- `price()`: Get total electricity cost
- `id()`: Get tracker session ID
- `emission_level()`: Get CO2 emission factor
- `measure_period()`: Get measurement period

### Utility Functions

```python
from eco2ai import available_devices, set_params, get_params, summary

# Check available hardware
devices = available_devices()

# Set default parameters
set_params(
    project_name="Default Project",
    file_name="emission.csv",
    measure_period=10
)

# Get current parameters
params = get_params()

# Generate summary report
summary(file_name="emission.csv")
```

## Detailed Usage Examples

### Example 1: Machine Learning Model Training

```python
import eco2ai
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create tracker for ML experiment
tracker = eco2ai.Tracker(
    project_name="Random Forest Classification",
    experiment_description="Training RF with 1000 trees",
    file_name="ml_emissions.csv"
)

# Generate sample data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start tracking
tracker.start()

# Train model
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# Stop tracking
tracker.stop()

# Print metrics
print(f"Power Consumption: {tracker.consumption():.6f} kWh")
print(f"CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
```

### Example 2: Deep Learning Training with PyTorch

```python
import eco2ai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize tracker for training
tracker = eco2ai.Tracker(
    project_name="PyTorch MNIST",
    experiment_description="Simple NN with 3 layers",
    file_name="pytorch_emissions.csv",
    measure_period=5  # Measure every 5 seconds
)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# Start tracking training
num_epochs = 10
tracker.start_training(start_epoch=1)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    # Track this epoch
    tracker.new_epoch({
        'loss': avg_loss,
        'accuracy': accuracy
    })

tracker.stop_training()
print(f"\nTotal CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
```

### Example 3: Data Processing Pipeline

```python
import eco2ai
import pandas as pd
import numpy as np

@eco2ai.track
def process_large_dataset(file_path):
    """Process a large CSV file with automatic emission tracking."""
    # Read data
    df = pd.read_csv(file_path)

    # Heavy computation
    df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
    df['rolling_mean'] = df['value'].rolling(window=100).mean()

    # Group operations
    result = df.groupby('category').agg({
        'value': ['mean', 'std', 'min', 'max'],
        'normalized': 'mean'
    })

    return result

# This function will be automatically tracked
result = process_large_dataset('large_dataset.csv')
print("Processing complete! Check emission.csv for carbon footprint.")
```

### Example 4: Comparing Multiple Experiments

```python
import eco2ai
import time

def experiment_a():
    """CPU-intensive experiment."""
    result = sum(i**2 for i in range(10000000))
    return result

def experiment_b():
    """Memory-intensive experiment."""
    data = [list(range(1000)) for _ in range(10000)]
    result = sum(sum(row) for row in data)
    return result

# Track Experiment A
tracker_a = eco2ai.Tracker(
    project_name="Performance Comparison",
    experiment_description="CPU-intensive experiment A",
    file_name="comparison.csv"
)
tracker_a.start()
result_a = experiment_a()
tracker_a.stop()

print(f"Experiment A - CO2: {tracker_a.consumption() * tracker_a.emission_level() / 1000:.6f} kg")

# Track Experiment B
tracker_b = eco2ai.Tracker(
    project_name="Performance Comparison",
    experiment_description="Memory-intensive experiment B",
    file_name="comparison.csv"
)
tracker_b.start()
result_b = experiment_b()
tracker_b.stop()

print(f"Experiment B - CO2: {tracker_b.consumption() * tracker_b.emission_level() / 1000:.6f} kg")

# Generate comparison summary
from eco2ai import summary
summary_df = summary(filename="comparison.csv", write_to_file="comparison_summary.csv")
print("\nComparison Summary:")
print(summary_df)
```

### Example 5: Long-Running Task with Cost Tracking

```python
import eco2ai
import time
import random

# Configure tracker with time-based electricity pricing
tracker = eco2ai.Tracker(
    project_name="Data Analysis Pipeline",
    experiment_description="24-hour batch processing",
    file_name="batch_emissions.csv",
    electricity_pricing={
        "0:00-6:00": 0.08,    # Off-peak: $0.08/kWh
        "6:00-9:00": 0.12,    # Morning: $0.12/kWh
        "9:00-17:00": 0.18,   # Peak: $0.18/kWh
        "17:00-21:00": 0.15,  # Evening: $0.15/kWh
        "21:00-0:00": 0.10    # Night: $0.10/kWh
    },
    measure_period=30  # Check every 30 seconds
)

tracker.start()

# Simulate long-running batch processing
print("Starting batch processing...")
for i in range(10):
    # Simulate computation
    _ = sum(j**2 for j in range(1000000))
    time.sleep(2)  # Simulate I/O wait
    print(f"Batch {i+1}/10 complete")

tracker.stop()

# Display cost and emissions
print(f"\n{'='*50}")
print(f"Processing Complete!")
print(f"{'='*50}")
print(f"Power Consumption: {tracker.consumption():.6f} kWh")
print(f"CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
print(f"Electricity Cost: ${tracker.price():.4f}")
print(f"{'='*50}")
```

### Example 6: Hyperparameter Tuning with Tracking

```python
import eco2ai
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Track hyperparameter search
tracker = eco2ai.Tracker(
    project_name="Hyperparameter Tuning",
    experiment_description="GridSearch for Random Forest",
    file_name="tuning_emissions.csv"
)

tracker.start()

# Perform grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

tracker.stop()

print(f"Best Parameters: {best_params}")
print(f"Best CV Score: {best_score:.4f}")
print(f"CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
print(f"Power Consumed: {tracker.consumption():.6f} kWh")
```

### Example 7: Cloud/Data Center Usage with PUE

```python
import eco2ai

# For data center deployment, account for PUE (Power Usage Effectiveness)
# PUE = Total Facility Power / IT Equipment Power
# Typical values: 1.2 (very efficient) to 2.0 (average)

tracker = eco2ai.Tracker(
    project_name="Cloud Model Training",
    experiment_description="ResNet training on AWS",
    file_name="cloud_emissions.csv",
    pue=1.58,  # AWS average PUE
    alpha_2_code="US",
    region="Virginia",  # AWS us-east-1
    measure_period=15
)

tracker.start()

# Your cloud-based training code here
# ... model training ...

tracker.stop()

# The emissions will account for the full data center overhead
print(f"Total emissions (including PUE): {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg CO2")
```

## Additional Examples

For more detailed examples, see the `examples/` directory:

- `basic_tracking.py` - Simple tracking example
- `training_tracking.py` - Training process monitoring
- `decorator_example.py` - Using the @track decorator
- `custom_config.py` - Custom configuration options
- `electricity_pricing.py` - Time-based pricing example
- `jsonbin_example.py` - Cloud storage integration

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Eco2AI in your research, please cite:

```bibtex
@software{eco2ai,
  author = {Yang, Yinuo},
  title = {Eco2AI: Carbon Emission Tracker for AI/ML Experiments},
  year = {2025},
  url = {https://github.com/YinuoYang327/eco2ai}
}
```

## Support

- GitHub Issues: https://github.com/YinuoYang327/eco2ai/issues
- Documentation: https://github.com/YinuoYang327/eco2ai

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
