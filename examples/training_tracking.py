"""
Training Tracking Example

This example demonstrates how to track CO2 emissions
during a machine learning training process with per-epoch tracking.
"""

import eco2ai
import time
import random


def simulate_training_epoch(epoch):
    """Simulate training for one epoch"""
    # Simulate computation
    time.sleep(2)

    # Return simulated metrics
    loss = 1.0 / (epoch + 1) + random.uniform(0, 0.1)
    accuracy = min(0.95, 0.5 + epoch * 0.1 + random.uniform(0, 0.05))

    return loss, accuracy


def main():
    print("=== ML Training Tracking Example ===\n")

    # Create tracker
    tracker = eco2ai.Tracker(
        project_name="ML Training Example",
        experiment_description="ResNet-50 training simulation",
        file_name="training_emission.csv",
        measure_period=5
    )

    # Start training tracking
    num_epochs = 5
    print(f"Starting training for {num_epochs} epochs...\n")
    tracker.start_training(start_epoch=1)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Simulate training
        loss, accuracy = simulate_training_epoch(epoch)

        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        # Record this epoch's metrics
        tracker.new_epoch({
            'loss': loss,
            'accuracy': accuracy,
            'learning_rate': 0.001
        })

        print()

    # Stop training tracking
    tracker.stop_training()

    print("=== Training Complete ===")
    print(f"Results saved to: {tracker.file_name}")


if __name__ == "__main__":
    main()
