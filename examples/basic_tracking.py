"""
Basic Tracking Example

This example demonstrates the most basic usage of Eco2AI
to track CO2 emissions of a computational task.
"""

import eco2ai
import time


def main():
    print("=== Basic Eco2AI Tracking Example ===\n")

    # Create a tracker
    tracker = eco2ai.Tracker(
        project_name="Basic Example",
        experiment_description="Simple CPU computation",
        file_name="basic_emission.csv"
    )

    # Start tracking
    print("Starting emission tracking...")
    tracker.start()

    # Simulate some computational work
    print("Running computation...")
    result = 0
    for i in range(10000000):
        result += i ** 2
    time.sleep(5)

    # Stop tracking
    print("Stopping tracker...\n")
    tracker.stop()

    # Display results
    print("=== Results ===")
    print(f"Tracker ID: {tracker.id()}")
    print(f"Power Consumption: {tracker.consumption():.6f} kWh")
    print(f"CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
    print(f"\nResults saved to: {tracker.file_name}")


if __name__ == "__main__":
    main()
