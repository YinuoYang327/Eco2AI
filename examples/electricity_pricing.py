"""
Electricity Pricing Example

This example demonstrates how to track electricity costs
with time-based pricing.
"""

import eco2ai
import time


def main():
    print("=== Electricity Pricing Example ===\n")

    # Define time-based electricity pricing
    # Prices are per kWh
    pricing = {
        "8:30-19:00": 0.15,    # Peak hours: $0.15/kWh
        "19:00-6:00": 0.08,    # Night: $0.08/kWh
        "6:00-8:30": 0.12      # Morning: $0.12/kWh
    }

    print("Electricity Pricing Schedule:")
    print("  Peak (8:30-19:00): $0.15/kWh")
    print("  Night (19:00-6:00): $0.08/kWh")
    print("  Morning (6:00-8:30): $0.12/kWh\n")

    # Create tracker with electricity pricing
    tracker = eco2ai.Tracker(
        project_name="Cost Tracking Example",
        experiment_description="Tracking with electricity costs",
        file_name="cost_emission.csv",
        electricity_pricing=pricing,
        measure_period=5,
        ignore_warnings=True
    )

    # Start tracking
    print("Starting computation...")
    tracker.start()

    # Simulate some computational work
    result = 0
    for i in range(10000000):
        result += i ** 2
    time.sleep(5)

    # Stop tracking
    tracker.stop()

    # Display results
    print("\n=== Results ===")
    print(f"Power Consumption: {tracker.consumption():.6f} kWh")
    print(f"CO2 Emissions: {tracker.consumption() * tracker.emission_level() / 1000:.6f} kg")
    print(f"Electricity Cost: ${tracker.price():.4f}")
    print(f"\nDetailed results saved to: {tracker.file_name}")


if __name__ == "__main__":
    main()
