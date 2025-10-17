"""
Custom Configuration Example

This example demonstrates advanced configuration options
including location settings, PUE, and measurement periods.
"""

import eco2ai
import time


def main():
    print("=== Custom Configuration Example ===\n")

    # Example 1: Specify location for accurate carbon intensity
    print("Example 1: Custom Location")
    tracker1 = eco2ai.Tracker(
        project_name="Custom Config Example",
        experiment_description="Tracking with custom location",
        file_name="custom_emission.csv",
        alpha_2_code="US",  # United States
        region="California",
        ignore_warnings=True
    )

    tracker1.start()
    time.sleep(3)
    tracker1.stop()

    print(f"  Location: {tracker1._country}")
    print(f"  Emission Level: {tracker1.emission_level():.2f} kg CO2/MWh\n")

    # Example 2: Data Center with PUE
    print("Example 2: Data Center Configuration")
    tracker2 = eco2ai.Tracker(
        project_name="Data Center Example",
        experiment_description="Tracking with PUE",
        file_name="datacenter_emission.csv",
        pue=1.5,  # Power Usage Effectiveness
        measure_period=5,  # Measure every 5 seconds
        ignore_warnings=True
    )

    tracker2.start()
    time.sleep(3)
    tracker2.stop()

    print(f"  PUE: {tracker2._pue}")
    print(f"  Measure Period: {tracker2.measure_period()} seconds")
    print(f"  Consumption: {tracker2.consumption():.6f} kWh\n")

    # Example 3: CPU Process Tracking
    print("Example 3: CPU Process Tracking")
    tracker3 = eco2ai.Tracker(
        project_name="CPU Tracking Example",
        experiment_description="Track all CPU processes",
        file_name="cpu_emission.csv",
        cpu_processes="all",  # Track all processes (default: "current")
        ignore_warnings=True
    )

    tracker3.start()
    time.sleep(3)
    tracker3.stop()

    print(f"  CPU Processes: {tracker3._cpu_processes}\n")

    print("=== All Examples Complete ===")
    print("Check the generated CSV files for detailed results")


if __name__ == "__main__":
    main()
