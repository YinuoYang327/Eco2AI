"""
Decorator Example

This example demonstrates using the @track decorator
to automatically track emissions of any function.
"""

import eco2ai
import time


@eco2ai.track
def heavy_computation():
    """
    A computationally intensive function that will be
    automatically tracked for CO2 emissions.
    """
    print("Running heavy computation...")

    # Simulate some heavy computation
    result = 0
    for i in range(10000000):
        result += i ** 2

    time.sleep(3)

    print("Computation complete!")
    return result


@eco2ai.track
def data_processing():
    """Another function to demonstrate decorator usage"""
    print("Processing data...")

    # Simulate data processing
    data = [i for i in range(1000000)]
    processed = [x ** 2 for x in data if x % 2 == 0]

    time.sleep(2)

    print("Data processing complete!")
    return processed


def main():
    print("=== Decorator Example ===\n")

    # Call decorated functions - tracking happens automatically
    print("Example 1: Heavy Computation")
    result1 = heavy_computation()
    print(f"Result: {result1}\n")

    print("Example 2: Data Processing")
    result2 = data_processing()
    print(f"Processed {len(result2)} items\n")

    print("=== All Results Saved ===")
    print("Check 'emission.csv' for detailed emissions data")


if __name__ == "__main__":
    main()
