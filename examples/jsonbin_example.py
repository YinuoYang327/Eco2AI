"""
JSONBin Integration Example

This example demonstrates how to upload tracking results
to JSONBin.io for cloud storage and sharing.

Note: You need a JSONBin.io API key to use this feature.
Get one free at: https://jsonbin.io/
"""

import eco2ai
import time


def main():
    print("=== JSONBin Integration Example ===\n")

    # IMPORTANT: Replace with your actual JSONBin.io API key
    API_KEY = "your_jsonbin_api_key_here"

    # Optional: If updating an existing bin, provide the bin ID
    # BIN_ID = "your_bin_id_here"

    if API_KEY == "your_jsonbin_api_key_here":
        print("ERROR: Please set your JSONBin.io API key in the script!")
        print("Get a free API key at: https://jsonbin.io/")
        return

    # Create tracker with JSONBin integration
    tracker = eco2ai.Tracker(
        project_name="JSONBin Example",
        experiment_description="Upload results to cloud",
        file_name="jsonbin_emission.csv",
        jsonbin_api_key=API_KEY,
        # jsonbin_bin_id=BIN_ID,  # Uncomment to update existing bin
        ignore_warnings=True
    )

    # Start tracking
    print("Starting computation...")
    tracker.start()

    # Simulate work
    time.sleep(3)
    result = sum(i ** 2 for i in range(1000000))

    # Stop tracking - this will automatically upload to JSONBin
    print("Stopping tracker and uploading to JSONBin...\n")
    tracker.stop()

    print("=== Results ===")
    print("Results have been:")
    print("  1. Saved locally to CSV and JSON files")
    print("  2. Uploaded to JSONBin.io")
    print("\nCheck the console output above for your JSONBin URL!")


if __name__ == "__main__":
    main()
