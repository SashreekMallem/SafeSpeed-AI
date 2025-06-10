#!/usr/bin/env python3

"""
Test script for the improved License Plate Recognition pipeline
This script tests the pipeline with a sample video file
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    # Check if the sample video exists
    if not os.path.exists("sample.mp4"):
        print("Error: sample.mp4 not found in current directory")
        return False
    
    # Check if configuration files exist
    config_files = [
        "dstest2_pgie_config.txt",
        "dstest2_sgie1_config.txt", 
        "dstest2_sgie2_config.txt",
        "dstest2_sgie3_config.txt",
        "dstest2_tracker_config.txt"
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"Warning: {config_file} not found")
    
    print("Dependencies check completed")
    return True

def run_pipeline_test():
    """Run the license plate recognition pipeline"""
    print("\n" + "="*50)
    print("STARTING LICENSE PLATE RECOGNITION PIPELINE TEST")
    print("="*50)
    
    try:
        # Run the pipeline with the sample video
        cmd = ["python3", "deepstream_test_2.py", "file", "sample.mp4"]
        print(f"Running command: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
            # Look for license plate detections
            if "License Plate Detected:" in line:
                print(f"✅ SUCCESS: {line.strip()}")
            elif "License Plate Detection -" in line:
                print(f"🔍 DETECTION: {line.strip()}")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"Error running pipeline: {e}")

def main():
    print("License Plate Recognition Pipeline Tester")
    print("-" * 40)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\nThis will run the LPR pipeline with the following improvements:")
    print("✅ Enhanced OCR with EasyOCR + Tesseract fallback")
    print("✅ Better image preprocessing for license plates")
    print("✅ Result smoothing to reduce flickering")
    print("✅ Improved detection logic for SGIE3 LPD results")
    print("✅ Debug information for troubleshooting")
    
    input("\nPress Enter to start the test (Ctrl+C to stop)...")
    
    run_pipeline_test()

if __name__ == "__main__":
    main()
