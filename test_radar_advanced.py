#!/usr/bin/env python3

"""
Advanced AWR1843BOOST Test Script
Tests radar configuration and data collection
"""

import serial
import time
import sys
import struct

def test_radar_advanced():
    """Advanced radar testing with configuration"""
    
    print("=" * 50)
    print("AWR1843BOOST Advanced Test")
    print("=" * 50)
    
    # Connect to both ports
    try:
        print("Connecting to radar ports...")
        config_port = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
        data_port = serial.Serial('/dev/ttyACM1', 921600, timeout=2)
        print("✅ Connected to both ports")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Clear any existing data
    config_port.reset_input_buffer()
    data_port.reset_input_buffer()
    
    # Send basic commands
    test_commands = [
        "version",
        "sensorStop", 
        "flushCfg"
    ]
    
    print("\n🔧 Testing Basic Commands:")
    for cmd in test_commands:
        print(f"Sending: {cmd}")
        config_port.write((cmd + '\n').encode())
        time.sleep(0.2)
        
        response = ""
        while config_port.in_waiting > 0:
            response += config_port.read(config_port.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)
        
        if response.strip():
            print(f"Response: {response.strip()}")
        else:
            print("No response")
        print("-" * 30)
    
    # Test a simple configuration
    print("\n🛰️ Testing Simple Configuration:")
    simple_config = [
        "sensorStop",
        "flushCfg", 
        "dfeDataOutputMode 1",
        "channelCfg 15 7 0",
        "adcCfg 2 1",
        "profileCfg 0 77 429 90 1 0 0 70 1 256 5209 0 0 0",
        "chirpCfg 0 0 0 0 0 0 0 1",
        "frameCfg 0 1 16 0 100 1 0"
    ]
    
    for cmd in simple_config:
        print(f"Config: {cmd}")
        config_port.write((cmd + '\n').encode())
        time.sleep(0.1)
        
        response = ""
        while config_port.in_waiting > 0:
            response += config_port.read(config_port.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.05)
        
        if "Done" in response or "Succeeded" in response:
            print("✅ Success")
        elif response.strip():
            print(f"Response: {response.strip()}")
        else:
            print("📡 Sent")
    
    # Try to start sensor
    print("\n🚀 Starting Sensor:")
    config_port.write(b'sensorStart\n')
    time.sleep(1)
    
    response = ""
    while config_port.in_waiting > 0:
        response += config_port.read(config_port.in_waiting).decode('utf-8', errors='ignore')
        time.sleep(0.1)
    
    print(f"Start response: {response}")
    
    # Check for data on data port
    print("\n📊 Checking for data stream...")
    time.sleep(2)  # Wait for data
    
    if data_port.in_waiting > 0:
        data_bytes = data_port.read(min(100, data_port.in_waiting))
        print(f"✅ Data received: {len(data_bytes)} bytes")
        print(f"First few bytes: {data_bytes[:20].hex()}")
    else:
        print("❌ No data received")
    
    # Stop sensor
    config_port.write(b'sensorStop\n')
    time.sleep(0.5)
    
    # Close ports
    config_port.close()
    data_port.close()
    
    print("\n✅ Advanced test completed!")
    return True

if __name__ == "__main__":
    test_radar_advanced()
