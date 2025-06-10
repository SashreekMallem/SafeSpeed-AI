#!/bin/bash

"""
AWR1843BOOST Setup Script for SafeSpeed AI
Sets up the radar sensor for integration with camera system
"""

echo "=========================================="
echo "AWR1843BOOST Setup for SafeSpeed AI"
echo "=========================================="

# Check if running as root for device permissions
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  Running as root. Consider adding user to dialout group instead."
fi

# Install required Python packages
echo "📦 Installing required Python packages..."
pip3 install pyserial numpy

# Check for AWR1843BOOST device
echo "🔍 Checking for AWR1843BOOST device..."

# Look for Texas Instruments devices
if lsusb | grep -i "texas instruments"; then
    echo "✅ TI device found in USB list"
    lsusb | grep -i "texas"
else
    echo "❌ No TI device found. Make sure AWR1843BOOST is connected."
fi

# Check for serial devices
echo "🔍 Checking for serial devices..."
if ls /dev/ttyACM* 2>/dev/null; then
    echo "✅ ACM devices found:"
    ls -la /dev/ttyACM*
else
    echo "❌ No /dev/ttyACM* devices found"
fi

# Check user permissions
echo "🔍 Checking user permissions..."
if groups $USER | grep -q dialout; then
    echo "✅ User $USER is in dialout group"
else
    echo "⚠️  User $USER is NOT in dialout group"
    echo "   Run: sudo usermod -a -G dialout $USER"
    echo "   Then logout and login again"
fi

# Create radar configuration directory
echo "📁 Creating radar configuration directory..."
mkdir -p radar_configs

# Create basic radar configuration file
echo "📝 Creating radar configuration files..."

cat > radar_configs/awr1843_basic.cfg << 'EOF'
% AWR1843 Basic Configuration for Vehicle Detection
% Range: ~200m, Velocity: ±50 m/s

sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 429 90 1 0 0 70 1 256 5209 0 0 0
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
frameCfg 0 1 16 0 100 1 0
lowPower 0 0
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
lvdsStreamCfg -1 0 0 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 4 127 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0 8.92
cfarFovCfg -1 1 -1 1.00
sensorStart
EOF

# Create radar test script
cat > test_radar_connection.py << 'EOF'
#!/usr/bin/env python3

import serial
import time
import sys

def test_radar_connection():
    """Test connection to AWR1843BOOST"""
    
    print("Testing AWR1843BOOST Connection...")
    
    # Try common ports
    ports_to_try = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
    
    for port in ports_to_try:
        try:
            print(f"Trying port {port}...")
            ser = serial.Serial(port, 115200, timeout=2)
            
            # Send a simple command
            ser.write(b'version\n')
            time.sleep(0.5)
            
            response = ser.read_all()
            if response:
                print(f"✅ Connected to radar on {port}")
                print(f"Response: {response.decode('utf-8', errors='ignore')}")
                ser.close()
                return port
            else:
                print(f"❌ No response on {port}")
            
            ser.close()
            
        except Exception as e:
            print(f"❌ Failed to connect to {port}: {e}")
    
    print("❌ Could not connect to radar on any port")
    return None

if __name__ == "__main__":
    test_radar_connection()
EOF

chmod +x test_radar_connection.py

# Create udev rules for consistent device naming
echo "📝 Creating udev rules for AWR1843BOOST..."

cat > 99-awr1843.rules << 'EOF'
# AWR1843BOOST udev rules
# Texas Instruments AWR1843 mmWave Radar
SUBSYSTEM=="tty", ATTRS{idVendor}=="0451", ATTRS{idProduct}=="fd03", SYMLINK+="awr1843_config"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0451", ATTRS{idProduct}=="fd03", SYMLINK+="awr1843_data"
EOF

echo "📋 To install udev rules (optional):"
echo "   sudo cp 99-awr1843.rules /etc/udev/rules.d/"
echo "   sudo udevadm control --reload-rules"

# Create integration test script
cat > test_radar_fusion.py << 'EOF'
#!/usr/bin/env python3

"""
Test script for radar fusion integration
"""

import sys
import time
from radar_fusion_system import RadarFusionPipeline, CameraTarget

def test_fusion():
    """Test the radar fusion system"""
    
    print("Testing Radar Fusion System...")
    
    # Initialize radar
    fusion = RadarFusionPipeline()
    
    if not fusion.initialize():
        print("❌ Radar initialization failed")
        return False
    
    print("✅ Radar initialized successfully")
    
    # Create mock camera targets
    mock_camera_targets = [
        CameraTarget(
            bbox=(100, 200, 150, 80),
            license_plate="ABC123",
            vehicle_type="sedan",
            confidence=0.85,
            timestamp=time.time()
        )
    ]
    
    # Test fusion for 10 seconds
    print("Testing fusion for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        fusion.process_camera_detections(mock_camera_targets)
        fused_results = fusion.get_fused_results()
        
        if fused_results:
            for result in fused_results:
                print(f"FUSED: LP={result.license_plate}, Speed={result.speed_mph:.1f}mph, Range={result.range_m:.1f}m")
        else:
            print("No radar targets detected")
        
        time.sleep(1)
    
    fusion.stop()
    print("✅ Fusion test completed")
    return True

if __name__ == "__main__":
    test_fusion()
EOF

chmod +x test_radar_fusion.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. 🔌 Connect your AWR1843BOOST via USB"
echo "2. 🧪 Test connection: python3 test_radar_connection.py"
echo "3. 🔬 Test fusion: python3 test_radar_fusion.py"
echo "4. 🚀 Run enhanced pipeline: python3 deepstream_radar_fusion.py file sample.mp4"
echo ""
echo "Troubleshooting:"
echo "- If permission denied: sudo usermod -a -G dialout $USER (then logout/login)"
echo "- If no device found: Check USB connection and install TI drivers"
echo "- If radar not working: Try different /dev/ttyACM* ports"
echo ""
echo "Configuration files created:"
echo "- radar_configs/awr1843_basic.cfg"
echo "- test_radar_connection.py"
echo "- test_radar_fusion.py"
echo "- 99-awr1843.rules (udev rules)"
