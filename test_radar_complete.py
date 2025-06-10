#!/usr/bin/env python3

"""
Complete AWR1843BOOST Configuration and Test
Implements full radar configuration for vehicle detection
"""

import serial
import time
import sys
import threading
import struct

class AWR1843Controller:
    def __init__(self):
        self.config_port = None
        self.data_port = None
        self.is_collecting = False
        
    def connect(self):
        """Connect to radar ports"""
        try:
            print("Connecting to AWR1843BOOST...")
            self.config_port = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
            self.data_port = serial.Serial('/dev/ttyACM1', 921600, timeout=2)
            
            # Clear buffers
            self.config_port.reset_input_buffer()
            self.data_port.reset_input_buffer()
            
            print("✅ Connected successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def send_config_command(self, command, expect_done=True):
        """Send configuration command and check response"""
        print(f"📡 {command}")
        self.config_port.write((command + '\n').encode())
        time.sleep(0.1)
        
        response = ""
        timeout = time.time() + 2
        while time.time() < timeout:
            if self.config_port.in_waiting > 0:
                response += self.config_port.read(self.config_port.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.05)
            if "Done" in response or "mmwDemo:/>" in response:
                break
        
        if expect_done and "Done" in response:
            print("   ✅ Success")
            return True
        elif "Error" in response:
            print(f"   ❌ Error: {response}")
            return False
        else:
            print(f"   📋 {response.strip()}")
            return True
    
    def configure_radar(self):
        """Complete radar configuration for vehicle detection"""
        print("\n🛰️ Configuring Radar for Vehicle Detection...")
        
        # Complete configuration sequence
        config_commands = [
            "sensorStop",
            "flushCfg",
            "dfeDataOutputMode 1",
            "channelCfg 15 7 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 77 429 90 1 0 0 70 1 256 5209 0 0 0",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "chirpCfg 1 1 0 0 0 0 0 4",
            "frameCfg 0 1 16 0 100 1 0",
            "lowPower 0 0",
            "guiMonitor -1 1 1 0 0 0 1",
            "cfarCfg -1 0 2 8 4 3 0 15 1",
            "cfarCfg -1 1 0 4 2 3 1 15 1", 
            "multiObjBeamForming -1 1 0.5",
            "clutterRemoval -1 0",
            "calibDcRangeSig -1 0 -5 8 256",
            "extendedMaxVelocity -1 0",
            "lvdsStreamCfg -1 0 0 0",
            "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
            "CQRxSatMonitor 0 3 4 127 0",
            "CQSigImgMonitor 0 127 4",
            "analogMonitor 0 0",
            "aoaFovCfg -1 -90 90 -90 90",
            "cfarFovCfg -1 0 0 8.92",
            "cfarFovCfg -1 1 -1 1.00"
        ]
        
        success_count = 0
        for cmd in config_commands:
            if self.send_config_command(cmd):
                success_count += 1
            time.sleep(0.05)
        
        print(f"\n📊 Configuration: {success_count}/{len(config_commands)} commands successful")
        return success_count == len(config_commands)
    
    def start_sensor(self):
        """Start the radar sensor"""
        print("\n🚀 Starting Radar Sensor...")
        return self.send_config_command("sensorStart", expect_done=False)
    
    def collect_data(self, duration=10):
        """Collect radar data for specified duration"""
        print(f"\n📊 Collecting data for {duration} seconds...")
        
        start_time = time.time()
        frame_count = 0
        total_bytes = 0
        
        while time.time() - start_time < duration:
            if self.data_port.in_waiting > 0:
                data = self.data_port.read(self.data_port.in_waiting)
                total_bytes += len(data)
                
                # Look for frame headers (magic word: 0x0201, 0x0403, 0x0605, 0x0807)
                magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
                if magic_word in data:
                    frame_count += data.count(magic_word)
                    print(f"📡 Frame {frame_count} detected ({len(data)} bytes)")
            
            time.sleep(0.1)
        
        print(f"\n📈 Data Collection Summary:")
        print(f"   Duration: {duration} seconds")
        print(f"   Frames detected: {frame_count}")
        print(f"   Total bytes: {total_bytes}")
        print(f"   Data rate: {total_bytes/duration:.1f} bytes/sec")
        
        return frame_count > 0
    
    def stop_sensor(self):
        """Stop the radar sensor"""
        print("\n🛑 Stopping Radar Sensor...")
        return self.send_config_command("sensorStop")
    
    def disconnect(self):
        """Disconnect from radar"""
        if self.config_port:
            self.config_port.close()
        if self.data_port:
            self.data_port.close()
        print("🔌 Disconnected")

def main():
    """Main test function"""
    print("=" * 60)
    print("AWR1843BOOST Complete Configuration Test")
    print("=" * 60)
    
    radar = AWR1843Controller()
    
    # Connect
    if not radar.connect():
        return False
    
    try:
        # Configure
        if not radar.configure_radar():
            print("❌ Configuration failed")
            return False
        
        # Start sensor
        if not radar.start_sensor():
            print("❌ Failed to start sensor")
            return False
        
        # Collect data
        data_received = radar.collect_data(duration=15)
        
        if data_received:
            print("\n🎉 SUCCESS: Radar is working and collecting data!")
        else:
            print("\n⚠️ Radar started but no data frames detected")
        
        # Stop sensor
        radar.stop_sensor()
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        radar.disconnect()
    
    return True

if __name__ == "__main__":
    main()
