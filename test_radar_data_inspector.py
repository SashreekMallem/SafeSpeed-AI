#!/usr/bin/env python3
"""
Enhanced radar test with improved data parsing and raw data inspection
"""

import serial
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
import threading
import struct

class AWR1843DataInspector:
    def __init__(self):
        self.cli_port: Optional[serial.Serial] = None
        self.data_port: Optional[serial.Serial] = None
        self.is_streaming = False
        
    def connect_radar(self) -> bool:
        """Connect to the AWR1843BOOST radar"""
        try:
            print("Connecting to AWR1843BOOST...")
            
            # Open CLI port (configuration)
            self.cli_port = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
            print("✅ CLI port (/dev/ttyACM0) opened successfully")
            
            # Open Data port (high-speed data)
            self.data_port = serial.Serial('/dev/ttyACM1', 921600, timeout=2)
            print("✅ Data port (/dev/ttyACM1) opened successfully")
            
            # Clear any existing data
            self.cli_port.reset_input_buffer()
            self.cli_port.reset_output_buffer()
            self.data_port.reset_input_buffer()
            self.data_port.reset_output_buffer()
            
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to radar: {e}")
            return False

    def send_command(self, command: str, wait_time: float = 0.5) -> Tuple[bool, str]:
        """Send command to radar and get response"""
        try:
            # Clear input buffer
            self.cli_port.reset_input_buffer()
            
            # Send command
            cmd_with_newline = command + '\n'
            self.cli_port.write(cmd_with_newline.encode())
            
            # Wait for processing
            time.sleep(wait_time)
            
            # Read response
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2.0:  # 2 second timeout
                if self.cli_port.in_waiting > 0:
                    try:
                        response += self.cli_port.read(self.cli_port.in_waiting).decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        pass
                    time.sleep(0.1)
                else:
                    break
            
            success = "Done" in response or len(response) > 0
            return success, response.strip()
            
        except Exception as e:
            return False, str(e)

    def configure_radar_minimal(self) -> bool:
        """Configure the radar with minimal working setup"""
        
        config_commands = [
            "sensorStop",
            "flushCfg", 
            "dfeDataOutputMode 1",
            "channelCfg 15 5 0",
            "adcCfg 2 1",
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 77 429 7 57.14 0 0 70 1 256 5209 0 0 30",
            "chirpCfg 0 0 0 0 0 0 0 1",
            "chirpCfg 1 1 0 0 0 0 0 4",
            "frameCfg 0 1 16 0 250 1 0",  # 250ms frame period
            "guiMonitor -1 1 1 0 0 0 1",  # Enable basic monitoring
            "cfarCfg -1 0 2 8 4 3 0 15 1",
            "cfarCfg -1 1 0 4 2 3 1 15 1",
            "sensorStart"
        ]
        
        print("🔧 Configuring radar with minimal setup...")
        success_count = 0
        
        for i, command in enumerate(config_commands, 1):
            print(f"[{i:2d}/{len(config_commands)}] {command}")
            
            wait_time = 1.0 if command in ['sensorStart', 'sensorStop', 'flushCfg'] else 0.5
            success, response = self.send_command(command, wait_time)
            
            if success or "Done" in response:
                print(f"    ✅ OK")
                success_count += 1
            else:
                print(f"    ⚠️  {response}")
                if command == 'sensorStart':
                    success_count += 1
            
            if command == 'sensorStart':
                time.sleep(2)
        
        print(f"📊 {success_count}/{len(config_commands)} commands successful")
        return success_count >= len(config_commands) - 1

    def inspect_raw_data(self, duration: int = 15):
        """Inspect raw data coming from the radar"""
        print(f"\n🔍 Inspecting raw radar data for {duration} seconds...")
        
        total_bytes = 0
        data_chunks = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                if self.data_port.in_waiting > 0:
                    # Read available data
                    raw_data = self.data_port.read(self.data_port.in_waiting)
                    total_bytes += len(raw_data)
                    
                    if len(raw_data) > 0:
                        data_chunks.append(raw_data)
                        print(f"📦 Received {len(raw_data)} bytes (Total: {total_bytes})")
                        
                        # Show first few bytes in hex
                        hex_preview = ' '.join([f'{b:02X}' for b in raw_data[:16]])
                        print(f"   First 16 bytes: {hex_preview}")
                        
                        # Look for potential magic words
                        self.search_magic_patterns(raw_data)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Data inspection stopped by user")
        
        print(f"\n📊 Raw Data Summary:")
        print(f"   Duration: {time.time() - start_time:.1f} seconds")
        print(f"   Total bytes received: {total_bytes}")
        print(f"   Data chunks: {len(data_chunks)}")
        print(f"   Average data rate: {total_bytes / (time.time() - start_time):.1f} bytes/sec")
        
        # Analyze all received data together
        if data_chunks:
            all_data = b''.join(data_chunks)
            self.analyze_data_patterns(all_data)

    def search_magic_patterns(self, data: bytes):
        """Search for known magic patterns in the data"""
        # Standard TI radar magic word (little endian)
        magic_patterns = [
            (b'\x01\x02\x03\x04\x05\x06\x07\x08', "TI Standard Magic"),
            (b'\x08\x07\x06\x05\x04\x03\x02\x01', "TI Reversed Magic"),
            (b'\x02\x01\x04\x03\x06\x05\x08\x07', "TI Alternative Magic"),
        ]
        
        for pattern, name in magic_patterns:
            idx = data.find(pattern)
            if idx != -1:
                print(f"   🎯 Found {name} at offset {idx}")

    def analyze_data_patterns(self, data: bytes):
        """Analyze patterns in the received data"""
        print(f"\n🔬 Analyzing {len(data)} bytes of radar data...")
        
        if len(data) < 10:
            print("   Not enough data to analyze")
            return
        
        # Check for repeating patterns
        print(f"   First 32 bytes: {' '.join([f'{b:02X}' for b in data[:32]])}")
        print(f"   Last 32 bytes:  {' '.join([f'{b:02X}' for b in data[-32:]])}")
        
        # Look for frame boundaries
        unique_bytes = len(set(data))
        print(f"   Unique byte values: {unique_bytes}/256")
        
        # Check if data looks like structured frames
        if len(data) >= 40:
            # Try to find frame headers by looking for consistent patterns
            potential_headers = []
            for i in range(len(data) - 40):
                if data[i:i+8] == b'\x01\x02\x03\x04\x05\x06\x07\x08':
                    potential_headers.append(i)
            
            if potential_headers:
                print(f"   🎯 Found {len(potential_headers)} potential frame headers")
                for i, pos in enumerate(potential_headers[:3]):  # Show first 3
                    print(f"   Header {i+1} at position {pos}")
                    self.try_parse_frame_at_position(data, pos)

    def try_parse_frame_at_position(self, data: bytes, pos: int):
        """Try to parse a frame header at the given position"""
        if pos + 40 > len(data):
            return
            
        try:
            header_data = data[pos:pos+40]
            
            # Parse header fields
            magic = struct.unpack('<Q', header_data[0:8])[0]
            version = struct.unpack('<I', header_data[8:12])[0]
            packet_len = struct.unpack('<I', header_data[12:16])[0]
            platform = struct.unpack('<I', header_data[16:20])[0]
            frame_num = struct.unpack('<I', header_data[20:24])[0]
            time_stamp = struct.unpack('<I', header_data[24:28])[0]
            num_objects = struct.unpack('<I', header_data[28:32])[0]
            num_tlvs = struct.unpack('<I', header_data[32:36])[0]
            
            print(f"      Magic: 0x{magic:016X}")
            print(f"      Frame: {frame_num}, Objects: {num_objects}, TLVs: {num_tlvs}")
            print(f"      Packet length: {packet_len}")
            
        except struct.error as e:
            print(f"      Parse error: {e}")

    def stop_radar(self):
        """Stop the radar sensor"""
        if self.cli_port:
            print("\n🛑 Stopping radar...")
            self.send_command("sensorStop", 1.0)

    def disconnect(self):
        """Close all connections"""
        if self.cli_port:
            self.cli_port.close()
        if self.data_port:
            self.data_port.close()

def main():
    radar = AWR1843DataInspector()
    
    try:
        # Connect
        if not radar.connect_radar():
            return
        
        # Configure
        if not radar.configure_radar_minimal():
            print("❌ Configuration failed")
            return
        
        print("\n✅ Radar configured and started!")
        
        # Inspect data
        radar.inspect_raw_data(15)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    
    finally:
        radar.stop_radar()
        radar.disconnect()
        print("Test completed.")

if __name__ == "__main__":
    main()
