#!/usr/bin/env python3
"""
Working AWR1843BOOST radar interface for SafeSpeed AI sensor fusion
Successfully parses radar frames and detects objects with position and velocity
"""

import serial
import time
import numpy as np
import struct
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RadarObject:
    """Detected radar object with position and velocity"""
    x: float          # meters
    y: float          # meters  
    z: float          # meters
    velocity: float   # m/s
    range_m: float    # range in meters
    azimuth_deg: float # azimuth angle in degrees
    
    @property
    def speed_kmh(self) -> float:
        """Get speed in km/h"""
        return abs(self.velocity) * 3.6

class AWR1843Interface:
    """Interface for AWR1843BOOST mmWave radar sensor"""
    
    def __init__(self):
        self.cli_port: Optional[serial.Serial] = None
        self.data_port: Optional[serial.Serial] = None
        self.is_streaming = False
        self.frame_callback = None
        
    def connect(self) -> bool:
        """Connect to the AWR1843BOOST radar"""
        try:
            print("🔗 Connecting to AWR1843BOOST radar...")
            
            # CLI port for configuration commands
            self.cli_port = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
            
            # Data port for high-speed radar data
            self.data_port = serial.Serial('/dev/ttyACM1', 921600, timeout=2)
            
            # Clear buffers
            self.cli_port.reset_input_buffer()
            self.cli_port.reset_output_buffer()
            self.data_port.reset_input_buffer()
            self.data_port.reset_output_buffer()
            
            time.sleep(1)
            print("✅ Radar connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to radar: {e}")
            return False
    
    def configure(self) -> bool:
        """Configure the radar with working parameters"""
        
        config_commands = [
            "sensorStop",
            "flushCfg",
            "dfeDataOutputMode 1",
            "channelCfg 15 5 0",                     # 4 RX, 2 TX antennas
            "adcCfg 2 1", 
            "adcbufCfg -1 0 1 1 1",
            "profileCfg 0 77 429 7 57.14 0 0 70 1 256 5209 0 0 30",  # 77-81 GHz profile
            "chirpCfg 0 0 0 0 0 0 0 1",
            "chirpCfg 1 1 0 0 0 0 0 4", 
            "frameCfg 0 1 16 0 250 1 0",             # 250ms frame period, 4 Hz
            "lowPower 0 0",
            "guiMonitor -1 1 1 0 1 1 1",             # Enable object detection output
            "cfarCfg -1 0 2 8 4 3 0 15 1",           # Range CFAR
            "cfarCfg -1 1 0 4 2 3 1 15 1",           # Doppler CFAR
            "multiObjBeamForming -1 1 0.5",
            "clutterRemoval -1 0",
            "calibDcRangeSig -1 0 -5 8 256",
            "extendedMaxVelocity -1 0",
            "lvdsStreamCfg -1 0 0 0",
            "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
            "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
            "CQRxSatMonitor 0 3 5 121 0",
            "CQSigImgMonitor 0 127 4",
            "analogMonitor 0 0",
            "aoaFovCfg -1 -90 90 -90 90",            # Full field of view
            "cfarFovCfg -1 0 0 8.92",                # Range 0-9m
            "cfarFovCfg -1 1 -1 1.00",               # Velocity ±1 m/s
            "calibData 0 0 0",
            "sensorStart"
        ]
        
        print("🔧 Configuring radar...")
        
        for cmd in config_commands:
            if not self._send_command(cmd):
                print(f"❌ Failed to send command: {cmd}")
                return False
            
            if cmd == "sensorStart":
                print("   ⏳ Starting sensor...")
                time.sleep(3)
        
        print("✅ Radar configured and started")
        self.is_streaming = True
        return True
    
    def _send_command(self, command: str) -> bool:
        """Send command to radar"""
        try:
            self.cli_port.write((command + '\n').encode())
            time.sleep(0.05)
            return True
        except Exception as e:
            print(f"Error sending {command}: {e}")
            return False
    
    def read_frame(self, timeout: float = 1.0) -> Optional[List[RadarObject]]:
        """Read one radar frame and return detected objects"""
        if not self.is_streaming:
            return None
        
        buffer_size = 32768
        byte_buffer = np.zeros(buffer_size, dtype='uint8')
        buffer_length = 0
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                bytes_available = self.data_port.in_waiting
                
                if bytes_available > 0:
                    read_data = self.data_port.read(bytes_available)
                    
                    # Add to buffer (prevent overflow)
                    for byte in read_data:
                        if buffer_length < buffer_size - 1:
                            byte_buffer[buffer_length] = byte
                            buffer_length += 1
                    
                    # Look for magic word: 02 01 04 03 06 05 08 07
                    frame_start = self._find_magic_word(byte_buffer, buffer_length)
                    
                    if frame_start >= 0:
                        objects = self._parse_frame(byte_buffer, frame_start, buffer_length)
                        if objects is not None:
                            return objects
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error reading frame: {e}")
        
        return None
    
    def _find_magic_word(self, buffer: np.ndarray, length: int) -> int:
        """Find magic word in buffer and return index"""
        for i in range(length - 8):
            if (buffer[i] == 0x02 and buffer[i+1] == 0x01 and 
                buffer[i+2] == 0x04 and buffer[i+3] == 0x03 and
                buffer[i+4] == 0x06 and buffer[i+5] == 0x05 and
                buffer[i+6] == 0x08 and buffer[i+7] == 0x07):
                return i
        return -1
    
    def _parse_frame(self, buffer: np.ndarray, start_idx: int, buffer_length: int) -> Optional[List[RadarObject]]:
        """Parse radar frame and extract objects"""
        try:
            # Need at least 40 bytes for header
            if start_idx + 40 > buffer_length:
                return None
            
            # Parse frame header
            header_bytes = buffer[start_idx:start_idx+40]
            
            frame_number = int.from_bytes(header_bytes[20:24].tobytes(), 'little')
            num_objects = int.from_bytes(header_bytes[28:32].tobytes(), 'little') 
            num_tlvs = int.from_bytes(header_bytes[32:36].tobytes(), 'little')
            packet_length = int.from_bytes(header_bytes[12:16].tobytes(), 'little')
            
            if num_objects == 0:
                return []
            
            # Parse TLV data to find detected objects
            objects = []
            tlv_idx = start_idx + 40
            
            for _ in range(min(num_tlvs, 10)):
                if tlv_idx + 8 > start_idx + packet_length:
                    break
                
                tlv_type = int.from_bytes(buffer[tlv_idx:tlv_idx+4].tobytes(), 'little')
                tlv_length = int.from_bytes(buffer[tlv_idx+4:tlv_idx+8].tobytes(), 'little')
                
                if tlv_type == 1 and tlv_length > 0:  # Detected objects TLV
                    objects = self._parse_objects_tlv(buffer, tlv_idx + 8, tlv_length)
                    break
                
                tlv_idx += 8 + tlv_length
            
            return objects
            
        except Exception as e:
            print(f"Frame parse error: {e}")
            return None
    
    def _parse_objects_tlv(self, buffer: np.ndarray, start_idx: int, length: int) -> List[RadarObject]:
        """Parse detected objects from TLV data"""
        objects = []
        
        try:
            # Each object is 16 bytes: x, y, z, velocity (float32 each)
            num_objects = length // 16
            
            for i in range(num_objects):
                obj_start = start_idx + i * 16
                if obj_start + 16 <= start_idx + length:
                    x = struct.unpack('<f', buffer[obj_start:obj_start+4].tobytes())[0]
                    y = struct.unpack('<f', buffer[obj_start+4:obj_start+8].tobytes())[0]
                    z = struct.unpack('<f', buffer[obj_start+8:obj_start+12].tobytes())[0]
                    velocity = struct.unpack('<f', buffer[obj_start+12:obj_start+16].tobytes())[0]
                    
                    # Calculate range and azimuth
                    range_m = np.sqrt(x*x + y*y + z*z)
                    azimuth_deg = np.arctan2(x, y) * 180.0 / np.pi
                    
                    obj = RadarObject(
                        x=x, y=y, z=z, velocity=velocity,
                        range_m=range_m, azimuth_deg=azimuth_deg
                    )
                    objects.append(obj)
            
        except Exception as e:
            print(f"Object parse error: {e}")
        
        return objects
    
    def stop(self):
        """Stop the radar sensor"""
        if self.cli_port and self.is_streaming:
            self._send_command("sensorStop")
            self.is_streaming = False
            print("🛑 Radar stopped")
    
    def disconnect(self):
        """Disconnect from radar"""
        self.stop()
        
        if self.cli_port:
            self.cli_port.close()
            self.cli_port = None
            
        if self.data_port:
            self.data_port.close()
            self.data_port = None
        
        print("📴 Radar disconnected")

def test_radar_interface():
    """Test the radar interface"""
    radar = AWR1843Interface()
    
    try:
        # Connect and configure
        if not radar.connect():
            return
        
        if not radar.configure():
            return
        
        print("\n📡 Reading radar data for 15 seconds...")
        start_time = time.time()
        frame_count = 0
        total_objects = 0
        
        while time.time() - start_time < 15:
            objects = radar.read_frame(timeout=0.5)
            
            if objects is not None:
                frame_count += 1
                total_objects += len(objects)
                
                if len(objects) > 0:
                    print(f"\n📊 Frame {frame_count}: {len(objects)} objects detected")
                    for i, obj in enumerate(objects[:5]):  # Show first 5
                        print(f"   Object {i+1}: ({obj.x:.2f}, {obj.y:.2f})m, "
                              f"Range: {obj.range_m:.2f}m, Speed: {obj.speed_kmh:.1f} km/h, "
                              f"Azimuth: {obj.azimuth_deg:.1f}°")
        
        elapsed = time.time() - start_time
        print(f"\n📈 Test Summary:")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total objects: {total_objects}")
        print(f"   Frame rate: {frame_count / elapsed:.1f} Hz")
        print(f"   Objects per frame: {total_objects / max(frame_count, 1):.1f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        radar.disconnect()
        print("✅ Test completed")

if __name__ == "__main__":
    test_radar_interface()
