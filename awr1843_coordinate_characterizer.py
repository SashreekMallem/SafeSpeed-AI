#!/usr/bin/env python3

"""
AWR1843 Coordinate System Characterization Tool
Based on research guide recommendations for understanding sensor behavior
"""

import sys
sys.path.append('/home/projecta/SafespeedAI')

from awr1843_interface import AWR1843Interface
import time
import numpy as np
import json
from datetime import datetime

class AWR1843Characterizer:
    """Tool to characterize AWR1843 coordinate system behavior"""
    
    def __init__(self):
        self.radar = AWR1843Interface()
        self.measurements = []
        
    def characterize_coordinate_system(self, duration=60):
        """Characterize coordinate system by collecting data over time"""
        print("🔬 AWR1843BOOST Coordinate System Characterization")
        print("=" * 60)
        print("This tool documents actual radar coordinate behavior")
        print("Move objects in front of radar during measurement period")
        print(f"Duration: {duration} seconds")
        print("=" * 60)
        
        if not self.radar.connect():
            print("❌ Failed to connect to radar")
            return
            
        if not self.radar.configure():
            print("❌ Failed to configure radar")
            return
            
        print("✅ Radar connected and configured")
        print("\n📊 Coordinate System Analysis:")
        print("   X-axis: Lateral position (left/right)")
        print("   Y-axis: Longitudinal position (distance)")
        print("   Velocity: Radial velocity (toward/away)")
        print("\n🎯 Test Procedure:")
        print("   1. Stand directly in front of radar (should see Y > 0, X ≈ 0)")
        print("   2. Move to radar's left (should see X < 0)")
        print("   3. Move to radar's right (should see X > 0)")
        print("   4. Walk toward radar (should see negative velocity)")
        print("   5. Walk away from radar (should see positive velocity)")
        print("   6. Move to different distances")
        
        start_time = time.time()
        frame_count = 0
        
        # Coordinate system statistics
        x_values = []
        y_values = []
        velocity_values = []
        range_values = []
        
        try:
            while time.time() - start_time < duration:
                objects = self.radar.read_frame(timeout=0.5)
                
                if objects and len(objects) > 0:
                    frame_count += 1
                    timestamp = time.time() - start_time
                    
                    print(f"\n⏱️  Frame {frame_count} (t={timestamp:.1f}s): {len(objects)} objects")
                    
                    for i, obj in enumerate(objects):
                        # Document the measurement
                        measurement = {
                            'timestamp': timestamp,
                            'frame': frame_count,
                            'object_id': i,
                            'x': obj.x,
                            'y': obj.y,
                            'z': obj.z,
                            'velocity': obj.velocity,
                            'range_m': obj.range_m,
                            'azimuth_deg': obj.azimuth_deg,
                            'speed_kmh': obj.speed_kmh
                        }
                        self.measurements.append(measurement)
                        
                        # Collect statistics
                        x_values.append(obj.x)
                        y_values.append(obj.y)
                        velocity_values.append(obj.velocity)
                        range_values.append(obj.range_m)
                        
                        print(f"   Object {i+1}: X={obj.x:6.2f}m, Y={obj.y:6.2f}m, Z={obj.z:6.2f}m")
                        print(f"            Range={obj.range_m:6.2f}m, Speed={obj.speed_kmh:6.1f}km/h")
                        print(f"            Azimuth={obj.azimuth_deg:6.1f}°, Velocity={obj.velocity:6.2f}m/s")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Characterization stopped by user")
        
        # Analysis
        elapsed = time.time() - start_time
        print(f"\n📊 Characterization Complete:")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Frames: {frame_count}")
        print(f"   Total measurements: {len(self.measurements)}")
        
        if len(self.measurements) > 0:
            self.analyze_measurements(x_values, y_values, velocity_values, range_values)
            self.save_measurements()
        
        self.radar.disconnect()
    
    def analyze_measurements(self, x_values, y_values, velocity_values, range_values):
        """Analyze collected measurements"""
        print(f"\n🔍 Coordinate System Analysis:")
        
        if x_values:
            print(f"   X-axis (Lateral):")
            print(f"      Range: {min(x_values):.2f}m to {max(x_values):.2f}m")
            print(f"      Mean: {np.mean(x_values):.2f}m, Std: {np.std(x_values):.2f}m")
            
        if y_values:
            print(f"   Y-axis (Longitudinal):")
            print(f"      Range: {min(y_values):.2f}m to {max(y_values):.2f}m")
            print(f"      Mean: {np.mean(y_values):.2f}m, Std: {np.std(y_values):.2f}m")
            
        if velocity_values:
            print(f"   Velocity (Radial):")
            print(f"      Range: {min(velocity_values):.2f}m/s to {max(velocity_values):.2f}m/s")
            print(f"      Mean: {np.mean(velocity_values):.2f}m/s, Std: {np.std(velocity_values):.2f}m/s")
            
        if range_values:
            print(f"   Range (Distance):")
            print(f"      Range: {min(range_values):.2f}m to {max(range_values):.2f}m")
            print(f"      Mean: {np.mean(range_values):.2f}m, Std: {np.std(range_values):.2f}m")
            
        # Field of view analysis
        if x_values and y_values:
            angles = [np.arctan2(x, y) * 180 / np.pi for x, y in zip(x_values, y_values)]
            print(f"   Angular Coverage:")
            print(f"      Range: {min(angles):.1f}° to {max(angles):.1f}°")
            print(f"      Total FOV: {max(angles) - min(angles):.1f}°")
    
    def save_measurements(self):
        """Save measurements to file for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/projecta/SafespeedAI/awr1843_characterization_{timestamp}.json"
        
        analysis_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_measurements': len(self.measurements),
                'tool_version': '1.0',
                'radar_config': 'AWR1843BOOST_standard'
            },
            'measurements': self.measurements
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
            
        print(f"\n💾 Data saved to: {filename}")
        print("   Use this data for coordinate transformation calibration")
    
    def quick_test(self):
        """Quick test to verify basic radar operation"""
        print("🔬 Quick AWR1843 Operation Test")
        print("=" * 40)
        
        if not self.radar.connect():
            print("❌ Failed to connect to radar")
            return
            
        if not self.radar.configure():
            print("❌ Failed to configure radar")
            return
            
        print("✅ Radar operational")
        print("📡 Reading 10 frames...")
        
        for i in range(10):
            objects = self.radar.read_frame(timeout=1.0)
            
            if objects:
                print(f"Frame {i+1}: {len(objects)} objects detected")
                for j, obj in enumerate(objects[:3]):  # Show first 3
                    print(f"   Obj{j+1}: ({obj.x:.2f}, {obj.y:.2f}m) {obj.speed_kmh:.1f}km/h")
            else:
                print(f"Frame {i+1}: No objects")
                
            time.sleep(0.5)
        
        self.radar.disconnect()
        print("✅ Quick test complete")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AWR1843 Coordinate System Characterizer')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Test mode: quick (10 frames) or full (60 seconds)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for full characterization (seconds)')
    
    args = parser.parse_args()
    
    characterizer = AWR1843Characterizer()
    
    if args.mode == 'quick':
        characterizer.quick_test()
    else:
        characterizer.characterize_coordinate_system(args.duration)


if __name__ == "__main__":
    main()
