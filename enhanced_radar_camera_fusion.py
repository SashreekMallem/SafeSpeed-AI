#!/usr/bin/env python3

"""
Enhanced Radar-Camera Fusion with Multiple Display Options
This script provides both console output and visual feedback for radar-camera fusion
"""

import cv2
import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import sys
import os
import argparse

# Add the project directory to Python path
sys.path.append('/home/projecta/SafespeedAI')

# Import our radar interface
from awr1843_interface import AWR1843Interface

@dataclass
class RadarObject:
    """Represents a detected radar object"""
    x: float  # meters
    y: float  # meters
    range_rate: float  # m/s (velocity towards/away from radar)
    range: float  # meters
    angle: float  # radians
    snr: float  # signal-to-noise ratio
    timestamp: float

@dataclass
class CameraObject:
    """Represents a detected camera object"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    class_name: str
    timestamp: float

@dataclass
class FusedObject:
    """Represents a fused radar-camera object"""
    radar_data: Optional[RadarObject]
    camera_data: Optional[CameraObject]
    estimated_speed: Optional[float]  # km/h
    distance: Optional[float]  # meters
    confidence: float
    timestamp: float

class EnhancedRadarCameraFusion:
    """Enhanced radar-camera fusion with multiple display options"""
    
    def __init__(self, camera_id=0, display_mode="console"):
        self.camera_id = camera_id
        self.display_mode = display_mode  # "console", "gui", "both"
        self.radar_interface = None
        self.camera = None
        
        # Data queues
        self.radar_queue = queue.Queue(maxsize=100)
        self.camera_queue = queue.Queue(maxsize=30)
        self.fusion_queue = queue.Queue(maxsize=50)
        
        # Threading control
        self.running = False
        self.radar_thread = None
        self.camera_thread = None
        self.fusion_thread = None
        self.display_thread = None
        
        # Fusion parameters
        self.max_fusion_distance = 200.0  # Maximum distance for associating objects (pixels)
        self.max_time_diff = 0.1  # Maximum time difference for fusion (seconds)
        
        # Simple vehicle detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Display variables
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection counters and statistics
        self.radar_count = 0
        self.camera_count = 0
        self.fusion_count = 0
        self.total_detections = 0
        self.last_display_time = time.time()
        
        # Current objects for display
        self.current_radar_objects = []
        self.current_camera_objects = []
        self.current_fused_objects = []
        
        # Colors for display
        self.colors = {
            'radar': (0, 255, 255),      # Yellow
            'camera': (0, 255, 0),       # Green  
            'fusion': (255, 0, 255),     # Magenta
            'text': (255, 255, 255),     # White
            'background': (0, 0, 0)      # Black
        }
        
    def initialize_radar(self):
        """Initialize radar connection"""
        try:
            print("🔗 Initializing AWR1843BOOST radar...")
            self.radar_interface = AWR1843Interface()
            
            if self.radar_interface.connect():
                print("✅ Radar connected successfully")
                if self.radar_interface.configure():
                    print("✅ Radar configured successfully")
                    return True
                else:
                    print("❌ Failed to configure radar")
                    return False
            else:
                print("❌ Failed to connect to radar")
                return False
        except Exception as e:
            print(f"❌ Radar initialization error: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera connection"""
        try:
            print(f"📷 Initializing camera {self.camera_id}...")
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print("❌ Failed to open camera")
                return False
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("✅ Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def radar_data_thread(self):
        """Thread for collecting radar data"""
        print("🎯 Starting radar data collection...")
        
        while self.running:
            try:
                if self.radar_interface and self.radar_interface.cli_port and self.radar_interface.data_port:
                    # Get radar frame data
                    frame_objects = self.radar_interface.read_frame(timeout=0.1)
                    
                    if frame_objects:
                        timestamp = time.time()
                        self.radar_count = len(frame_objects)
                        radar_objects = []
                        
                        for obj in frame_objects:
                            radar_obj = RadarObject(
                                x=obj.x,
                                y=obj.y,
                                range_rate=obj.velocity,
                                range=obj.range_m,
                                angle=obj.azimuth_deg * np.pi / 180,
                                snr=0.8,
                                timestamp=timestamp
                            )
                            radar_objects.append(radar_obj)
                            
                            # Add to queue
                            if not self.radar_queue.full():
                                self.radar_queue.put(radar_obj)
                            else:
                                try:
                                    self.radar_queue.get_nowait()
                                    self.radar_queue.put(radar_obj)
                                except queue.Empty:
                                    pass
                        
                        # Store for display
                        self.current_radar_objects = radar_objects
                    else:
                        self.radar_count = 0
                        self.current_radar_objects = []
                
                time.sleep(0.05)  # 20 Hz radar data collection
                
            except Exception as e:
                print(f"❌ Radar data thread error: {e}")
                time.sleep(1)
    
    def detect_vehicles_simple(self, frame):
        """Simple vehicle detection using background subtraction"""
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicles = []
            timestamp = time.time()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (vehicles should be reasonably large)
                if area > 1500:  # Minimum area for a vehicle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio and size
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 8.0 and w > 30 and h > 20:
                        camera_obj = CameraObject(
                            bbox=(x, y, w, h),
                            confidence=min(0.9, area / 20000),
                            class_name="vehicle",
                            timestamp=timestamp
                        )
                        vehicles.append(camera_obj)
            
            return vehicles
            
        except Exception as e:
            print(f"❌ Vehicle detection error: {e}")
            return []
    
    def camera_data_thread(self):
        """Thread for collecting camera data"""
        print("📹 Starting camera data collection...")
        
        while self.running:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    
                    if ret:
                        # Store current frame for display
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        # Simple vehicle detection
                        vehicles = self.detect_vehicles_simple(frame)
                        self.camera_count = len(vehicles)
                        self.current_camera_objects = vehicles
                        
                        # Add detected vehicles to queue
                        for vehicle in vehicles:
                            if not self.camera_queue.full():
                                self.camera_queue.put(vehicle)
                            else:
                                try:
                                    self.camera_queue.get_nowait()
                                    self.camera_queue.put(vehicle)
                                except queue.Empty:
                                    pass
                
                time.sleep(1/30)  # 30 FPS camera data collection
                
            except Exception as e:
                print(f"❌ Camera data thread error: {e}")
                time.sleep(1)
    
    def associate_radar_camera(self, radar_objects: List[RadarObject], 
                              camera_objects: List[CameraObject]) -> List[FusedObject]:
        """Associate radar and camera objects"""
        fused_objects = []
        used_camera_objects = set()
        
        for radar_obj in radar_objects:
            best_match = None
            best_distance = float('inf')
            best_camera_idx = -1
            
            # Convert radar coordinates to image coordinates
            radar_x, radar_y = self.convert_radar_to_image_coords(radar_obj)
            
            for i, camera_obj in enumerate(camera_objects):
                if i in used_camera_objects:
                    continue
                
                # Get center of camera bounding box
                cam_x = camera_obj.bbox[0] + camera_obj.bbox[2] / 2
                cam_y = camera_obj.bbox[1] + camera_obj.bbox[3] / 2
                
                # Calculate distance between radar and camera object positions
                distance = np.sqrt((radar_x - cam_x)**2 + (radar_y - cam_y)**2)
                
                if distance < self.max_fusion_distance and distance < best_distance:
                    best_distance = distance
                    best_match = camera_obj
                    best_camera_idx = i
            
            # Create fused object
            if best_match:
                used_camera_objects.add(best_camera_idx)
                
                # Calculate estimated speed
                estimated_speed_ms = abs(radar_obj.range_rate)
                estimated_speed_kmh = estimated_speed_ms * 3.6
                
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=best_match,
                    estimated_speed=estimated_speed_kmh,
                    distance=radar_obj.range,
                    confidence=min(0.95, (radar_obj.snr + best_match.confidence) / 2),
                    timestamp=max(radar_obj.timestamp, best_match.timestamp)
                )
            else:
                # Radar-only object
                estimated_speed_ms = abs(radar_obj.range_rate)
                estimated_speed_kmh = estimated_speed_ms * 3.6
                
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=None,
                    estimated_speed=estimated_speed_kmh,
                    distance=radar_obj.range,
                    confidence=radar_obj.snr,
                    timestamp=radar_obj.timestamp
                )
            
            fused_objects.append(fused_obj)
        
        # Add unmatched camera objects
        for i, camera_obj in enumerate(camera_objects):
            if i not in used_camera_objects:
                fused_obj = FusedObject(
                    radar_data=None,
                    camera_data=camera_obj,
                    estimated_speed=None,
                    distance=None,
                    confidence=camera_obj.confidence,
                    timestamp=camera_obj.timestamp
                )
                fused_objects.append(fused_obj)
        
        return fused_objects
    
    def fusion_thread(self):
        """Thread for sensor fusion processing"""
        print("🔗 Starting sensor fusion...")
        
        while self.running:
            try:
                # Collect recent data
                radar_objects = []
                camera_objects = []
                current_time = time.time()
                
                # Get recent radar objects
                temp_radar = []
                while not self.radar_queue.empty():
                    try:
                        obj = self.radar_queue.get_nowait()
                        temp_radar.append(obj)
                        if current_time - obj.timestamp < 0.2:
                            radar_objects.append(obj)
                    except queue.Empty:
                        break
                
                # Put back recent objects
                for obj in temp_radar[-10:]:  # Keep last 10
                    if not self.radar_queue.full():
                        self.radar_queue.put(obj)
                
                # Get recent camera objects
                temp_camera = []
                while not self.camera_queue.empty():
                    try:
                        obj = self.camera_queue.get_nowait()
                        temp_camera.append(obj)
                        if current_time - obj.timestamp < 0.2:
                            camera_objects.append(obj)
                    except queue.Empty:
                        break
                
                # Put back recent objects
                for obj in temp_camera[-5:]:  # Keep last 5
                    if not self.camera_queue.full():
                        self.camera_queue.put(obj)
                
                # Perform fusion
                if radar_objects or camera_objects:
                    fused_objects = self.associate_radar_camera(radar_objects, camera_objects)
                    self.fusion_count = len(fused_objects)
                    self.current_fused_objects = fused_objects
                    
                    # Add to output queue
                    for fused_obj in fused_objects:
                        if not self.fusion_queue.full():
                            self.fusion_queue.put(fused_obj)
                        else:
                            try:
                                self.fusion_queue.get_nowait()
                                self.fusion_queue.put(fused_obj)
                            except queue.Empty:
                                pass
                else:
                    self.fusion_count = 0
                    self.current_fused_objects = []
                
                time.sleep(0.05)  # 20 Hz fusion processing
                
            except Exception as e:
                print(f"❌ Fusion thread error: {e}")
                time.sleep(1)
    
    def convert_radar_to_image_coords(self, radar_obj: RadarObject, img_width=1280, img_height=720) -> Tuple[int, int]:
        """Convert radar coordinates to image pixel coordinates"""
        # Scale factors (adjust based on radar range and camera field of view)
        scale_x = 40  # pixels per meter lateral
        scale_y = 15  # pixels per meter longitudinal
        
        # Convert to image coordinates (radar origin at image center)
        img_x = int(img_width / 2 + radar_obj.x * scale_x)
        img_y = int(img_height / 2 - radar_obj.y * scale_y)
        
        # Clamp to image bounds
        img_x = max(0, min(img_width - 1, img_x))
        img_y = max(0, min(img_height - 1, img_y))
        
        return img_x, img_y
    
    def display_console_status(self):
        """Display detailed console status"""
        current_time = time.time()
        if current_time - self.last_display_time < 1.0:  # Update every 1 second
            return
            
        self.last_display_time = current_time
        
        # Clear screen and show status
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🚗" * 25)
        print("🚗  SafeSpeed AI - RADAR-CAMERA FUSION STATUS  🚗")
        print("🚗" * 25)
        print()
        
        # System status
        print(f"⏰ Time: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f"🔴 Status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}")
        print()
        
        # Detection counts
        print("📊 DETECTION SUMMARY:")
        print(f"   🎯 Radar Objects:  {self.radar_count:3d}")
        print(f"   📷 Camera Objects: {self.camera_count:3d}")
        print(f"   🔗 Fused Objects:  {self.fusion_count:3d}")
        print()
        
        # Detailed object information
        if self.current_radar_objects or self.current_camera_objects or self.current_fused_objects:
            print("🎯 DETAILED OBJECT ANALYSIS:")
            print("-" * 75)
            print(f"{'ID':<3} {'TYPE':<8} {'SPEED':<12} {'DISTANCE':<10} {'POSITION':<15} {'SNR':<8}")
            print("-" * 75)
            
            obj_id = 1
            
            # Show radar objects
            for obj in self.current_radar_objects[:10]:
                speed_kmh = abs(obj.range_rate) * 3.6
                position = f"({obj.x:.1f},{obj.y:.1f})"
                print(f"{obj_id:<3} {'RADAR':<8} {speed_kmh:<12.1f} {obj.range:<10.1f} {position:<15} {obj.snr:<8.2f}")
                obj_id += 1
            
            # Show camera objects
            for obj in self.current_camera_objects[:5]:
                bbox_center = f"({obj.bbox[0]+obj.bbox[2]//2},{obj.bbox[1]+obj.bbox[3]//2})"
                print(f"{obj_id:<3} {'CAMERA':<8} {'N/A':<12} {'N/A':<10} {bbox_center:<15} {obj.confidence:<8.2f}")
                obj_id += 1
            
            # Show fused objects
            for obj in self.current_fused_objects[:5]:
                obj_type = "FUSED" if (obj.radar_data and obj.camera_data) else \
                          "RADAR" if obj.radar_data else "CAMERA"
                
                speed = f"{obj.estimated_speed:.1f}" if obj.estimated_speed else "N/A"
                distance = f"{obj.distance:.1f}" if obj.distance else "N/A"
                
                if obj.radar_data:
                    position = f"({obj.radar_data.x:.1f},{obj.radar_data.y:.1f})"
                elif obj.camera_data:
                    bbox_center = f"({obj.camera_data.bbox[0]+obj.camera_data.bbox[2]//2},{obj.camera_data.bbox[1]+obj.camera_data.bbox[3]//2})"
                    position = f"IMG{bbox_center}"
                else:
                    position = "N/A"
                
                print(f"{obj_id:<3} {obj_type:<8} {speed:<12} {distance:<10} {position:<15} {obj.confidence:<8.2f}")
                obj_id += 1
                
            total_objects = len(self.current_radar_objects) + len(self.current_camera_objects) + len(self.current_fused_objects)
            if total_objects > 20:
                print(f"... and {total_objects - 20} more objects")
        else:
            print("ℹ️  No objects currently detected")
        
        print()
        print("🎮 CONTROLS:")
        print("   • Press Ctrl+C to stop")
        if self.display_mode in ["gui", "both"]:
            print("   • Press 'q' in GUI window to quit")
        print()
    
    def display_thread_func(self):
        """Thread for handling display output"""
        print(f"🎬 Starting display in {self.display_mode} mode...")
        
        while self.running:
            try:
                if self.display_mode in ["console", "both"]:
                    self.display_console_status()
                
                if self.display_mode in ["gui", "both"] and self.current_frame is not None:
                    try:
                        with self.frame_lock:
                            display_frame = self.current_frame.copy()
                        
                        # Draw all objects on frame
                        self.draw_objects_on_frame(display_frame)
                        
                        # Show the frame
                        cv2.imshow('SafeSpeed AI - Radar-Camera Fusion', display_frame)
                        
                        # Check for quit key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("🛑 GUI quit requested")
                            self.running = False
                            break
                    except cv2.error as e:
                        if self.display_mode == "both":
                            print(f"⚠️  GUI display failed, continuing in console mode: {e}")
                            self.display_mode = "console"
                        else:
                            print(f"❌ GUI display error: {e}")
                            time.sleep(1)
                
                time.sleep(0.1)  # 10 Hz display update
                
            except Exception as e:
                print(f"❌ Display thread error: {e}")
                time.sleep(1)
        
        if self.display_mode in ["gui", "both"]:
            cv2.destroyAllWindows()
    
    def draw_objects_on_frame(self, frame):
        """Draw all detected objects on the frame"""
        # Draw camera objects (green boxes)
        for obj in self.current_camera_objects:
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['camera'], 2)
            cv2.putText(frame, f"Vehicle: {obj.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['camera'], 2)
        
        # Draw radar objects (yellow circles)
        for obj in self.current_radar_objects:
            img_x, img_y = self.convert_radar_to_image_coords(obj)
            
            # Main detection point
            cv2.circle(frame, (img_x, img_y), 8, self.colors['radar'], -1)
            cv2.circle(frame, (img_x, img_y), 15, self.colors['radar'], 2)
            
            # Speed and distance info
            speed_kmh = abs(obj.range_rate) * 3.6
            cv2.putText(frame, f"{speed_kmh:.1f} km/h", (img_x + 20, img_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['radar'], 2)
            cv2.putText(frame, f"{obj.range:.1f}m", (img_x + 20, img_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['radar'], 2)
        
        # Draw fusion connections (magenta lines)
        for obj in self.current_fused_objects:
            if obj.radar_data and obj.camera_data:
                radar_x, radar_y = self.convert_radar_to_image_coords(obj.radar_data)
                cam_x = obj.camera_data.bbox[0] + obj.camera_data.bbox[2] // 2
                cam_y = obj.camera_data.bbox[1] + obj.camera_data.bbox[3] // 2
                
                # Draw connection line
                cv2.line(frame, (radar_x, radar_y), (cam_x, cam_y), self.colors['fusion'], 3)
                
                # Fusion info at midpoint
                mid_x = (radar_x + cam_x) // 2
                mid_y = (radar_y + cam_y) // 2
                cv2.putText(frame, f"FUSED: {obj.confidence:.2f}", (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['fusion'], 2)
        
        # Draw status overlay
        self.draw_status_overlay(frame)
    
    def draw_status_overlay(self, frame):
        """Draw system status overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 120), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Status text
        y_pos = 35
        cv2.putText(frame, "SafeSpeed AI - Radar-Camera Fusion", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        y_pos += 20
        cv2.putText(frame, f"Radar: {self.radar_count} | Camera: {self.camera_count} | Fused: {self.fusion_count}",
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        y_pos += 15
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        y_pos += 15
        cv2.putText(frame, "Press 'q' to quit", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def start(self):
        """Start the fusion system"""
        print("🚀 Starting Enhanced Radar-Camera Fusion System...")
        
        # Initialize systems
        radar_ok = self.initialize_radar()
        camera_ok = self.initialize_camera()
        
        if not radar_ok and not camera_ok:
            print("❌ Failed to initialize both radar and camera")
            return False
        
        if not radar_ok:
            print("⚠️  Running in camera-only mode")
        
        if not camera_ok:
            print("⚠️  Running in radar-only mode")
        
        # Start threads
        self.running = True
        
        if radar_ok:
            self.radar_thread = threading.Thread(target=self.radar_data_thread, daemon=True)
            self.radar_thread.start()
        
        if camera_ok:
            self.camera_thread = threading.Thread(target=self.camera_data_thread, daemon=True)
            self.camera_thread.start()
        
        self.fusion_thread = threading.Thread(target=self.fusion_thread, daemon=True)
        self.fusion_thread.start()
        
        self.display_thread = threading.Thread(target=self.display_thread_func, daemon=True)
        self.display_thread.start()
        
        print("✅ Enhanced fusion system started successfully")
        print(f"📺 Display mode: {self.display_mode}")
        
        if self.display_mode in ["gui", "both"]:
            print("🎬 GUI window should open shortly...")
        
        return True
    
    def stop(self):
        """Stop the fusion system"""
        print("🛑 Stopping fusion system...")
        
        self.running = False
        
        # Close camera
        if self.camera:
            self.camera.release()
        
        # Disconnect radar
        if self.radar_interface:
            self.radar_interface.disconnect()
        
        cv2.destroyAllWindows()
        print("✅ Fusion system stopped")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Radar-Camera Fusion System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--display', choices=['console', 'gui', 'both'], default='console',
                       help='Display mode (default: console)')
    
    args = parser.parse_args()
    
    print("🚗 SafeSpeed AI - Enhanced Radar-Camera Fusion")
    print("=" * 60)
    print(f"📷 Camera ID: {args.camera}")
    print(f"📺 Display Mode: {args.display}")
    print("=" * 60)
    
    fusion_system = EnhancedRadarCameraFusion(camera_id=args.camera, display_mode=args.display)
    
    try:
        if fusion_system.start():
            print("\n⏱️  System running... Press Ctrl+C to stop")
            
            # Keep main thread alive
            while fusion_system.running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        fusion_system.stop()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
