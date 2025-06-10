#!/usr/bin/env python3

"""
Simple Radar-Camera Fusion System
This script demonstrates basic sensor fusion between AWR1843BOOST radar and camera
without the complexity of DeepStream, for step-by-step testing.
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

class SimpleRadarCameraFusion:
    """Simple radar-camera fusion system for testing"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
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
        self.max_fusion_distance = 200.0  # Maximum distance for associating radar/camera objects (pixels)
        self.max_time_diff = 0.1  # Maximum time difference for fusion (seconds)
        
        # Simple vehicle detection (using basic CV for now)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Display variables
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.display_enabled = True
        
        # Detection counters for display
        self.radar_count = 0
        self.camera_count = 0
        self.fusion_count = 0
        
    def initialize_radar(self):
        """Initialize radar connection"""
        try:
            print("Initializing AWR1843BOOST radar...")
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
            print(f"Initializing camera {self.camera_id}...")
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
        print("🔄 Starting radar data collection...")
        
        while self.running:
            try:
                if self.radar_interface and self.radar_interface.cli_port and self.radar_interface.data_port:
                    # Get radar frame data
                    frame_objects = self.radar_interface.read_frame(timeout=0.1)
                    
                    if frame_objects:
                        timestamp = time.time()
                        self.radar_count = len(frame_objects)
                        
                        for obj in frame_objects:
                            radar_obj = RadarObject(
                                x=obj.x,
                                y=obj.y,
                                range_rate=obj.velocity,
                                range=obj.range_m,
                                angle=obj.azimuth_deg * np.pi / 180,  # Convert to radians
                                snr=0.8,  # Default SNR
                                timestamp=timestamp
                            )
                            
                            # Add to queue if not full
                            if not self.radar_queue.full():
                                self.radar_queue.put(radar_obj)
                            else:
                                # Remove oldest item to make room
                                try:
                                    self.radar_queue.get_nowait()
                                    self.radar_queue.put(radar_obj)
                                except queue.Empty:
                                    pass
                    else:
                        self.radar_count = 0
                
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicles = []
            timestamp = time.time()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (vehicles should be reasonably large)
                if area > 2000:  # Minimum area for a vehicle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (vehicles are wider than tall typically)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 5.0:
                        camera_obj = CameraObject(
                            bbox=(x, y, w, h),
                            confidence=min(0.9, area / 10000),  # Simple confidence based on size
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
        print("🔄 Starting camera data collection...")
        
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
                        
                        # Add detected vehicles to queue
                        for vehicle in vehicles:
                            if not self.camera_queue.full():
                                self.camera_queue.put(vehicle)
                            else:
                                # Remove oldest item to make room
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
            
            # Convert radar coordinates to image coordinates using the same function
            radar_image_x, radar_image_y = self.convert_radar_to_image_coords(radar_obj)
            
            for i, camera_obj in enumerate(camera_objects):
                if i in used_camera_objects:
                    continue
                
                # Get center of camera bounding box
                cam_x = camera_obj.bbox[0] + camera_obj.bbox[2] / 2
                cam_y = camera_obj.bbox[1] + camera_obj.bbox[3] / 2
                
                # Calculate distance between radar and camera object positions
                distance = np.sqrt((radar_image_x - cam_x)**2 + (radar_image_y - cam_y)**2)
                
                if distance < self.max_fusion_distance and distance < best_distance:
                    best_distance = distance
                    best_match = camera_obj
                    best_camera_idx = i
            
            # Create fused object
            if best_match:
                used_camera_objects.add(best_camera_idx)
                
                # Calculate estimated speed (radar gives radial velocity)
                estimated_speed_ms = abs(radar_obj.range_rate)  # m/s
                estimated_speed_kmh = estimated_speed_ms * 3.6  # km/h
                
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=best_match,
                    estimated_speed=estimated_speed_kmh,
                    distance=radar_obj.range,
                    confidence=min(0.9, (radar_obj.snr + best_match.confidence) / 2),
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
        print("🔄 Starting sensor fusion...")
        
        while self.running:
            try:
                # Collect recent radar and camera data
                radar_objects = []
                camera_objects = []
                current_time = time.time()
                
                # Get radar objects from the last 100ms
                while not self.radar_queue.empty():
                    try:
                        radar_obj = self.radar_queue.get_nowait()
                        if current_time - radar_obj.timestamp < 0.1:  # Within 100ms
                            radar_objects.append(radar_obj)
                    except queue.Empty:
                        break
                
                # Get camera objects from the last 100ms
                while not self.camera_queue.empty():
                    try:
                        camera_obj = self.camera_queue.get_nowait()
                        if current_time - camera_obj.timestamp < 0.1:  # Within 100ms
                            camera_objects.append(camera_obj)
                    except queue.Empty:
                        break
                
                # Perform fusion
                if radar_objects or camera_objects:
                    fused_objects = self.associate_radar_camera(radar_objects, camera_objects)
                    self.fusion_count = len(fused_objects)
                    
                    # Add fused objects to output queue
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
                
                time.sleep(0.05)  # 20 Hz fusion processing
                
            except Exception as e:
                print(f"❌ Fusion thread error: {e}")
                time.sleep(1)
    
    def convert_radar_to_image_coords(self, radar_obj: RadarObject, img_width=1280, img_height=720) -> Tuple[int, int]:
        """Convert radar coordinates to image pixel coordinates"""
        # Simple coordinate transformation
        # Radar: x=lateral, y=longitudinal (forward)
        # Camera: center at (640, 360)
        
        # Scale factors (adjust based on radar range and camera field of view)
        scale_x = 50  # pixels per meter lateral
        scale_y = 20  # pixels per meter longitudinal
        
        # Convert to image coordinates
        img_x = int(img_width / 2 + radar_obj.x * scale_x)
        img_y = int(img_height / 2 - radar_obj.y * scale_y)
        
        # Clamp to image bounds
        img_x = max(0, min(img_width - 1, img_x))
        img_y = max(0, min(img_height - 1, img_y))
        
        return img_x, img_y
    
    def draw_radar_objects(self, frame, radar_objects: List[RadarObject]):
        """Draw radar objects on the frame"""
        for obj in radar_objects:
            img_x, img_y = self.convert_radar_to_image_coords(obj)
            
            # Draw radar detection point
            cv2.circle(frame, (img_x, img_y), 8, (0, 255, 255), -1)  # Yellow circle
            cv2.circle(frame, (img_x, img_y), 12, (0, 255, 255), 2)   # Yellow border
            
            # Draw speed info
            speed_kmh = abs(obj.range_rate) * 3.6
            speed_text = f"{speed_kmh:.1f} km/h"
            cv2.putText(frame, speed_text, (img_x + 15, img_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw distance info
            dist_text = f"{obj.range:.1f}m"
            cv2.putText(frame, dist_text, (img_x + 15, img_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    def draw_camera_objects(self, frame, camera_objects: List[CameraObject]):
        """Draw camera detected objects on the frame"""
        for obj in camera_objects:
            x, y, w, h = obj.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            
            # Draw confidence
            conf_text = f"{obj.class_name}: {obj.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_fused_objects(self, frame, fused_objects: List[FusedObject]):
        """Draw fused objects with combined radar-camera info"""
        for obj in fused_objects:
            if obj.radar_data and obj.camera_data:
                # Draw connection line between radar point and camera bbox
                radar_x, radar_y = self.convert_radar_to_image_coords(obj.radar_data)
                cam_x = obj.camera_data.bbox[0] + obj.camera_data.bbox[2] // 2
                cam_y = obj.camera_data.bbox[1] + obj.camera_data.bbox[3] // 2
                
                # Draw fusion line
                cv2.line(frame, (radar_x, radar_y), (cam_x, cam_y), (255, 0, 255), 2)  # Magenta line
                
                # Draw fusion confidence
                mid_x = (radar_x + cam_x) // 2
                mid_y = (radar_y + cam_y) // 2
                fusion_text = f"Fusion: {obj.confidence:.2f}"
                cv2.putText(frame, fusion_text, (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    def draw_status_overlay(self, frame):
        """Draw system status overlay"""
        # Background for status text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        y_pos = 35
        cv2.putText(frame, "SafeSpeed AI - Radar-Camera Fusion", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 25
        cv2.putText(frame, f"Radar Objects: {self.radar_count}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_pos += 20
        cv2.putText(frame, f"Camera Objects: {self.camera_count}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_pos += 20
        cv2.putText(frame, f"Fused Objects: {self.fusion_count}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        y_pos += 25
        cv2.putText(frame, "Press 'q' to quit", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def display_thread_func(self):
        """Thread for visual display"""
        print("🔄 Starting visual display...")
        
        while self.running:
            try:
                if self.current_frame is not None:
                    with self.frame_lock:
                        display_frame = self.current_frame.copy()
                    
                    # Get recent objects for display
                    radar_objects = []
                    camera_objects = []
                    fused_objects = []
                    
                    # Collect radar objects from queue (non-blocking peek)
                    temp_radar = []
                    while not self.radar_queue.empty():
                        try:
                            obj = self.radar_queue.get_nowait()
                            temp_radar.append(obj)
                            radar_objects.append(obj)
                        except queue.Empty:
                            break
                    # Put them back
                    for obj in temp_radar:
                        if not self.radar_queue.full():
                            self.radar_queue.put(obj)
                    
                    # Collect camera objects
                    temp_camera = []
                    while not self.camera_queue.empty():
                        try:
                            obj = self.camera_queue.get_nowait()
                            temp_camera.append(obj)
                            camera_objects.append(obj)
                        except queue.Empty:
                            break
                    # Put them back
                    for obj in temp_camera:
                        if not self.camera_queue.full():
                            self.camera_queue.put(obj)
                    
                    # Collect fused objects
                    temp_fused = []
                    while not self.fusion_queue.empty():
                        try:
                            obj = self.fusion_queue.get_nowait()
                            temp_fused.append(obj)
                            fused_objects.append(obj)
                        except queue.Empty:
                            break
                    # Put them back
                    for obj in temp_fused:
                        if not self.fusion_queue.full():
                            self.fusion_queue.put(obj)
                    
                    # Draw all objects on frame
                    self.draw_camera_objects(display_frame, camera_objects)
                    self.draw_radar_objects(display_frame, radar_objects)
                    self.draw_fused_objects(display_frame, fused_objects)
                    self.draw_status_overlay(display_frame)
                    
                    # Show the frame
                    cv2.imshow('SafeSpeed AI - Radar-Camera Fusion', display_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Display quit requested")
                        self.running = False
                        break
                
                time.sleep(1/30)  # 30 FPS display
                
            except Exception as e:
                print(f"❌ Display thread error: {e}")
                time.sleep(1)
        
        cv2.destroyAllWindows()
        print("✅ Display thread stopped")
    
    def start(self):
        """Start the fusion system"""
        print("🚀 Starting Simple Radar-Camera Fusion System...")
        
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
        
        print("✅ Fusion system started successfully")
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
        
        print("✅ Fusion system stopped")
    
    def get_latest_results(self) -> List[FusedObject]:
        """Get the latest fusion results"""
        results = []
        
        while not self.fusion_queue.empty():
            try:
                results.append(self.fusion_queue.get_nowait())
            except queue.Empty:
                break
        
        return results
    
    def display_results(self):
        """Display fusion results in a simple format"""
        results = self.get_latest_results()
        
        if results:
            print(f"\n📊 Fusion Results ({len(results)} objects):")
            print("-" * 80)
            
            for i, obj in enumerate(results):
                print(f"Object {i+1}:")
                
                if obj.radar_data:
                    print(f"  🎯 Radar: Distance={obj.distance:.1f}m, Speed={obj.estimated_speed:.1f}km/h")
                    print(f"           Position=({obj.radar_data.x:.1f}, {obj.radar_data.y:.1f})")
                
                if obj.camera_data:
                    bbox = obj.camera_data.bbox
                    print(f"  📷 Camera: BBox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
                    print(f"            Class={obj.camera_data.class_name}, Conf={obj.camera_data.confidence:.2f}")
                
                print(f"  🔗 Fusion Confidence: {obj.confidence:.2f}")
                print()


def main():
    """Main function for testing the simple fusion system"""
    print("🚗 SafeSpeed AI - Simple Radar-Camera Fusion Test")
    print("=" * 60)
    
    fusion_system = SimpleRadarCameraFusion(camera_id=0)
    
    try:
        if fusion_system.start():
            print("\n🎬 Visual display window opened!")
            print("📊 Real-time radar-camera fusion display:")
            print("   📷 Green boxes = Camera detections")
            print("   🟡 Yellow circles = Radar detections") 
            print("   💜 Magenta lines = Fused objects")
            print("   ⌨️  Press 'q' in the window to quit")
            print("\n⏱️  System running... Waiting for display window...")
            
            # Wait for system to run (display thread will handle exit)
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
