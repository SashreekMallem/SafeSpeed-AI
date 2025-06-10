#!/usr/bin/env python3

"""
Fixed Web-based Visual Display for Radar-Camera Fusion
Addresses coordinate mapping and detection issues
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sys
import os
from flask import Flask, render_template, jsonify, Response
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

class FixedWebDisplayFusion:
    """Fixed web-based visual display for radar-camera fusion"""
    
    def __init__(self, camera_id=0, port=5000):
        self.camera_id = camera_id
        self.port = port
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
        
        # Fusion parameters - MUCH more lenient for testing
        self.max_fusion_distance = 400.0  # Increased from 200 pixels
        self.max_time_diff = 0.5  # Increased from 0.1 seconds
        
        # Improved vehicle detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,  # Disable shadow detection 
            varThreshold=50,      # Increase threshold to reduce noise
            history=500           # Longer history for better background model
        )
        
        # Display variables
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection counters and statistics
        self.radar_count = 0
        self.camera_count = 0
        self.fusion_count = 0
        self.last_update_time = time.time()
        
        # Current objects for display
        self.current_radar_objects = []
        self.current_camera_objects = []
        self.current_fused_objects = []
        
        # Colors for display (BGR format for OpenCV)
        self.colors = {
            'radar': (0, 255, 255),      # Yellow
            'camera': (0, 255, 0),       # Green  
            'fusion': (255, 0, 255),     # Magenta
            'text': (255, 255, 255),     # White
            'background': (0, 0, 0),     # Black
            'connection': (255, 100, 255) # Light Magenta
        }
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
        # Debug info
        self.debug_info = []
        
    def setup_flask_routes(self):
        """Setup Flask web routes"""
        
        @self.app.route('/')
        def index():
            return '''
<!DOCTYPE html>
<html>
<head>
    <title>SafeSpeed AI - FIXED Radar Camera Fusion</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
        }
        .header {
            text-align: center;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .video-section {
            flex: 2;
        }
        .status-section {
            flex: 1;
            background-color: #2a2a2a;
            padding: 20px;
            border-radius: 10px;
            height: fit-content;
        }
        .video-frame {
            width: 100%;
            max-width: 800px;
            border: 3px solid #4ecdc4;
            border-radius: 10px;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #3a3a3a;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .radar-stat { border-left: 4px solid #ffff00; }
        .camera-stat { border-left: 4px solid #00ff00; }
        .fusion-stat { border-left: 4px solid #ff00ff; }
        .objects-list {
            background-color: #3a3a3a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .object-item {
            background-color: #4a4a4a;
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid;
        }
        .radar-object { border-left-color: #ffff00; }
        .camera-object { border-left-color: #00ff00; }
        .fused-object { border-left-color: #ff00ff; }
        .legend {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .running { background-color: #00ff00; }
        .stopped { background-color: #ff0000; }
        .debug-section {
            background-color: #3a3a3a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
        }
        .fusion-highlight {
            background-color: #ff00ff !important;
            color: white !important;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 SafeSpeed AI - FIXED Radar Camera Fusion 🚗</h1>
        <p>Real-time Visual Display with Proper Coordinate Mapping</p>
    </div>
    
    <div class="container">
        <div class="video-section">
            <h2>📹 Live Camera Feed with Fixed Overlays</h2>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ffff00;"></div>
                    <span>Radar Objects (Yellow Circles)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #00ff00;"></div>
                    <span>Camera Objects (Green Boxes)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff00ff;"></div>
                    <span>🎉 FUSION LINES (Magenta)</span>
                </div>
            </div>
            <img id="videoFeed" class="video-frame" src="/video_feed" alt="Live Video Feed">
        </div>
        
        <div class="status-section">
            <h2>📊 System Status</h2>
            <div>
                <span class="status-indicator running" id="statusIndicator"></span>
                <strong id="systemStatus">RUNNING</strong>
            </div>
            <p><strong>Time:</strong> <span id="currentTime"></span></p>
            
            <div class="stats">
                <div class="stat-card radar-stat">
                    <h3>🎯 Radar</h3>
                    <div style="font-size: 24px; font-weight: bold;" id="radarCount">0</div>
                    <div>Objects</div>
                </div>
                <div class="stat-card camera-stat">
                    <h3>📷 Camera</h3>
                    <div style="font-size: 24px; font-weight: bold;" id="cameraCount">0</div>
                    <div>Objects</div>
                </div>
                <div class="stat-card fusion-stat">
                    <h3>🔗 Fused</h3>
                    <div style="font-size: 24px; font-weight: bold;" id="fusionCount">0</div>
                    <div>Objects</div>
                </div>
            </div>
            
            <div class="objects-list">
                <h3>🎯 Detected Objects</h3>
                <div id="objectsList">
                    <p>No objects detected</p>
                </div>
            </div>
            
            <div class="debug-section">
                <h3>🔧 Debug Info</h3>
                <div id="debugInfo">
                    <p>Debug information will appear here</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update data every second
        function updateData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update counts
                    document.getElementById('radarCount').textContent = data.radar_count;
                    document.getElementById('cameraCount').textContent = data.camera_count;
                    document.getElementById('fusionCount').textContent = data.fusion_count;
                    
                    // Highlight fusion count if > 0
                    const fusionCard = document.querySelector('.fusion-stat');
                    if (data.fusion_count > 0) {
                        fusionCard.classList.add('fusion-highlight');
                    } else {
                        fusionCard.classList.remove('fusion-highlight');
                    }
                    
                    // Update time
                    document.getElementById('currentTime').textContent = new Date().toLocaleTimeString();
                    
                    // Update objects list
                    updateObjectsList(data.objects);
                    
                    // Update debug info
                    updateDebugInfo(data.debug_info);
                })
                .catch(error => console.error('Error:', error));
        }
        
        function updateObjectsList(objects) {
            const objectsList = document.getElementById('objectsList');
            if (objects.length === 0) {
                objectsList.innerHTML = '<p>No objects detected</p>';
                return;
            }
            
            const html = objects.map((obj, index) => {
                const className = obj.type.toLowerCase() + '-object';
                let details = '';
                
                if (obj.type === 'RADAR') {
                    details = `Speed: ${obj.speed.toFixed(1)} km/h | Distance: ${obj.distance.toFixed(1)}m | Pos: (${obj.x.toFixed(1)}, ${obj.y.toFixed(1)}) | Img: (${obj.img_x}, ${obj.img_y})`;
                } else if (obj.type === 'CAMERA') {
                    details = `Confidence: ${obj.confidence.toFixed(2)} | Center: (${obj.bbox_center_x}, ${obj.bbox_center_y}) | Size: ${obj.bbox_width}x${obj.bbox_height}`;
                } else if (obj.type === 'FUSED') {
                    details = `🎉 Speed: ${obj.speed ? obj.speed.toFixed(1) + ' km/h' : 'N/A'} | Distance: ${obj.distance ? obj.distance.toFixed(1) + 'm' : 'N/A'} | Confidence: ${obj.confidence.toFixed(2)}`;
                }
                
                return `
                    <div class="object-item ${className}">
                        <strong>${obj.type} #${index + 1}</strong><br>
                        <small>${details}</small>
                    </div>
                `;
            }).join('');
            
            objectsList.innerHTML = html;
        }
        
        function updateDebugInfo(debugInfo) {
            const debugDiv = document.getElementById('debugInfo');
            if (debugInfo && debugInfo.length > 0) {
                const html = debugInfo.slice(-5).map(info => `<p><small>${info}</small></p>`).join('');
                debugDiv.innerHTML = html;
            }
        }
        
        // Update every 1 second
        setInterval(updateData, 1000);
        updateData(); // Initial load
    </script>
</body>
</html>
            '''
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def api_status():
            objects_data = []
            
            # Add radar objects with image coordinates
            for i, obj in enumerate(self.current_radar_objects):
                img_x, img_y = self.convert_radar_to_image_coords(obj)
                objects_data.append({
                    'type': 'RADAR',
                    'speed': abs(obj.range_rate) * 3.6,
                    'distance': obj.range,
                    'x': obj.x,
                    'y': obj.y,
                    'img_x': img_x,
                    'img_y': img_y,
                    'confidence': obj.snr
                })
            
            # Add camera objects
            for i, obj in enumerate(self.current_camera_objects):
                bbox_center_x = obj.bbox[0] + obj.bbox[2] // 2
                bbox_center_y = obj.bbox[1] + obj.bbox[3] // 2
                objects_data.append({
                    'type': 'CAMERA',
                    'bbox_center_x': bbox_center_x,
                    'bbox_center_y': bbox_center_y,
                    'bbox_width': obj.bbox[2],
                    'bbox_height': obj.bbox[3],
                    'confidence': obj.confidence
                })
            
            # Add fused objects
            for i, obj in enumerate(self.current_fused_objects):
                if obj.radar_data and obj.camera_data:  # Only true fused objects
                    obj_data = {
                        'type': 'FUSED',
                        'speed': obj.estimated_speed,
                        'distance': obj.distance,
                        'confidence': obj.confidence
                    }
                    objects_data.append(obj_data)
            
            return jsonify({
                'radar_count': self.radar_count,
                'camera_count': self.camera_count,
                'fusion_count': self.fusion_count,
                'objects': objects_data,
                'debug_info': self.debug_info[-10:],  # Last 10 debug messages
                'timestamp': time.time()
            })
    
    def generate_frames(self):
        """Generate video frames for web streaming"""
        while True:
            if self.annotated_frame is not None:
                with self.frame_lock:
                    frame = self.annotated_frame.copy()
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
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
                            
                            # Debug output
                            if len(self.debug_info) > 50:
                                self.debug_info = self.debug_info[-25:]  # Keep last 25
                            
                            img_x, img_y = self.convert_radar_to_image_coords(radar_obj)
                            self.debug_info.append(f"Radar: ({obj.x:.2f},{obj.y:.2f}m) -> Img:({img_x},{img_y}px)")
                        
                        self.current_radar_objects = radar_objects
                    else:
                        self.radar_count = 0
                        self.current_radar_objects = []
                
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                print(f"❌ Radar data thread error: {e}")
                time.sleep(1)
    
    def detect_vehicles_improved(self, frame):
        """Improved vehicle detection - less sensitive to faces"""
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # More aggressive noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicles = []
            timestamp = time.time()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Much higher area threshold to avoid face detection
                if area > 8000:  # Increased from 1500 - should ignore faces
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # More restrictive filters for vehicle-like shapes
                    if (0.5 < aspect_ratio < 4.0 and  # More vehicle-like aspect ratio
                        w > 80 and h > 60 and          # Larger minimum size
                        area / (w * h) > 0.3):         # Density check
                        
                        camera_obj = CameraObject(
                            bbox=(x, y, w, h),
                            confidence=min(0.9, area / 50000),  # Adjusted scaling
                            class_name="vehicle",
                            timestamp=timestamp
                        )
                        vehicles.append(camera_obj)
                        
                        # Debug output
                        center_x = x + w // 2
                        center_y = y + h // 2
                        self.debug_info.append(f"Camera: Area:{area} Center:({center_x},{center_y}) Size:{w}x{h}")
            
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
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                        
                        vehicles = self.detect_vehicles_improved(frame)
                        self.camera_count = len(vehicles)
                        self.current_camera_objects = vehicles
                        
                        # Create annotated frame for display
                        self.create_annotated_frame(frame)
                
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"❌ Camera data thread error: {e}")
                time.sleep(1)
    
    def convert_radar_to_image_coords(self, radar_obj: RadarObject, img_width=1280, img_height=720) -> Tuple[int, int]:
        """FIXED: Convert radar coordinates to image pixel coordinates"""
        
        # MUCH LARGER scale factors to spread objects across the image
        # These values need to be calibrated based on your radar's field of view
        scale_x = 150  # Increased from 40 - pixels per meter lateral  
        scale_y = 80   # Increased from 15 - pixels per meter longitudinal
        
        # Convert to image coordinates 
        # Radar X (lateral) maps to image X
        # Radar Y (longitudinal) maps to image Y (inverted because radar Y=0 is close, image Y=0 is top)
        img_x = int(img_width / 2 + radar_obj.x * scale_x)
        img_y = int(img_height - 100 - radar_obj.y * scale_y)  # 100px from bottom as "radar position"
        
        # Clamp to image bounds
        img_x = max(0, min(img_width - 1, img_x))
        img_y = max(0, min(img_height - 1, img_y))
        
        return img_x, img_y
    
    def associate_radar_camera(self, radar_objects: List[RadarObject], 
                              camera_objects: List[CameraObject]) -> List[FusedObject]:
        """Associate radar and camera objects with debug output"""
        fused_objects = []
        used_camera_objects = set()
        
        # Debug fusion attempts
        self.debug_info.append(f"FUSION: {len(radar_objects)} radar, {len(camera_objects)} camera objects")
        
        for radar_obj in radar_objects:
            best_match = None
            best_distance = float('inf')
            best_camera_idx = -1
            
            radar_x, radar_y = self.convert_radar_to_image_coords(radar_obj)
            
            for i, camera_obj in enumerate(camera_objects):
                if i in used_camera_objects:
                    continue
                
                cam_x = camera_obj.bbox[0] + camera_obj.bbox[2] / 2
                cam_y = camera_obj.bbox[1] + camera_obj.bbox[3] / 2
                
                distance = np.sqrt((radar_x - cam_x)**2 + (radar_y - cam_y)**2)
                
                self.debug_info.append(f"Distance: Radar({radar_x},{radar_y}) to Cam({int(cam_x)},{int(cam_y)}) = {distance:.1f}px")
                
                if distance < self.max_fusion_distance and distance < best_distance:
                    best_distance = distance
                    best_match = camera_obj
                    best_camera_idx = i
            
            if best_match:
                used_camera_objects.add(best_camera_idx)
                estimated_speed_kmh = abs(radar_obj.range_rate) * 3.6
                
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=best_match,
                    estimated_speed=estimated_speed_kmh,
                    distance=radar_obj.range,
                    confidence=min(0.95, (radar_obj.snr + best_match.confidence) / 2),
                    timestamp=max(radar_obj.timestamp, best_match.timestamp)
                )
                
                self.debug_info.append(f"🎉 FUSION SUCCESS: Distance {best_distance:.1f}px")
            else:
                estimated_speed_kmh = abs(radar_obj.range_rate) * 3.6
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
                # Perform fusion with current objects
                if self.current_radar_objects or self.current_camera_objects:
                    fused_objects = self.associate_radar_camera(
                        self.current_radar_objects, self.current_camera_objects)
                    
                    # Count only true fused objects (both radar and camera data)
                    true_fused = [obj for obj in fused_objects 
                                if obj.radar_data and obj.camera_data]
                    self.fusion_count = len(true_fused)
                    self.current_fused_objects = fused_objects
                else:
                    self.fusion_count = 0
                    self.current_fused_objects = []
                
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                print(f"❌ Fusion thread error: {e}")
                time.sleep(1)
    
    def create_annotated_frame(self, frame):
        """Create annotated frame with all overlays"""
        annotated = frame.copy()
        
        # Draw camera objects (green boxes)
        for obj in self.current_camera_objects:
            x, y, w, h = obj.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), self.colors['camera'], 3)
            cv2.putText(annotated, f"Vehicle: {obj.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['camera'], 2)
        
        # Draw radar objects (yellow circles) - MUCH more visible
        for obj in self.current_radar_objects:
            img_x, img_y = self.convert_radar_to_image_coords(obj)
            
            # MUCH LARGER and more visible radar points
            cv2.circle(annotated, (img_x, img_y), 20, self.colors['radar'], -1)  # Filled circle
            cv2.circle(annotated, (img_x, img_y), 30, self.colors['radar'], 4)   # Outer ring
            
            # Speed and distance info - larger text
            speed_kmh = abs(obj.range_rate) * 3.6
            cv2.putText(annotated, f"{speed_kmh:.1f} km/h", (img_x + 35, img_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['radar'], 3)
            cv2.putText(annotated, f"{obj.range:.1f}m", (img_x + 35, img_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['radar'], 3)
            
            # Show radar coordinates
            cv2.putText(annotated, f"({obj.x:.1f},{obj.y:.1f})", (img_x + 35, img_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['radar'], 2)
        
        # Draw fusion connections (magenta lines) - VERY THICK
        fusion_pairs = [obj for obj in self.current_fused_objects 
                       if obj.radar_data and obj.camera_data]
        
        for obj in fusion_pairs:
            radar_x, radar_y = self.convert_radar_to_image_coords(obj.radar_data)
            cam_x = obj.camera_data.bbox[0] + obj.camera_data.bbox[2] // 2
            cam_y = obj.camera_data.bbox[1] + obj.camera_data.bbox[3] // 2
            
            # Draw VERY THICK connection line
            cv2.line(annotated, (radar_x, radar_y), (cam_x, cam_y), self.colors['fusion'], 8)
            
            # Fusion info at midpoint - larger text
            mid_x = (radar_x + cam_x) // 2
            mid_y = (radar_y + cam_y) // 2
            cv2.putText(annotated, f"🎉 FUSED: {obj.confidence:.2f}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['fusion'], 3)
        
        # Draw status overlay
        self.draw_status_overlay(annotated)
        
        with self.frame_lock:
            self.annotated_frame = annotated
    
    def draw_status_overlay(self, frame):
        """Draw system status overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 160), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y_pos = 35
        cv2.putText(frame, "SafeSpeed AI - FIXED Radar-Camera Fusion", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        
        y_pos += 25
        cv2.putText(frame, f"Radar: {self.radar_count} | Camera: {self.camera_count} | Fused: {self.fusion_count}",
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        y_pos += 20
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        y_pos += 20
        cv2.putText(frame, f"Fusion Distance Threshold: {self.max_fusion_distance:.0f}px", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        y_pos += 25
        # Legend
        cv2.putText(frame, "Yellow=Radar  Green=Camera  Magenta=Fusion", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def start(self):
        """Start the fusion system"""
        print("🚀 Starting FIXED Web-based Radar-Camera Fusion System...")
        
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
        
        print("✅ FIXED System started successfully")
        print(f"🌐 Web interface will be available at: http://localhost:{self.port}")
        print(f"🎬 Open your browser to see the FIXED visual display!")
        print("🔧 FIXES APPLIED:")
        print("   • Fixed radar coordinate mapping (larger scale factors)")
        print("   • Improved camera detection (higher thresholds to avoid faces)")
        print("   • Increased fusion distance threshold (400px)")
        print("   • Added debug information display")
        print("   • Made radar points much more visible")
        
        return True
    
    def stop(self):
        """Stop the fusion system"""
        print("🛑 Stopping fusion system...")
        
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if self.radar_interface:
            self.radar_interface.disconnect()
        
        print("✅ Fusion system stopped")
    
    def run_web_server(self):
        """Run the Flask web server"""
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='FIXED Web-based Radar-Camera Fusion Visualization')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--port', type=int, default=5001, help='Web server port (default: 5001)')
    
    args = parser.parse_args()
    
    print("🔧" * 30)
    print("🔧  SafeSpeed AI - FIXED Visual Display  🔧")
    print("🔧" * 30)
    print(f"📷 Camera ID: {args.camera}")
    print(f"🌐 Web Port: {args.port}")
    print("=" * 60)
    
    fusion_system = FixedWebDisplayFusion(camera_id=args.camera, port=args.port)
    
    try:
        if fusion_system.start():
            print(f"\n🎬 Starting FIXED web server on port {args.port}...")
            print(f"🌐 Open your browser and go to: http://localhost:{args.port}")
            print("⏱️  Press Ctrl+C to stop")
            
            fusion_system.run_web_server()
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        fusion_system.stop()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
