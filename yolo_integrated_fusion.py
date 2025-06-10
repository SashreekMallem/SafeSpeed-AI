#!/usr/bin/env python3

"""
YOLO-Integrated Radar-Camera Fusion System
Implements proper vehicle detection using YOLO and Hungarian algorithm for optimal association
Based on research requirements for achieving >80% fusion rates with sub-pixel accuracy
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
import sys
import os
from flask import Flask, render_template, jsonify, Response
import argparse
from scipy.optimize import linear_sum_assignment
import urllib.request

# Add the project directory to Python path
sys.path.append('/home/projecta/SafespeedAI')

# Import our radar interface
from awr1843_interface import AWR1843Interface

@dataclass
class RadarObject:
    """Represents a detected radar object"""
    x: float  # meters (lateral)
    y: float  # meters (longitudinal) 
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
    class_id: int = 0  # YOLO class ID

@dataclass
class FusedObject:
    """Represents a fused radar-camera object"""
    radar_data: Optional[RadarObject]
    camera_data: Optional[CameraObject]
    estimated_speed: Optional[float]  # km/h
    distance: Optional[float]  # meters
    confidence: float
    timestamp: float
    fusion_quality: float = 0.0  # Quality metric for fusion

class YOLOVehicleDetector:
    """YOLO-based vehicle detection system"""
    
    def __init__(self):
        self.net = None
        self.output_layers = None
        self.classes = []
        self.initialized = False
        self.vehicle_classes = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
        
    def initialize(self):
        """Initialize YOLO network"""
        try:
            print("🔧 Initializing YOLO vehicle detector...")
            
            # Try to download YOLO files if they don't exist
            weights_path = '/home/projecta/SafespeedAI/yolo/yolov3.weights'
            config_path = '/home/projecta/SafespeedAI/yolo/yolov3.cfg'
            classes_path = '/home/projecta/SafespeedAI/yolo/coco.names'
            
            # Create yolo directory
            os.makedirs('/home/projecta/SafespeedAI/yolo', exist_ok=True)
            
            # Download YOLO files if they don't exist
            if not os.path.exists(weights_path):
                print("📥 Downloading YOLOv3 weights...")
                urllib.request.urlretrieve(
                    'https://pjreddie.com/media/files/yolov3.weights',
                    weights_path
                )
                
            if not os.path.exists(config_path):
                print("📥 Downloading YOLOv3 config...")
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
                    config_path
                )
                
            if not os.path.exists(classes_path):
                print("📥 Downloading COCO classes...")
                urllib.request.urlretrieve(
                    'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
                    classes_path
                )
            
            # Load YOLO network
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Use GPU if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("✅ Using GPU acceleration for YOLO")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                print("⚠️  Using CPU for YOLO (consider GPU for better performance)")
                
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
                
            self.initialized = True
            print("✅ YOLO detector initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            print("⚠️  Falling back to enhanced background subtraction")
            return False
    
    def detect_vehicles(self, frame) -> List[CameraObject]:
        """Detect vehicles using YOLO"""
        if not self.initialized:
            return []
            
        try:
            height, width, channels = frame.shape
            timestamp = time.time()
            
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            
            # Run inference
            outputs = self.net.forward(self.output_layers)
            
            # Parse detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter for vehicles only
                    if class_id in self.vehicle_classes and confidence > 0.3:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            
            vehicles = []
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    class_name = self.classes[class_id] if class_id < len(self.classes) else "vehicle"
                    
                    # Ensure bounding box is within frame
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 20 and h > 20:  # Minimum size filter
                        camera_obj = CameraObject(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            class_name=class_name,
                            timestamp=timestamp,
                            class_id=class_id
                        )
                        vehicles.append(camera_obj)
                        
            return vehicles
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return []

class HungarianAssociator:
    """Hungarian algorithm-based object association"""
    
    def __init__(self, max_distance=150.0, max_time_diff=0.15):
        self.max_distance = max_distance
        self.max_time_diff = max_time_diff
        
    def calculate_cost_matrix(self, radar_objects: List[RadarObject], 
                            camera_objects: List[CameraObject],
                            coordinate_transform_func) -> np.ndarray:
        """Calculate cost matrix for Hungarian algorithm"""
        if not radar_objects or not camera_objects:
            return np.array([])
            
        cost_matrix = np.zeros((len(radar_objects), len(camera_objects)))
        
        for i, radar_obj in enumerate(radar_objects):
            radar_x, radar_y = coordinate_transform_func(radar_obj)
            
            for j, camera_obj in enumerate(camera_objects):
                # Calculate camera object center
                cam_x = camera_obj.bbox[0] + camera_obj.bbox[2] / 2
                cam_y = camera_obj.bbox[1] + camera_obj.bbox[3] / 2
                
                # Spatial distance
                spatial_distance = np.sqrt((radar_x - cam_x)**2 + (radar_y - cam_y)**2)
                
                # Temporal distance
                time_diff = abs(radar_obj.timestamp - camera_obj.timestamp)
                
                # Combined cost with penalties
                if spatial_distance > self.max_distance or time_diff > self.max_time_diff:
                    cost = 1e6  # Very high cost for invalid associations
                else:
                    # Normalized cost combining spatial and temporal factors
                    spatial_cost = spatial_distance / self.max_distance
                    temporal_cost = time_diff / self.max_time_diff
                    
                    # Weight spatial distance more heavily
                    cost = 0.8 * spatial_cost + 0.2 * temporal_cost
                    
                    # Bonus for higher confidence detections
                    confidence_bonus = (1.0 - camera_obj.confidence) * 0.1
                    cost += confidence_bonus
                
                cost_matrix[i, j] = cost
                
        return cost_matrix
    
    def associate(self, radar_objects: List[RadarObject], 
                 camera_objects: List[CameraObject],
                 coordinate_transform_func) -> List[Tuple[int, int, float]]:
        """
        Associate radar and camera objects using Hungarian algorithm
        Returns list of (radar_idx, camera_idx, cost) tuples
        """
        if not radar_objects or not camera_objects:
            return []
            
        cost_matrix = self.calculate_cost_matrix(radar_objects, camera_objects, coordinate_transform_func)
        
        if cost_matrix.size == 0:
            return []
            
        # Apply Hungarian algorithm
        radar_indices, camera_indices = linear_sum_assignment(cost_matrix)
        
        associations = []
        for r_idx, c_idx in zip(radar_indices, camera_indices):
            cost = cost_matrix[r_idx, c_idx]
            if cost < 1e5:  # Only valid associations
                associations.append((r_idx, c_idx, cost))
                
        return associations

class YOLOIntegratedFusion:
    """YOLO-integrated radar-camera fusion system"""
    
    def __init__(self, camera_id=0, port=5002):
        self.camera_id = camera_id
        self.port = port
        self.radar_interface = None
        self.camera = None
        
        # Detection systems
        self.yolo_detector = YOLOVehicleDetector()
        self.hungarian_associator = HungarianAssociator()
        
        # Calibration data
        self.calibration_matrix = None
        self.coord_ranges = {'x': [-5, 5], 'y': [0, 20]}  # Default ranges
        
        # Threading control
        self.running = False
        self.radar_thread = None
        self.camera_thread = None
        self.fusion_thread = None
        
        # Data storage
        self.current_radar_objects = []
        self.current_camera_objects = []
        self.current_fused_objects = []
        
        # Statistics
        self.radar_count = 0
        self.camera_count = 0
        self.fusion_count = 0
        self.fusion_rate = 0.0
        
        # Debug information
        self.debug_info = []
        
        # Flask app setup
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
        # Enhanced background subtraction as fallback
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=30,
            history=300
        )
        
        # Display variables
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        
        print("🎯 YOLO-Integrated Fusion System Initialized")
        
    def load_calibration_data(self):
        """Load calibration data if available"""
        try:
            # Try to load the most recent calibration file
            calib_files = [f for f in os.listdir('/home/projecta/SafespeedAI/') 
                          if f.startswith('corner_reflector_calibration_') and f.endswith('.json')]
            
            if calib_files:
                calib_files.sort(reverse=True)  # Most recent first
                calib_file = calib_files[0]
                
                with open(f'/home/projecta/SafespeedAI/{calib_file}', 'r') as f:
                    calib_data = json.load(f)
                    
                print(f"✅ Loaded calibration data from {calib_file}")
                
                # Extract coordinate ranges if available
                if 'coordinate_ranges' in calib_data:
                    self.coord_ranges = calib_data['coordinate_ranges']
                    
                return True
                
        except Exception as e:
            print(f"⚠️  Could not load calibration data: {e}")
            
        return False
        
    def calibrated_coordinate_transform(self, radar_obj: RadarObject, 
                                     img_width=1280, img_height=720) -> Tuple[int, int]:
        """Transform radar coordinates to image coordinates using calibration data"""
        try:
            if self.calibration_matrix is not None:
                # Use proper calibration matrix when available
                # This is a placeholder for the full implementation
                pass
                
            # Enhanced coordinate transformation based on characterization
            x_range = self.coord_ranges['x']
            y_range = self.coord_ranges['y']
            
            # Adaptive scaling based on actual coordinate ranges
            if x_range[1] - x_range[0] > 0 and y_range[1] - y_range[0] > 0:
                x_scale = img_width / (x_range[1] - x_range[0])
                y_scale = img_height / (y_range[1] - y_range[0]) 
                
                # Center the coordinate system
                x_offset = -x_range[0] if x_range[0] < 0 else 0
                y_offset = -y_range[0] if y_range[0] < 0 else 0
                
                img_x = int((radar_obj.x + x_offset) * x_scale * 0.6 + img_width / 2)
                img_y = int(img_height - (radar_obj.y + y_offset) * y_scale * 0.4)
            else:
                # Research-based improved scaling
                scale_x = 50
                scale_y = 25
                
                img_x = int(img_width / 2 + radar_obj.x * scale_x)
                img_y = int(img_height - 100 - radar_obj.y * scale_y)
            
            # Clamp to image bounds
            img_x = max(10, min(img_width - 10, img_x))
            img_y = max(10, min(img_height - 10, img_y))
            
            return img_x, img_y
            
        except Exception as e:
            print(f"❌ Coordinate transform error: {e}")
            # Fallback transformation
            img_x = int(640 + radar_obj.x * 40)
            img_y = int(360 - radar_obj.y * 20)
            return max(0, min(img_width, img_x)), max(0, min(img_height, img_y))
    
    def enhanced_fallback_detection(self, frame) -> List[CameraObject]:
        """Enhanced fallback vehicle detection if YOLO fails"""
        try:
            fg_mask = self.background_subtractor.apply(frame)
            
            # Enhanced morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Noise reduction
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.erode(fg_mask, kernel_small)
            fg_mask = cv2.dilate(fg_mask, kernel_small)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vehicles = []
            timestamp = time.time()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Research-based vehicle filtering
                if area > 12000:  # Higher threshold to avoid false positives
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Strict vehicle-like criteria
                    if (0.7 < aspect_ratio < 3.5 and  
                        w > 80 and h > 60 and          
                        area / (w * h) > 0.35):         
                        
                        camera_obj = CameraObject(
                            bbox=(x, y, w, h),
                            confidence=min(0.8, area / 60000),
                            class_name="vehicle",
                            timestamp=timestamp
                        )
                        vehicles.append(camera_obj)
                        
            return vehicles
            
        except Exception as e:
            print(f"❌ Fallback detection error: {e}")
            return []
    
    def create_fused_objects(self, radar_objects: List[RadarObject], 
                           camera_objects: List[CameraObject]) -> List[FusedObject]:
        """Create fused objects using Hungarian algorithm"""
        fused_objects = []
        
        if not radar_objects and not camera_objects:
            return fused_objects
            
        # Get associations using Hungarian algorithm
        associations = self.hungarian_associator.associate(
            radar_objects, camera_objects, self.calibrated_coordinate_transform
        )
        
        # Track which objects have been associated
        used_radar = set()
        used_camera = set()
        
        # Create fused objects for valid associations
        for radar_idx, camera_idx, cost in associations:
            if radar_idx < len(radar_objects) and camera_idx < len(camera_objects):
                radar_obj = radar_objects[radar_idx]
                camera_obj = camera_objects[camera_idx]
                
                # Calculate fusion quality based on cost
                fusion_quality = max(0.0, 1.0 - cost)
                
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=camera_obj,
                    estimated_speed=abs(radar_obj.range_rate) * 3.6,
                    distance=radar_obj.range,
                    confidence=min(0.95, (radar_obj.snr + camera_obj.confidence) / 2),
                    timestamp=max(radar_obj.timestamp, camera_obj.timestamp),
                    fusion_quality=fusion_quality
                )
                
                fused_objects.append(fused_obj)
                used_radar.add(radar_idx)
                used_camera.add(camera_idx)
                
                self.debug_info.append(f"🎯 Hungarian Fusion: R{radar_idx}+C{camera_idx}, Cost: {cost:.3f}, Quality: {fusion_quality:.3f}")
        
        # Add unmatched radar objects
        for i, radar_obj in enumerate(radar_objects):
            if i not in used_radar:
                fused_obj = FusedObject(
                    radar_data=radar_obj,
                    camera_data=None,
                    estimated_speed=abs(radar_obj.range_rate) * 3.6,
                    distance=radar_obj.range,
                    confidence=radar_obj.snr,
                    timestamp=radar_obj.timestamp,
                    fusion_quality=0.5  # Partial quality for radar-only
                )
                fused_objects.append(fused_obj)
                
        # Add unmatched camera objects
        for i, camera_obj in enumerate(camera_objects):
            if i not in used_camera:
                fused_obj = FusedObject(
                    radar_data=None,
                    camera_data=camera_obj,
                    estimated_speed=None,
                    distance=None,
                    confidence=camera_obj.confidence,
                    timestamp=camera_obj.timestamp,
                    fusion_quality=0.3  # Lower quality for camera-only
                )
                fused_objects.append(fused_obj)
                
        return fused_objects
    
    def calculate_fusion_metrics(self):
        """Calculate fusion performance metrics"""
        total_radar = len(self.current_radar_objects)
        total_camera = len(self.current_camera_objects)
        
        # Count successful fusions (objects with both radar and camera data)
        successful_fusions = sum(1 for obj in self.current_fused_objects 
                               if obj.radar_data is not None and obj.camera_data is not None)
        
        # Calculate fusion rate
        if total_radar > 0 and total_camera > 0:
            possible_fusions = min(total_radar, total_camera)
            self.fusion_rate = successful_fusions / possible_fusions if possible_fusions > 0 else 0.0
        else:
            self.fusion_rate = 0.0
            
        self.fusion_count = successful_fusions
        
        # Add metrics to debug info
        self.debug_info.append(f"📊 Fusion Metrics: {successful_fusions}/{min(total_radar, total_camera)} = {self.fusion_rate:.1%}")
    
    def setup_flask_routes(self):
        """Setup Flask web routes"""
        @self.app.route('/')
        def index():
            return render_template('yolo_fusion.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/status')
        def api_status():
            return jsonify({
                'radar_count': self.radar_count,
                'camera_count': self.camera_count,
                'fusion_count': self.fusion_count,
                'fusion_rate': self.fusion_rate,
                'debug_info': self.debug_info[-10:],  # Last 10 debug messages
                'yolo_enabled': self.yolo_detector.initialized,
                'timestamp': time.time()
            })
    
    def generate_frames(self):
        """Generate video frames for web streaming"""
        while True:
            with self.frame_lock:
                if self.annotated_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.annotated_frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def initialize_systems(self):
        """Initialize all systems"""
        success = True
        
        # Initialize YOLO detector
        if not self.yolo_detector.initialize():
            print("⚠️  YOLO initialization failed, using enhanced fallback detection")
            
        # Initialize radar
        try:
            print("🔗 Initializing AWR1843BOOST radar...")
            self.radar_interface = AWR1843Interface()
            
            if self.radar_interface.connect():
                print("✅ Radar connected successfully")
                if self.radar_interface.configure():
                    print("✅ Radar configured successfully")
                else:
                    print("❌ Failed to configure radar")
                    success = False
            else:
                print("❌ Failed to connect to radar")
                success = False
        except Exception as e:
            print(f"❌ Radar initialization error: {e}")
            success = False
            
        # Initialize camera
        try:
            print("📷 Initializing camera...")
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                print("❌ Failed to open camera")
                success = False
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                print("✅ Camera initialized successfully")
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            success = False
            
        # Load calibration data
        self.load_calibration_data()
        
        return success
    
    def radar_data_thread(self):
        """Thread for collecting radar data"""
        print("🎯 Starting radar data collection...")
        
        while self.running:
            try:
                if self.radar_interface:
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
                        
                        self.current_radar_objects = radar_objects
                    else:
                        self.radar_count = 0
                        self.current_radar_objects = []
                        
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                print(f"❌ Radar thread error: {e}")
                time.sleep(0.1)
    
    def camera_data_thread(self):
        """Thread for collecting camera data"""
        print("📷 Starting camera data collection...")
        
        while self.running:
            try:
                if self.camera:
                    ret, frame = self.camera.read()
                    
                    if ret:
                        # Attempt YOLO detection first
                        if self.yolo_detector.initialized:
                            camera_objects = self.yolo_detector.detect_vehicles(frame)
                        else:
                            # Fallback to enhanced background subtraction
                            camera_objects = self.enhanced_fallback_detection(frame)
                        
                        self.camera_count = len(camera_objects)
                        self.current_camera_objects = camera_objects
                        
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                    else:
                        self.camera_count = 0
                        self.current_camera_objects = []
                        
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Camera thread error: {e}")
                time.sleep(0.1)
    
    def fusion_thread(self):
        """Thread for processing fusion"""
        print("🔬 Starting fusion processing...")
        
        while self.running:
            try:
                # Clear debug info
                self.debug_info = []
                
                # Create fused objects
                fused_objects = self.create_fused_objects(
                    self.current_radar_objects, 
                    self.current_camera_objects
                )
                
                self.current_fused_objects = fused_objects
                
                # Calculate metrics
                self.calculate_fusion_metrics()
                
                # Update annotated frame
                self.update_display_frame()
                
                time.sleep(0.1)  # 10 Hz fusion processing
                
            except Exception as e:
                print(f"❌ Fusion thread error: {e}")
                time.sleep(0.1)
    
    def update_display_frame(self):
        """Update the display frame with annotations"""
        with self.frame_lock:
            if self.current_frame is not None:
                display_frame = self.current_frame.copy()
                
                # Draw camera detections (green boxes)
                for obj in self.current_camera_objects:
                    x, y, w, h = obj.bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{obj.class_name}: {obj.confidence:.2f}"
                    cv2.putText(display_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw radar detections (yellow circles)
                for obj in self.current_radar_objects:
                    img_x, img_y = self.calibrated_coordinate_transform(obj)
                    cv2.circle(display_frame, (img_x, img_y), 8, (0, 255, 255), -1)
                    label = f"R: {obj.speed_kmh:.1f}km/h"
                    cv2.putText(display_frame, label, (img_x + 10, img_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw fusion connections (magenta lines)
                for obj in self.current_fused_objects:
                    if obj.radar_data and obj.camera_data:
                        # Radar position
                        radar_x, radar_y = self.calibrated_coordinate_transform(obj.radar_data)
                        
                        # Camera center
                        cam_x = obj.camera_data.bbox[0] + obj.camera_data.bbox[2] // 2
                        cam_y = obj.camera_data.bbox[1] + obj.camera_data.bbox[3] // 2
                        
                        # Draw connection line
                        cv2.line(display_frame, (radar_x, radar_y), (cam_x, cam_y), (255, 0, 255), 2)
                        
                        # Draw fusion quality
                        mid_x = (radar_x + cam_x) // 2
                        mid_y = (radar_y + cam_y) // 2
                        quality_text = f"Q:{obj.fusion_quality:.2f}"
                        cv2.putText(display_frame, quality_text, (mid_x, mid_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # Add statistics overlay
                stats_text = [
                    f"YOLO Fusion System - Port {self.port}",
                    f"Radar: {self.radar_count} objects",
                    f"Camera: {self.camera_count} objects ({self.yolo_detector.initialized and 'YOLO' or 'Fallback'})",
                    f"Fused: {self.fusion_count} objects",
                    f"Fusion Rate: {self.fusion_rate:.1%}",
                    f"Avg Quality: {np.mean([obj.fusion_quality for obj in self.current_fused_objects]):.2f}" if self.current_fused_objects else "Avg Quality: 0.00"
                ]
                
                for i, text in enumerate(stats_text):
                    cv2.putText(display_frame, text, (10, 30 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                self.annotated_frame = display_frame
    
    def start(self):
        """Start the YOLO-integrated fusion system"""
        print("🎯" * 30)
        print("🎯  YOLO-Integrated Fusion System  🎯")
        print("🎯" * 30)
        print("✅ YOLO vehicle detection")
        print("✅ Hungarian algorithm association")
        print("✅ Research-based coordinate transformation")
        print("✅ Fusion quality metrics")
        print("=" * 60)
        
        if not self.initialize_systems():
            print("❌ System initialization failed")
            return False
        
        self.running = True
        
        # Start processing threads
        self.radar_thread = threading.Thread(target=self.radar_data_thread, daemon=True)
        self.camera_thread = threading.Thread(target=self.camera_data_thread, daemon=True)
        self.fusion_thread = threading.Thread(target=self.fusion_thread, daemon=True)
        
        self.radar_thread.start()
        self.camera_thread.start()
        self.fusion_thread.start()
        
        print(f"🌐 Starting web server on http://localhost:{self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
    
    def stop(self):
        """Stop the fusion system"""
        print("🛑 Stopping YOLO-integrated fusion system...")
        self.running = False
        
        if self.radar_interface:
            self.radar_interface.stop()
            self.radar_interface.disconnect()
            
        if self.camera:
            self.camera.release()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLO-Integrated Radar-Camera Fusion')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--port', type=int, default=5002, help='Web server port')
    
    args = parser.parse_args()
    
    fusion_system = YOLOIntegratedFusion(camera_id=args.camera, port=args.port)
    
    try:
        fusion_system.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        fusion_system.stop()

if __name__ == "__main__":
    main()
