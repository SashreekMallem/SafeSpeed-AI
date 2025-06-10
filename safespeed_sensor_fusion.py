#!/usr/bin/env python3
"""
Complete SafeSpeed AI sensor fusion system integrating:
- AWR1843BOOST mmWave radar (for precise speed measurement)  
- Camera AI pipeline (for license plate recognition)
- Real-time sensor fusion for accurate speed + LPR
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import pyds
import cv2
import numpy as np
import threading
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import math

# Import our working radar interface
from awr1843_interface import AWR1843Interface, RadarObject

@dataclass
class CameraTarget:
    """Vehicle detected by camera"""
    frame_id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    license_plate: Optional[str] = None  
    timestamp: float = 0.0

@dataclass  
class FusedTarget:
    """Fused target with camera + radar data"""
    target_id: int
    camera_data: CameraTarget
    radar_data: Optional[RadarObject]
    speed_kmh: float
    position_m: Tuple[float, float]  # x, y in meters
    timestamp: float
    track_history: List[Tuple[float, float, float]] = None  # (x, y, timestamp)

class RadarCameraFusion:
    """Sensor fusion engine for radar + camera data"""
    
    def __init__(self):
        self.radar = AWR1843Interface()
        self.radar_thread = None
        self.radar_running = False
        self.latest_radar_objects = []
        self.radar_lock = threading.Lock()
        
        # Fusion parameters
        self.max_association_distance = 2.0  # meters
        self.speed_filter_alpha = 0.3  # Smoothing factor
        self.target_tracks = {}  # target_id -> FusedTarget
        self.next_target_id = 1
        
    def start_radar(self) -> bool:
        """Start radar data collection"""
        print("🔗 Starting radar subsystem...")
        
        if not self.radar.connect():
            return False
            
        if not self.radar.configure():
            return False
            
        # Start radar reading thread
        self.radar_running = True
        self.radar_thread = threading.Thread(target=self._radar_loop, daemon=True)
        self.radar_thread.start()
        
        print("✅ Radar subsystem started")
        return True
    
    def _radar_loop(self):
        """Radar data collection loop"""
        print("📡 Radar data collection started")
        
        while self.radar_running:
            try:
                objects = self.radar.read_frame(timeout=0.3)
                
                if objects is not None:
                    with self.radar_lock:
                        self.latest_radar_objects = objects
                        
                    if len(objects) > 0:
                        print(f"📊 Radar: {len(objects)} objects, "
                              f"speeds: {[f'{obj.speed_kmh:.1f}' for obj in objects[:3]]} km/h")
                        
            except Exception as e:
                print(f"❌ Radar error: {e}")
                time.sleep(0.1)
    
    def get_radar_objects(self) -> List[RadarObject]:
        """Get latest radar objects thread-safe"""
        with self.radar_lock:
            return self.latest_radar_objects.copy()
    
    def associate_targets(self, camera_targets: List[CameraTarget]) -> List[FusedTarget]:
        """Associate camera detections with radar objects"""
        radar_objects = self.get_radar_objects()
        fused_targets = []
        
        # Simple nearest-neighbor association
        for cam_target in camera_targets:
            # Convert camera bbox to approximate world position
            cam_x, cam_y = self._camera_to_world_coords(cam_target.bbox)
            
            best_radar = None
            best_distance = float('inf')
            
            # Find closest radar object
            for radar_obj in radar_objects:
                distance = math.sqrt((cam_x - radar_obj.x)**2 + (cam_y - radar_obj.y)**2)
                
                if distance < self.max_association_distance and distance < best_distance:
                    best_distance = distance
                    best_radar = radar_obj
            
            # Create fused target
            speed_kmh = best_radar.speed_kmh if best_radar else 0.0
            position = (best_radar.x, best_radar.y) if best_radar else (cam_x, cam_y)
            
            fused_target = FusedTarget(
                target_id=self.next_target_id,
                camera_data=cam_target,
                radar_data=best_radar,
                speed_kmh=speed_kmh,
                position_m=position,
                timestamp=time.time()
            )
            
            fused_targets.append(fused_target)
            self.next_target_id += 1
        
        return fused_targets
    
    def _camera_to_world_coords(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Convert camera bounding box to approximate world coordinates"""
        # Simple approximation - in real system would use camera calibration
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Rough conversion assuming camera FOV and distance
        world_x = (center_x - 640) * 0.01  # Scale factor
        world_y = 2.0 + (480 - center_y) * 0.005  # Assume 2m forward distance
        
        return world_x, world_y
    
    def stop_radar(self):
        """Stop radar subsystem"""
        self.radar_running = False
        if self.radar_thread:
            self.radar_thread.join()
        self.radar.disconnect()
        print("🛑 Radar subsystem stopped")

class SafeSpeedAISensorFusion:
    """Main SafeSpeed AI system with sensor fusion"""
    
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        
        # Initialize fusion engine
        self.fusion = RadarCameraFusion()
        
        # Detection tracking
        self.frame_count = 0
        self.detections_log = []
        
        # Pipeline components
        self.pipeline = None
        self.loop = None
        
    def create_pipeline(self):
        """Create the enhanced GStreamer pipeline with sensor fusion"""
        
        pipeline_str = f"""
        filesrc location=/home/projecta/SafespeedAI/sample_720p.h264
        ! h264parse
        ! nvv4l2decoder
        ! m.sink_0
        nvstreammux name=m batch-size=1 width=1280 height=720
            ! nvinfer config-file-path=/home/projecta/SafespeedAI/deepstream_app_config.txt batch-size=1 unique-id=1
            ! nvmultistreamtiler rows=1 columns=1 width=1280 height=720
            ! nvvideoconvert
            ! nvdsosd
            ! nvegltransform
            ! nveglglessink name=sink
        """
        
        print("🔧 Creating enhanced pipeline with sensor fusion...")
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get the source pad of nvinfer for probe attachment
        nvinfer = self.pipeline.get_by_name("nvinfer0")
        if not nvinfer:
            # Try alternative names
            for element in self.pipeline.iterate_elements():
                if element.get_factory().get_name() == "nvinfer":
                    nvinfer = element
                    break
        
        if nvinfer:
            nvinfer_src_pad = nvinfer.get_static_pad("src")
            if nvinfer_src_pad:
                nvinfer_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.nvinfer_src_pad_buffer_probe, 0)
                print("✅ Attached probe to nvinfer source pad")
        
        # Bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call)
        
    def nvinfer_src_pad_buffer_probe(self, pad, info, u_data):
        """Enhanced probe callback with sensor fusion"""
        
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                self.frame_count += 1
                
                # Extract camera detections
                camera_targets = self._extract_camera_targets(frame_meta)
                print(f"\n🎯 Frame {self.frame_count}: {len(camera_targets)} camera detections")
                
                # Perform sensor fusion
                if camera_targets:
                    fused_targets = self.fusion.associate_targets(camera_targets)
                    
                    # Display results
                    for target in fused_targets:
                        cam = target.camera_data
                        radar = target.radar_data
                        
                        print(f"   🚗 Vehicle {target.target_id}:")
                        print(f"      Camera: bbox=({cam.bbox[0]}, {cam.bbox[1]}, {cam.bbox[2]}, {cam.bbox[3]})")
                        print(f"      License: {cam.license_plate or 'Not detected'}")
                        
                        if radar:
                            print(f"      Radar: pos=({radar.x:.2f}, {radar.y:.2f})m")
                            print(f"      Speed: {radar.speed_kmh:.1f} km/h")
                            print(f"      Range: {radar.range_m:.1f}m, Azimuth: {radar.azimuth_deg:.1f}°")
                        else:
                            print(f"      Speed: No radar data")
                        
                        # Log detection
                        detection = {
                            'frame': self.frame_count,
                            'target_id': target.target_id,
                            'license_plate': cam.license_plate,
                            'speed_kmh': target.speed_kmh,
                            'position': target.position_m,
                            'timestamp': target.timestamp
                        }
                        self.detections_log.append(detection)
                
                # Add text overlay for sensor fusion info
                self._add_sensor_fusion_overlay(frame_meta, camera_targets)
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
    
    def _extract_camera_targets(self, frame_meta) -> List[CameraTarget]:
        """Extract vehicle detections from camera"""
        targets = []
        
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Check if it's a vehicle detection (car, truck, bus, etc.)
                if obj_meta.class_id in [0, 2, 5, 7]:  # COCO classes for vehicles
                    
                    # Extract bounding box
                    bbox = (
                        int(obj_meta.rect_params.left),
                        int(obj_meta.rect_params.top), 
                        int(obj_meta.rect_params.width),
                        int(obj_meta.rect_params.height)
                    )
                    
                    # Extract license plate if available (from SGIE3)
                    license_plate = None
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                            if classifier_meta.unique_component_id == 3:  # SGIE3 - LPR
                                if classifier_meta.label_info_list:
                                    label_info = pyds.NvDsLabelInfo.cast(classifier_meta.label_info_list.data)
                                    license_plate = label_info.result_label
                                    break
                        except StopIteration:
                            break
                        try:
                            l_classifier = l_classifier.next
                        except StopIteration:
                            break
                    
                    target = CameraTarget(
                        frame_id=frame_meta.frame_num,
                        bbox=bbox,
                        confidence=obj_meta.confidence,
                        license_plate=license_plate,
                        timestamp=time.time()
                    )
                    targets.append(target)
                
            except StopIteration:
                break
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        return targets
    
    def _add_sensor_fusion_overlay(self, frame_meta, targets: List[CameraTarget]):
        """Add sensor fusion information to video overlay"""
        
        # Get radar object count
        radar_objects = self.fusion.get_radar_objects()
        
        # Create display meta
        display_meta = pyds.nvds_acquire_display_meta_from_pool(frame_meta.batch_meta)
        
        # Add radar status text
        txt_params = display_meta.text_params[0]
        txt_params.display_text = f"Radar: {len(radar_objects)} objects"
        txt_params.x_offset = 10
        txt_params.y_offset = 10  
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 12
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        display_meta.num_labels = 1
        
        # Add camera status text
        if len(targets) > 0:
            txt_params2 = display_meta.text_params[1]
            txt_params2.display_text = f"Camera: {len(targets)} vehicles"
            txt_params2.x_offset = 10
            txt_params2.y_offset = 35
            txt_params2.font_params.font_name = "Serif"
            txt_params2.font_params.font_size = 12
            txt_params2.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
            txt_params2.set_bg_clr = 1
            txt_params2.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            display_meta.num_labels = 2
        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    
    def bus_call(self, bus, message):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("📺 End-of-stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ Pipeline error: {err}")
            self.loop.quit()
        return True
    
    def run(self):
        """Run the complete sensor fusion system"""
        print("🚀 Starting SafeSpeed AI Sensor Fusion System")
        
        # Start radar subsystem
        if not self.fusion.start_radar():
            print("❌ Failed to start radar - running camera only")
        
        # Create and start pipeline
        self.create_pipeline()
        
        print("▶️  Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Unable to set the pipeline to playing state")
            return
        
        # Run main loop
        try:
            self.loop = GObject.MainLoop()
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        # Cleanup
        print("🧹 Cleaning up...")
        self.pipeline.set_state(Gst.State.NULL)
        self.fusion.stop_radar()
        
        # Print summary
        print(f"\n📊 Session Summary:")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Detections logged: {len(self.detections_log)}")
        
        if self.detections_log:
            speeds = [d['speed_kmh'] for d in self.detections_log if d['speed_kmh'] > 0]
            if speeds:
                print(f"   Average speed: {np.mean(speeds):.1f} km/h")
                print(f"   Max speed: {max(speeds):.1f} km/h")

def main():
    """Main entry point"""
    fusion_system = SafeSpeedAISensorFusion()
    fusion_system.run()

if __name__ == "__main__":
    main()
