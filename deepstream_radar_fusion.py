#!/usr/bin/env python3

"""
Enhanced SafeSpeed AI with AWR1843 Radar Fusion
Integrates mmWave radar speed detection with camera-based license plate recognition
"""

import sys
sys.path.append('../')
import configparser
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

import pyds
import collections
import cv2
import numpy as np
import threading
import time

# Import our radar fusion system
from radar_fusion_system import RadarFusionPipeline, CameraTarget, FusedTarget

# Existing constants
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000

# Global radar fusion pipeline
radar_fusion = None
use_radar_fusion = True

# Initialize a deque to store the last few detection results for smoothing
detection_history = collections.deque(maxlen=5)
speed_history = collections.deque(maxlen=10)  # Store speed measurements

def init_radar_fusion():
    """Initialize radar fusion system"""
    global radar_fusion, use_radar_fusion
    
    try:
        print("Initializing AWR1843 Radar Fusion...")
        radar_fusion = RadarFusionPipeline()
        
        if radar_fusion.initialize():
            print("✅ Radar fusion initialized successfully")
            use_radar_fusion = True
        else:
            print("⚠️  Radar not available, running camera-only mode")
            use_radar_fusion = False
            radar_fusion = None
            
    except Exception as e:
        print(f"❌ Radar initialization failed: {e}")
        print("Continuing with camera-only mode")
        use_radar_fusion = False
        radar_fusion = None

def smooth_speed_measurements(new_speeds):
    """Smooth speed measurements to reduce noise"""
    global speed_history
    
    # Add new speeds to history
    speed_history.extend(new_speeds)
    
    if len(speed_history) < 3:
        return new_speeds
    
    # Calculate moving average for smoothing
    recent_speeds = list(speed_history)[-5:]  # Last 5 measurements
    if recent_speeds:
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        # Return smoothed speed if within reasonable range of current measurements
        if any(abs(speed - avg_speed) < 5 for speed in new_speeds):  # Within 5 mph
            return [avg_speed]
    
    return new_speeds

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Enhanced buffer probe with radar fusion"""
    global radar_fusion, use_radar_fusion
    
    frame_number = 0
    # Initialize object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,
        PGIE_CLASS_ID_ROADSIGN: 0
    }
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # Get frame data for processing
    frame_data = None
    try:
        # Get the frame data from the buffer
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), 0)
        frame_data = np.array(n_frame, copy=True, order='C')
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2BGR)
    except Exception as e:
        print(f"Error extracting frame data: {e}")
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        
        # Collect camera detections for radar fusion
        camera_targets = []
        license_plates_detected = []
        
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
                
            obj_counter[obj_meta.class_id] += 1
            
            # Create camera target for radar fusion
            if obj_meta.class_id == PGIE_CLASS_ID_VEHICLE:
                camera_target = CameraTarget(
                    bbox=(
                        int(obj_meta.rect_params.left),
                        int(obj_meta.rect_params.top),
                        int(obj_meta.rect_params.width),
                        int(obj_meta.rect_params.height)
                    ),
                    license_plate=None,  # Will be filled by license plate detection
                    vehicle_type=getattr(obj_meta, 'obj_label', 'vehicle'),
                    confidence=obj_meta.confidence,
                    timestamp=time.time(),
                    track_id=getattr(obj_meta, 'object_id', None)
                )
                camera_targets.append(camera_target)
            
            # Check for license plate detections (SGIE3)
            if hasattr(obj_meta, 'unique_component_id') and obj_meta.unique_component_id == 4:
                if frame_data is not None:
                    # Extract license plate text using existing OCR
                    license_text = extract_license_plate_text(frame_data, obj_meta)
                    if license_text:
                        license_plates_detected.append(license_text)
                        # Update corresponding camera target with license plate
                        for cam_target in camera_targets:
                            if cam_target.license_plate is None:
                                cam_target.license_plate = license_text
                                break
            
            try: 
                l_obj = l_obj.next
            except StopIteration:
                break

        # Process with radar fusion if available
        speed_info = ""
        fused_results = []
        
        if use_radar_fusion and radar_fusion and camera_targets:
            try:
                radar_fusion.process_camera_detections(camera_targets)
                fused_results = radar_fusion.get_fused_results()
                
                if fused_results:
                    speeds = [result.speed_mph for result in fused_results]
                    smoothed_speeds = smooth_speed_measurements(speeds)
                    
                    # Create speed info string
                    if smoothed_speeds:
                        avg_speed = sum(smoothed_speeds) / len(smoothed_speeds)
                        speed_info = f" Speed: {avg_speed:.1f}mph"
                        
                        # Log detailed fusion results
                        for result in fused_results:
                            print(f"FUSED: LP={result.license_plate}, Speed={result.speed_mph:.1f}mph, Range={result.range_m:.1f}m")
                
            except Exception as e:
                print(f"Radar fusion error: {e}")

        # Create license plate info
        license_info = ""
        if license_plates_detected:
            license_info = f" LP: {', '.join(license_plates_detected)}"

        # Add the current detection results to the history
        detection_history.append(obj_counter.copy())

        # Calculate the average detection results over the history
        avg_obj_counter = {
            PGIE_CLASS_ID_VEHICLE: sum(d[PGIE_CLASS_ID_VEHICLE] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_PERSON: sum(d[PGIE_CLASS_ID_PERSON] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_BICYCLE: sum(d[PGIE_CLASS_ID_BICYCLE] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_ROADSIGN: sum(d[PGIE_CLASS_ID_ROADSIGN] for d in detection_history) / len(detection_history)
        }

        # Acquiring a display meta object
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
        # Enhanced display text with speed information
        display_text = "Frame={} Objects={} Vehicles={:.1f} Persons={:.1f}{}{}".format(
            frame_number, num_rects, 
            avg_obj_counter[PGIE_CLASS_ID_VEHICLE], 
            avg_obj_counter[PGIE_CLASS_ID_PERSON],
            license_info, speed_info
        )
        
        py_nvosd_text_params.display_text = display_text

        # Set display properties
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Past tracking metadata (existing code)
    l_user = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if(user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
                        pass  # Simplified past frame processing
        try:
            l_user = l_user.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

# Placeholder for existing license plate extraction function
def extract_license_plate_text(frame_data, obj_meta):
    """Placeholder for existing license plate OCR function"""
    # This would be your existing OCR implementation
    return "ABC123"  # Placeholder

def main(args):
    """Enhanced main function with radar fusion"""
    
    # Initialize radar fusion first
    init_radar_fusion()
    
    # Check input arguments
    if len(args) != 3:
        sys.stderr.write("usage: %s <source-type> <source-path>\n" % args[0])
        sys.stderr.write("source-type: 'camera' or 'file'\n")
        sys.exit(1)

    source_type = args[1]
    source_path = args[2]

    platform_info = PlatformInfo()
    Gst.init(None)

    print("Creating Enhanced SafeSpeed AI Pipeline with Radar Fusion")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Create elements (same as your existing pipeline)
    if source_type == 'camera':
        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        source.set_property('device', source_path)
    elif source_type == 'file':
        source = Gst.ElementFactory.make("filesrc", "file-source")
        decodebin = Gst.ElementFactory.make("decodebin", "decoder")
        source.set_property('location', source_path)

    # Create remaining pipeline elements (abbreviated for clarity)
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-nvinference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    # Create sink
    if platform_info.is_integrated_gpu():
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    else:
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    # Set properties
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    
    pgie.set_property('config-file-path', "dstest2_pgie_config.txt")
    sgie1.set_property('config-file-path', "dstest2_sgie1_config.txt")
    sgie2.set_property('config-file-path', "dstest2_sgie2_config.txt")
    sgie3.set_property('config-file-path', "dstest2_sgie3_config.txt")
    
    # Configure tracker
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    for key in config['tracker']:
        if key == 'tracker-width':
            tracker.set_property('tracker-width', config.getint('tracker', key))
        elif key == 'tracker-height':
            tracker.set_property('tracker-height', config.getint('tracker', key))
        elif key == 'gpu-id':
            tracker.set_property('gpu_id', config.getint('tracker', key))
        elif key == 'll-lib-file':
            tracker.set_property('ll-lib-file', config.get('tracker', key))
        elif key == 'll-config-file':
            tracker.set_property('ll-config-file', config.get('tracker', key))

    sink.set_property('sync', False)

    # Add elements to pipeline
    elements = [source, streammux, pgie, tracker, sgie1, sgie2, sgie3, nvvidconv, nvosd, sink]
    if source_type == 'file':
        elements.insert(1, decodebin)
    elif source_type == 'camera':
        elements.insert(1, caps_v4l2src)
        
    for element in elements:
        pipeline.add(element)

    # Link elements
    if source_type == 'camera':
        source.link(caps_v4l2src)
        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad = caps_v4l2src.get_static_pad("src")
        srcpad.link(sinkpad)
    else:
        source.link(decodebin)
        decodebin.connect("pad-added", lambda decodebin, pad, streammux: 
                         pad.link(streammux.get_request_pad("sink_0")))

    # Link the main pipeline
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(sgie3)
    sgie3.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Create event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add probe to OSD
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start pipeline
    print("Starting Enhanced SafeSpeed AI Pipeline with Radar Fusion")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        # Cleanup
        if radar_fusion:
            radar_fusion.stop()
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
