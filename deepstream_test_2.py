#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

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
import easyocr
import threading
import pytesseract
import time

# Initialize OCR readers (thread-safe)
ocr_reader = None
ocr_lock = threading.Lock()
use_tesseract = False  # Flag to switch between EasyOCR and Tesseract

def init_ocr():
    global ocr_reader, use_tesseract
    if ocr_reader is None:
        try:
            print("Initializing EasyOCR reader...")
            ocr_reader = easyocr.Reader(['en'], gpu=True)
            use_tesseract = False
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            print("Falling back to Tesseract OCR")
            use_tesseract = True

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000

# Initialize a deque to store the last few detection results for smoothing
detection_history = collections.deque(maxlen=5)
license_plate_history = collections.deque(maxlen=10)  # Store recent license plate detections

def smooth_license_plate_results(current_plates):
    """Smooth license plate results to reduce flickering"""
    global license_plate_history
    
    # Add current detections to history
    license_plate_history.extend(current_plates)
    
    # Count occurrences of each license plate
    plate_counts = {}
    for plate in license_plate_history:
        plate_counts[plate] = plate_counts.get(plate, 0) + 1
    
    # Return plates that appear frequently (at least 3 times in recent history)
    frequent_plates = [plate for plate, count in plate_counts.items() if count >= 3]
    
    return frequent_plates

def enhance_license_plate_image(image):
    """Apply image enhancements for better OCR"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    
    # Additional sharpening for better character recognition
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(morphed, -1, kernel_sharp)
    
    return sharpened

def extract_license_plate_text_tesseract(enhanced_crop):
    """Extract text using Tesseract OCR"""
    try:
        # Configure Tesseract for license plate recognition
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Get text from Tesseract
        text = pytesseract.image_to_string(enhanced_crop, config=config)
        text = text.strip().replace(' ', '').replace('\n', '').upper()
        
        # Filter out very short or long results
        if len(text) >= 3 and len(text) <= 10 and text.isalnum():
            return text
        
        return ""
        
    except Exception as e:
        print(f"Tesseract OCR error: {e}")
        return ""

def extract_license_plate_text_easyocr(enhanced_crop):
    """Extract text using EasyOCR"""
    global ocr_reader, ocr_lock
    
    try:
        # Perform OCR with thread safety
        with ocr_lock:
            results = ocr_reader.readtext(enhanced_crop, 
                                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                        paragraph=False,
                                        width_ths=0.9,
                                        height_ths=0.7)
        
        # Extract text with confidence filtering
        license_texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5 and len(text.strip()) > 2:  # Filter out low confidence and short texts
                clean_text = text.strip().replace(' ', '').upper()
                license_texts.append((clean_text, confidence))
        
        if license_texts:
            # Return the highest confidence text
            best_text = max(license_texts, key=lambda x: x[1])
            return best_text[0]
        
        return ""
        
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""

def extract_license_plate_text(frame_data, obj_meta):
    """Extract and OCR license plate text from detected regions"""
    global use_tesseract
    
    try:
        # Initialize OCR if not done
        if not use_tesseract and ocr_reader is None:
            with ocr_lock:
                init_ocr()
        
        # Get frame dimensions
        frame_height = frame_data.shape[0]
        frame_width = frame_data.shape[1]
        
        # Get bounding box coordinates
        left = int(obj_meta.rect_params.left)
        top = int(obj_meta.rect_params.top) 
        width = int(obj_meta.rect_params.width)
        height = int(obj_meta.rect_params.height)
        
        # Add some padding to capture full license plate
        padding = 5
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(frame_width, left + width + 2 * padding)
        bottom = min(frame_height, top + height + 2 * padding)
        
        if right <= left or bottom <= top:
            return ""
        
        # Extract license plate region
        lp_crop = frame_data[top:bottom, left:right]
        
        if lp_crop.size == 0:
            return ""
        
        # Resize for better OCR (minimum 200px width, maximum 800px)
        if lp_crop.shape[1] < 200:
            scale_factor = 200 / lp_crop.shape[1]
            new_width = int(lp_crop.shape[1] * scale_factor)
            new_height = int(lp_crop.shape[0] * scale_factor)
            lp_crop = cv2.resize(lp_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif lp_crop.shape[1] > 800:
            scale_factor = 800 / lp_crop.shape[1]
            new_width = int(lp_crop.shape[1] * scale_factor)
            new_height = int(lp_crop.shape[0] * scale_factor)
            lp_crop = cv2.resize(lp_crop, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Enhance the image
        enhanced_crop = enhance_license_plate_image(lp_crop)
        
        # Try OCR with selected engine
        if use_tesseract:
            return extract_license_plate_text_tesseract(enhanced_crop)
        else:
            return extract_license_plate_text_easyocr(enhanced_crop)
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return ""

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    # Get frame data for OCR processing
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
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        
        # Track license plates detected in this frame
        license_plates_detected = []
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            
            # Debug: Print all object metadata for troubleshooting
            if frame_number % 30 == 0:  # Print debug info every 30 frames to avoid spam
                print(f"Frame {frame_number} - Object: class_id={obj_meta.class_id}, conf={obj_meta.confidence:.3f}, unique_id={getattr(obj_meta, 'unique_component_id', 'N/A')}")
            
            # Check if this is a license plate detection (from SGIE3 - LPD)
            # SGIE3 has unique_component_id = 4 and detects license plates
            if hasattr(obj_meta, 'unique_component_id') and obj_meta.unique_component_id == 4:  # SGIE3 ID
                print(f"License Plate Detection - Class ID: {obj_meta.class_id}, Confidence: {obj_meta.confidence}")
                if frame_data is not None:
                    # Extract license plate text using OCR
                    license_text = extract_license_plate_text(frame_data, obj_meta)
                    if license_text:
                        license_plates_detected.append(license_text)
                        print(f"License Plate Detected: {license_text}")
            # Also check for any objects with parent that might be license plates
            elif hasattr(obj_meta, 'parent') and obj_meta.parent:
                # Check if parent is a vehicle and this could be a license plate
                parent_obj = obj_meta.parent
                if hasattr(parent_obj, 'class_id') and parent_obj.class_id == PGIE_CLASS_ID_VEHICLE:
                    print(f"Potential LP from parent - Class ID: {obj_meta.class_id}, Confidence: {obj_meta.confidence}")
                    if frame_data is not None and obj_meta.confidence > 0.3:  # Lower threshold for secondary detections
                        license_text = extract_license_plate_text(frame_data, obj_meta)
                        if license_text:
                            license_plates_detected.append(license_text)
                            print(f"License Plate from Parent Detected: {license_text}")
            
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Add license plate info to display if any detected
        license_info = ""
        if license_plates_detected:
            # Apply smoothing to reduce flickering
            smooth_plates = smooth_license_plate_results(license_plates_detected)
            if smooth_plates:
                license_info = f" LP: {', '.join(smooth_plates)}"
            else:
                # Show current detection but with indicator it's not confirmed
                license_info = f" LP?: {', '.join(license_plates_detected)}"

        # Add the current detection results to the history
        detection_history.append(obj_counter.copy())

        # Calculate the average detection results over the history
        avg_obj_counter = {
            PGIE_CLASS_ID_VEHICLE: sum(d[PGIE_CLASS_ID_VEHICLE] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_PERSON: sum(d[PGIE_CLASS_ID_PERSON] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_BICYCLE: sum(d[PGIE_CLASS_ID_BICYCLE] for d in detection_history) / len(detection_history),
            PGIE_CLASS_ID_ROADSIGN: sum(d[PGIE_CLASS_ID_ROADSIGN] for d in detection_history) / len(detection_history)
        }

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}{}".format(frame_number, num_rects, avg_obj_counter[PGIE_CLASS_ID_VEHICLE], avg_obj_counter[PGIE_CLASS_ID_PERSON], license_info)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break		
    #past tracking meta data
    l_user=batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            # Note that l_user.data needs a cast to pyds.NvDsUserMeta
            # The casting is done by pyds.NvDsUserMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone
            user_meta=pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
                # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                print("streamId=",miscDataStream.streamID)
                print("surfaceStreamID=",miscDataStream.surfaceStreamID)
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    print("numobj=",miscDataObj.numObj)
                    print("uniqueId=",miscDataObj.uniqueId)
                    print("classId=",miscDataObj.classId)
                    print("objLabel=",miscDataObj.objLabel)
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
                        print('frameNum:', miscDataFrame.frameNum)
                        print('tBbox.left:', miscDataFrame.tBbox.left)
                        print('tBbox.width:', miscDataFrame.tBbox.width)
                        print('tBbox.top:', miscDataFrame.tBbox.top)
                        print('tBbox.right:', miscDataFrame.tBbox.height)
                        print('confidence:', miscDataFrame.confidence)
                        print('age:', miscDataFrame.age)
        try:
            l_user=l_user.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK	
	


def main(args):
    # Check input arguments
    if len(args) != 3:
        sys.stderr.write("usage: %s <source-type> <source-path>\n" % args[0])
        sys.stderr.write("source-type: 'camera' or 'file'\n")
        sys.exit(1)

    source_type = args[1]
    source_path = args[2]

    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    # Set debug level for specific elements
    Gst.debug_set_threshold_for_name("nvinfer", Gst.DebugLevel.DEBUG)
    Gst.debug_set_threshold_for_name("nvtracker", Gst.DebugLevel.DEBUG)
    Gst.debug_set_threshold_for_name("customparser", Gst.DebugLevel.DEBUG)  # Add this line

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    if source_type == 'camera':
        # Source element for reading from the USB camera
        print("Creating USB Camera Source \n ")
        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        if not source:
            sys.stderr.write(" Unable to create USB Camera Source \n")

        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        if not caps_v4l2src:
            sys.stderr.write(" Unable to create v4l2src capsfilter \n")

        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        source.set_property('device', source_path)

    elif source_type == 'file':
        # Source element for reading from the video file
        print("Creating Video File Source \n ")
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            sys.stderr.write(" Unable to create File Source \n")

        decodebin = Gst.ElementFactory.make("decodebin", "decoder")
        if not decodebin:
            sys.stderr.write(" Unable to create Decodebin \n")

        source.set_property('location', source_path)

    else:
        sys.stderr.write("Invalid source type. Use 'camera' or 'file'.\n")
        sys.exit(1)

    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)

    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")

    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary1-nvinference-engine")
    if not sgie1:
        sys.stderr.write(" Unable to make sgie1 \n")


    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie2:
        sys.stderr.write(" Unable to make sgie2 \n")
        
    sgie3 = Gst.ElementFactory.make("nvinfer", "secondary3-nvinference-engine")
    if not sgie3:
        sys.stderr.write(" Unable to make sgie3 \n")
        
    # Remove SGIE4 - we'll use OCR on license plate crops instead
        

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    if platform_info.is_integrated_gpu():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

    print("Playing from %s source %s" % (source_type, source_path))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    #Set properties of pgie and sgie
    pgie.set_property('config-file-path', "dstest2_pgie_config.txt")
    sgie1.set_property('config-file-path', "dstest2_sgie1_config.txt")
    sgie2.set_property('config-file-path', "dstest2_sgie2_config.txt")
    sgie3.set_property('config-file-path', "dstest2_sgie3_config.txt")


    
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('dstest2_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
    # Set sync = false to avoid late frame drops at the display-sink
    sink.set_property('sync', False)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    if source_type == 'camera':
        pipeline.add(caps_v4l2src)
        pipeline.add(vidconvsrc)
        pipeline.add(nvvidconvsrc)
        pipeline.add(caps_vidconvsrc)
    else:
        pipeline.add(decodebin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie1)
    pipeline.add(sgie2)
    pipeline.add(sgie3)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # we link the elements together
    # v4l2src -> nvvideoconvert -> mux -> 
    # nvinfer -> nvvideoconvert -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    if source_type == 'camera':
        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)

        sinkpad = streammux.request_pad_simple("sink_0")
        if not sinkpad:
            sys.stderr.write(" Unable to get the sink pad of streammux \n")
        srcpad = caps_vidconvsrc.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
        srcpad.link(sinkpad)
    else:
        source.link(decodebin)
        decodebin.connect("pad-added", decodebin_pad_added, streammux)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(sgie3)
    sgie3.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def decodebin_pad_added(decodebin, pad, streammux):
    caps = pad.get_current_caps()
    gstname = caps.get_structure(0).get_name()
    if gstname.find("video") != -1:
        pad.link(streammux.get_request_pad("sink_0"))

if __name__ == '__main__':
    sys.exit(main(sys.argv))

