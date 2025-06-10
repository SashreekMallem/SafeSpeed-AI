#!/usr/bin/env python3

"""
Corner Reflector Calibration System for AWR1843BOOST + Camera Fusion
Implements research-based calibration using target-based approach with PnP solving
"""

import cv2
import numpy as np
import time
import threading
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.append('/home/projecta/SafespeedAI')
from awr1843_interface import AWR1843Interface

class CornerReflectorCalibration:
    """Research-based calibration using corner reflectors for radar-camera fusion"""
    
    def __init__(self):
        self.radar = AWR1843Interface()
        self.camera = None
        self.calibration_data = []
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Calibration parameters
        self.min_calibration_points = 10  # Research: Multiple target positions required
        self.detection_timeout = 5.0  # Seconds to wait for stable detection
        
    def initialize_systems(self):
        """Initialize radar and camera systems"""
        print("🔧 Initializing Calibration Systems...")
        
        # Initialize radar
        if not self.radar.connect():
            print("❌ Failed to connect to radar")
            return False
            
        if not self.radar.configure():
            print("❌ Failed to configure radar")
            return False
            
        print("✅ Radar initialized")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("❌ Failed to open camera")
            return False
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        return True
    
    def calibrate_camera_intrinsics(self):
        """Calibrate camera intrinsic parameters using chessboard pattern"""
        print("\n📐 Camera Intrinsic Calibration")
        print("=" * 50)
        print("Instructions:")
        print("1. Print a chessboard pattern (9x6 squares)")
        print("2. Hold it flat in front of the camera")
        print("3. Move it to different positions and angles")
        print("4. Press SPACE to capture, ESC to finish")
        
        # Chessboard dimensions
        pattern_size = (9, 6)
        square_size = 1.0  # Can be any unit, we'll use relative measurements
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        print("\n📸 Collecting calibration images...")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_chess, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            display_frame = frame.copy()
            
            if ret_chess:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners
                cv2.drawChessboardCorners(display_frame, pattern_size, corners2, ret_chess)
                cv2.putText(display_frame, "Press SPACE to capture", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Chessboard not detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(display_frame, f"Captured: {len(objpoints)}/15", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, "ESC to finish calibration", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and ret_chess:  # Space to capture
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"✅ Captured image {len(objpoints)}")
                
            elif key == 27:  # ESC to finish
                break
        
        cv2.destroyAllWindows()
        
        if len(objpoints) < 5:
            print("❌ Need at least 5 calibration images")
            return False
        
        print(f"\n🔍 Calculating camera parameters from {len(objpoints)} images...")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
            print("✅ Camera calibration successful!")
            print(f"Camera Matrix:\n{camera_matrix}")
            print(f"Distortion Coefficients: {dist_coeffs.flatten()}")
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            mean_error /= len(objpoints)
            print(f"Mean reprojection error: {mean_error:.3f} pixels")
            
            # Save calibration
            calib_data = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'reprojection_error': mean_error,
                'calibration_date': datetime.now().isoformat()
            }
            
            with open('/home/projecta/SafespeedAI/camera_intrinsics.json', 'w') as f:
                json.dump(calib_data, f, indent=2)
            
            print("💾 Camera calibration saved to camera_intrinsics.json")
            return True
        else:
            print("❌ Camera calibration failed")
            return False
    
    def collect_corner_reflector_data(self):
        """Collect corner reflector calibration data points"""
        print("\n🎯 Corner Reflector Calibration Data Collection")
        print("=" * 60)
        print("Instructions:")
        print("1. Place a large metallic corner reflector in view of both sensors")
        print("2. Position it at different locations (at least 10 positions)")
        print("3. For each position, click on the reflector in the camera image")
        print("4. System will record radar detection and camera pixel coordinates")
        print("5. Press 'c' to capture point, 'f' to finish")
        
        calibration_points = []
        current_radar_objects = []
        
        def radar_thread():
            """Background thread to collect radar data"""
            nonlocal current_radar_objects
            while True:
                objects = self.radar.read_frame(timeout=0.5)
                if objects:
                    current_radar_objects = objects
                time.sleep(0.1)
        
        # Start radar data collection thread
        radar_thread_obj = threading.Thread(target=radar_thread, daemon=True)
        radar_thread_obj.start()
        
        # Mouse callback for clicking on reflector
        clicked_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_point
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point = (x, y)
        
        cv2.namedWindow('Corner Reflector Calibration')
        cv2.setMouseCallback('Corner Reflector Calibration', mouse_callback)
        
        print("\n📍 Collecting calibration points...")
        
        while len(calibration_points) < 20:  # Collect up to 20 points
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            
            # Show radar detections as overlays
            for obj in current_radar_objects:
                # Simple coordinate mapping for visualization
                img_x = int(640 + obj.x * 100)
                img_y = int(360 - obj.y * 50)
                if 0 < img_x < 1280 and 0 < img_y < 720:
                    cv2.circle(display_frame, (img_x, img_y), 10, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"R{len(current_radar_objects)}", 
                               (img_x + 15, img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Show collected points
            for i, point in enumerate(calibration_points):
                cv2.circle(display_frame, (int(point['image_x']), int(point['image_y'])), 
                          5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{i+1}", 
                           (int(point['image_x']) + 10, int(point['image_y'])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Instructions overlay
            cv2.putText(display_frame, f"Points collected: {len(calibration_points)}/10 minimum", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Click on corner reflector, press 'c' to capture", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Radar objects: {len(current_radar_objects)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if clicked_point:
                cv2.circle(display_frame, clicked_point, 8, (255, 0, 0), 2)
                cv2.putText(display_frame, "Press 'c' to confirm this point", 
                           (clicked_point[0] + 10, clicked_point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.imshow('Corner Reflector Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and clicked_point and len(current_radar_objects) > 0:
                # Find closest radar object to clicked point
                best_radar_obj = None
                min_distance = float('inf')
                
                for obj in current_radar_objects:
                    img_x = int(640 + obj.x * 100)
                    img_y = int(360 - obj.y * 50)
                    distance = np.sqrt((clicked_point[0] - img_x)**2 + (clicked_point[1] - img_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_radar_obj = obj
                
                if best_radar_obj and min_distance < 100:  # Within 100 pixels
                    calib_point = {
                        'point_id': len(calibration_points) + 1,
                        'image_x': clicked_point[0],
                        'image_y': clicked_point[1],
                        'radar_x': best_radar_obj.x,
                        'radar_y': best_radar_obj.y,
                        'radar_z': best_radar_obj.z,
                        'radar_range': best_radar_obj.range_m,
                        'radar_azimuth': best_radar_obj.azimuth_deg,
                        'timestamp': time.time()
                    }
                    
                    calibration_points.append(calib_point)
                    print(f"✅ Point {len(calibration_points)}: Image({clicked_point[0]}, {clicked_point[1]}) -> Radar({best_radar_obj.x:.2f}, {best_radar_obj.y:.2f})")
                    clicked_point = None
                else:
                    print("❌ No radar object near clicked point")
            
            elif key == ord('f') and len(calibration_points) >= self.min_calibration_points:
                break
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        if len(calibration_points) >= self.min_calibration_points:
            print(f"✅ Collected {len(calibration_points)} calibration points")
            self.calibration_data = calibration_points
            
            # Save calibration data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'/home/projecta/SafespeedAI/corner_reflector_calibration_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump({
                    'calibration_points': calibration_points,
                    'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                    'distortion_coefficients': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
                    'collection_date': datetime.now().isoformat(),
                    'num_points': len(calibration_points)
                }, f, indent=2)
            
            print(f"💾 Calibration data saved to: {filename}")
            return True
        else:
            print(f"❌ Need at least {self.min_calibration_points} points, got {len(calibration_points)}")
            return False
    
    def calculate_extrinsic_parameters(self):
        """Calculate extrinsic parameters using PnP solving"""
        if not self.calibration_data or self.camera_matrix is None:
            print("❌ Need calibration data and camera matrix")
            return False
        
        print("\n🧮 Calculating Extrinsic Parameters using PnP Solving...")
        
        # Prepare points for PnP solving
        object_points = []  # 3D points in radar coordinate system
        image_points = []   # 2D points in camera image
        
        for point in self.calibration_data:
            # Radar coordinates (3D)
            object_points.append([point['radar_x'], point['radar_y'], point['radar_z']])
            # Camera coordinates (2D)
            image_points.append([point['image_x'], point['image_y']])
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        print(f"Solving PnP with {len(object_points)} point correspondences...")
        
        # Solve PnP problem using RANSAC for robustness
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.camera_matrix, self.dist_coeffs,
            confidence=0.99, reprojectionError=5.0
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            print("✅ Extrinsic calibration successful!")
            print(f"Inliers: {len(inliers) if inliers is not None else 0}/{len(object_points)}")
            print(f"Translation Vector (radar to camera):\n{translation_vector.flatten()}")
            print(f"Rotation Matrix:\n{rotation_matrix}")
            
            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                object_points, rotation_vector, translation_vector, 
                self.camera_matrix, self.dist_coeffs
            )
            
            errors = []
            for i in range(len(image_points)):
                error = np.linalg.norm(image_points[i] - projected_points[i].flatten())
                errors.append(error)
            
            mean_error = np.mean(errors)
            print(f"Mean reprojection error: {mean_error:.2f} pixels")
            
            # Save complete calibration
            calibration_result = {
                'extrinsic_calibration': {
                    'translation_vector': translation_vector.flatten().tolist(),
                    'rotation_vector': rotation_vector.flatten().tolist(),
                    'rotation_matrix': rotation_matrix.tolist(),
                    'mean_reprojection_error': float(mean_error),
                    'inlier_count': len(inliers) if inliers is not None else len(object_points),
                    'total_points': len(object_points)
                },
                'intrinsic_calibration': {
                    'camera_matrix': self.camera_matrix.tolist(),
                    'distortion_coefficients': self.dist_coeffs.tolist()
                },
                'calibration_points': self.calibration_data,
                'calibration_complete_date': datetime.now().isoformat()
            }
            
            with open('/home/projecta/SafespeedAI/complete_calibration.json', 'w') as f:
                json.dump(calibration_result, f, indent=2)
            
            print("💾 Complete calibration saved to complete_calibration.json")
            print("🎉 Calibration procedure complete!")
            
            return True
        else:
            print("❌ PnP solving failed")
            return False
    
    def run_full_calibration(self):
        """Run complete calibration procedure"""
        print("🎯" * 30)
        print("🎯  Research-Based Calibration System  🎯")
        print("🎯" * 30)
        
        if not self.initialize_systems():
            return False
        
        # Step 1: Camera intrinsic calibration
        print("\nStep 1: Camera Intrinsic Calibration")
        if not self.calibrate_camera_intrinsics():
            print("❌ Camera calibration failed")
            return False
        
        # Step 2: Corner reflector data collection
        print("\nStep 2: Corner Reflector Data Collection")
        if not self.collect_corner_reflector_data():
            print("❌ Data collection failed")
            return False
        
        # Step 3: Extrinsic parameter calculation
        print("\nStep 3: Extrinsic Parameter Calculation")
        if not self.calculate_extrinsic_parameters():
            print("❌ Extrinsic calibration failed")
            return False
        
        print("\n🎉 Complete Calibration Successful!")
        print("✅ Camera intrinsic parameters calculated")
        print("✅ Radar-camera extrinsic parameters calculated") 
        print("✅ Ready for accurate sensor fusion")
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()
        self.radar.disconnect()
        cv2.destroyAllWindows()


def main():
    """Main calibration function"""
    calibrator = CornerReflectorCalibration()
    
    try:
        success = calibrator.run_full_calibration()
        
        if success:
            print("\n🎯 Calibration Complete - Ready for Production Fusion!")
        else:
            print("\n❌ Calibration Failed - Check setup and try again")
            
    except KeyboardInterrupt:
        print("\n⏹️  Calibration interrupted by user")
    except Exception as e:
        print(f"\n❌ Calibration error: {e}")
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()
