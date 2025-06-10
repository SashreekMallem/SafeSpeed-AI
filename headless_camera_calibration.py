#!/usr/bin/env python3
"""
Headless Camera Calibration System
==================================

Automated camera intrinsic calibration without GUI dependencies.
Captures images automatically and processes them for calibration.

Usage:
  python3 headless_camera_calibration.py --capture    # Capture calibration images
  python3 headless_camera_calibration.py --process    # Process saved images
  python3 headless_camera_calibration.py --both       # Do both (default)
"""

import cv2
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
import glob

class HeadlessCameraCalibrator:
    def __init__(self):
        # Chessboard dimensions (internal corners)
        self.CHESSBOARD_SIZE = (9, 6)  # 9x6 internal corners
        self.SQUARE_SIZE = 25.0  # mm, adjust based on your printed pattern
        
        # Calibration data
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        
        # Camera setup
        self.camera = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Create calibration directory
        self.calib_dir = "calibration_images"
        os.makedirs(self.calib_dir, exist_ok=True)
        
        print("🎯 Headless Camera Calibration System")
        print("=" * 50)
        
    def init_camera(self):
        """Initialize camera"""
        print("📸 Initializing camera...")
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("❌ Failed to open camera")
        
        # Set camera properties for better quality
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        
    def prepare_object_points(self):
        """Prepare 3D object points"""
        objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= self.SQUARE_SIZE
        return objp
        
    def capture_calibration_images(self, num_images=30, interval=2.0):
        """
        Automatically capture calibration images
        
        Args:
            num_images: Number of images to capture
            interval: Seconds between captures
        """
        print(f"📸 Capturing {num_images} calibration images...")
        print(f"⏱️  Image capture interval: {interval} seconds")
        print("\nInstructions:")
        print("1. Print a 9x6 chessboard pattern")
        print("2. Hold it steady in front of the camera")
        print("3. Move to different positions and angles between captures")
        print("4. Ensure good lighting and sharp focus")
        print("\nStarting capture in 5 seconds...")
        
        for i in range(5):
            print(f"Starting in {5-i}...")
            time.sleep(1)
        
        self.init_camera()
        
        captured = 0
        failed = 0
        
        while captured < num_images:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to capture frame")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.CHESSBOARD_SIZE, None
            )
            
            if ret_corners:
                # Refine corners
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.calib_dir}/calibration_{captured:02d}_{timestamp}.jpg"
                
                # Draw corners for visual verification
                img_with_corners = frame.copy()
                cv2.drawChessboardCorners(img_with_corners, self.CHESSBOARD_SIZE, corners, True)
                cv2.imwrite(filename, img_with_corners)
                
                # Also save original
                orig_filename = f"{self.calib_dir}/original_{captured:02d}_{timestamp}.jpg"
                cv2.imwrite(orig_filename, frame)
                
                captured += 1
                print(f"✅ Captured image {captured}/{num_images} - {filename}")
                
                # Wait for next capture
                if captured < num_images:
                    print(f"⏳ Next capture in {interval} seconds... (move chessboard to new position)")
                    time.sleep(interval)
            else:
                failed += 1
                if failed % 10 == 0:  # Print every 10 failures
                    print(f"⚠️  Chessboard not detected (failed {failed} times)")
                time.sleep(0.1)  # Short wait before retry
                
        self.camera.release()
        print(f"📸 Image capture complete! {captured} images saved to {self.calib_dir}/")
        
    def process_calibration_images(self):
        """Process saved calibration images for calibration"""
        print("🔧 Processing calibration images...")
        
        # Find all calibration images
        pattern = f"{self.calib_dir}/original_*.jpg"
        image_files = sorted(glob.glob(pattern))
        
        if not image_files:
            print(f"❌ No calibration images found in {self.calib_dir}/")
            print(f"   Looking for pattern: {pattern}")
            return False
            
        print(f"📁 Found {len(image_files)} calibration images")
        
        # Prepare object points
        objp = self.prepare_object_points()
        
        # Reset calibration data
        self.obj_points = []
        self.img_points = []
        
        processed = 0
        
        for i, filename in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(filename)}")
            
            # Load image
            img = cv2.imread(filename)
            if img is None:
                print(f"❌ Failed to load {filename}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, None)
            
            if ret:
                # Refine corners
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                # Add to calibration data
                self.obj_points.append(objp)
                self.img_points.append(corners)
                processed += 1
                print(f"   ✅ Chessboard detected and processed")
            else:
                print(f"   ❌ Chessboard not detected")
                
        if processed < 10:
            print(f"⚠️  Warning: Only {processed} valid images. Need at least 10 for good calibration")
            return False
            
        print(f"✅ Processed {processed} valid calibration images")
        
        # Perform calibration
        print("🔧 Performing camera calibration...")
        
        img_shape = gray.shape[::-1]
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, img_shape, None, None
        )
        
        if not ret:
            print("❌ Camera calibration failed")
            return False
            
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
            
        mean_error = total_error / len(self.obj_points)
        
        # Store calibration results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Save calibration data
        calib_data = {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'reprojection_error': mean_error,
            'image_count': processed,
            'image_size': img_shape,
            'timestamp': datetime.now().isoformat(),
            'chessboard_size': self.CHESSBOARD_SIZE,
            'square_size_mm': self.SQUARE_SIZE
        }
        
        calib_filename = f"camera_intrinsic_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(calib_filename, 'w') as f:
            json.dump(calib_data, f, indent=2)
            
        print("=" * 50)
        print("🎉 CAMERA CALIBRATION COMPLETE!")
        print("=" * 50)
        print(f"📊 Calibration Results:")
        print(f"   • Images processed: {processed}")
        print(f"   • Mean reprojection error: {mean_error:.4f} pixels")
        print(f"   • Image size: {img_shape}")
        print(f"   • Focal length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
        print(f"   • Principal point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
        print(f"   • Distortion coefficients: {dist_coeffs.flatten()}")
        print(f"📁 Calibration data saved to: {calib_filename}")
        
        # Quality assessment
        if mean_error < 0.5:
            print("✅ Excellent calibration quality!")
        elif mean_error < 1.0:
            print("✅ Good calibration quality")
        elif mean_error < 2.0:
            print("⚠️  Acceptable calibration quality")
        else:
            print("❌ Poor calibration quality - consider recapturing images")
            
        return True
        
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()

def main():
    parser = argparse.ArgumentParser(description='Headless Camera Calibration System')
    parser.add_argument('--capture', action='store_true', help='Capture calibration images only')
    parser.add_argument('--process', action='store_true', help='Process existing images only')
    parser.add_argument('--both', action='store_true', help='Capture and process (default)')
    parser.add_argument('--num-images', type=int, default=30, help='Number of images to capture')
    parser.add_argument('--interval', type=float, default=2.0, help='Seconds between captures')
    
    args = parser.parse_args()
    
    # Default to both if no specific mode is selected
    if not any([args.capture, args.process, args.both]):
        args.both = True
    
    calibrator = HeadlessCameraCalibrator()
    
    try:
        if args.capture or args.both:
            calibrator.capture_calibration_images(args.num_images, args.interval)
            
        if args.process or args.both:
            success = calibrator.process_calibration_images()
            if not success:
                print("❌ Calibration processing failed")
                return 1
                
        print("\n🎉 Calibration procedure complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Calibration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Calibration error: {e}")
        return 1
    finally:
        calibrator.cleanup()

if __name__ == "__main__":
    exit(main())
