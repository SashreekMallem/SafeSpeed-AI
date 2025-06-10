#!/usr/bin/env python3

"""
Calibrated Coordinate Transformation System
Implements proper 3D-to-2D projection using calibration data
"""

import numpy as np
import cv2
import json
import os
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class RadarPoint3D:
    """3D radar point in radar coordinate system"""
    x: float  # lateral (meters)
    y: float  # longitudinal (meters) 
    z: float  # elevation (meters)

@dataclass
class ImagePoint2D:
    """2D point in camera image coordinates"""
    u: int  # pixel x coordinate
    v: int  # pixel y coordinate

class CalibratedCoordinateTransform:
    """Research-based coordinate transformation using proper calibration"""
    
    def __init__(self, calibration_file: str = "/home/projecta/SafespeedAI/complete_calibration.json"):
        self.calibration_file = calibration_file
        self.is_calibrated = False
        
        # Calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None
        self.rotation_matrix = None
        
        # Fallback parameters (if calibration not available)
        self.fallback_camera_matrix = np.array([
            [800, 0, 640],
            [0, 800, 360], 
            [0, 0, 1]
        ], dtype=float)
        self.fallback_dist_coeffs = np.zeros(4)
        
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """Load calibration data from file"""
        if not os.path.exists(self.calibration_file):
            print(f"⚠️  Calibration file not found: {self.calibration_file}")
            print("   Run corner_reflector_calibration.py first for accurate results")
            print("   Using fallback parameters for basic functionality")
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                calib_data = json.load(f)
            
            # Load intrinsic parameters
            intrinsic = calib_data['intrinsic_calibration']
            self.camera_matrix = np.array(intrinsic['camera_matrix'])
            self.dist_coeffs = np.array(intrinsic['distortion_coefficients'])
            
            # Load extrinsic parameters
            extrinsic = calib_data['extrinsic_calibration']
            self.rotation_vector = np.array(extrinsic['rotation_vector'])
            self.translation_vector = np.array(extrinsic['translation_vector'])
            self.rotation_matrix = np.array(extrinsic['rotation_matrix'])
            
            self.is_calibrated = True
            
            print("✅ Loaded calibrated transformation parameters")
            print(f"   Mean reprojection error: {extrinsic['mean_reprojection_error']:.2f} pixels")
            print(f"   Calibration points: {extrinsic['inlier_count']}/{extrinsic['total_points']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            print("   Using fallback parameters")
            return False
    
    def radar_to_camera_3d(self, radar_point: RadarPoint3D) -> np.ndarray:
        """Transform 3D radar coordinates to 3D camera coordinates"""
        if not self.is_calibrated:
            # Fallback: simple coordinate system assumption
            # Assume radar is mounted slightly behind and above camera
            camera_point = np.array([
                radar_point.x,           # lateral stays same
                radar_point.y + 0.2,     # radar slightly behind camera
                radar_point.z - 0.1      # radar slightly above camera
            ])
            return camera_point
        
        # Use calibrated transformation
        radar_3d = np.array([[radar_point.x], [radar_point.y], [radar_point.z]])
        
        # Apply rotation and translation: P_camera = R * P_radar + T
        camera_3d = self.rotation_matrix @ radar_3d + self.translation_vector.reshape(3, 1)
        
        return camera_3d.flatten()
    
    def camera_3d_to_image_2d(self, camera_3d: np.ndarray) -> ImagePoint2D:
        """Project 3D camera coordinates to 2D image coordinates"""
        camera_matrix = self.camera_matrix if self.is_calibrated else self.fallback_camera_matrix
        dist_coeffs = self.dist_coeffs if self.is_calibrated else self.fallback_dist_coeffs
        
        # Handle points behind camera (negative Z)
        if camera_3d[2] <= 0:
            # Project to edge of image for points behind camera
            return ImagePoint2D(u=0, v=0)
        
        # Use OpenCV projection
        object_points = camera_3d.reshape(1, 1, 3)
        rvec = np.zeros(3)  # No additional rotation (already in camera coordinates)
        tvec = np.zeros(3)  # No additional translation
        
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        u = int(image_points[0][0][0])
        v = int(image_points[0][0][1])
        
        return ImagePoint2D(u=u, v=v)
    
    def radar_to_image(self, radar_point: RadarPoint3D, 
                      img_width: int = 1280, img_height: int = 720) -> ImagePoint2D:
        """Complete transformation: radar 3D -> camera 3D -> image 2D"""
        
        # Step 1: Transform radar coordinates to camera 3D coordinates
        camera_3d = self.radar_to_camera_3d(radar_point)
        
        # Step 2: Project camera 3D coordinates to image 2D coordinates
        image_point = self.camera_3d_to_image_2d(camera_3d)
        
        # Clamp to image bounds
        image_point.u = max(0, min(img_width - 1, image_point.u))
        image_point.v = max(0, min(img_height - 1, image_point.v))
        
        return image_point
    
    def get_transformation_info(self) -> dict:
        """Get information about current transformation parameters"""
        info = {
            'calibrated': self.is_calibrated,
            'calibration_file': self.calibration_file,
            'calibration_exists': os.path.exists(self.calibration_file)
        }
        
        if self.is_calibrated:
            info.update({
                'camera_focal_length': float(self.camera_matrix[0, 0]),
                'camera_center': [float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])],
                'translation_vector': self.translation_vector.tolist(),
                'rotation_matrix_available': self.rotation_matrix is not None
            })
        else:
            info.update({
                'using_fallback': True,
                'fallback_focal_length': float(self.fallback_camera_matrix[0, 0]),
                'fallback_center': [640, 360]
            })
        
        return info
    
    def validate_transformation(self, test_points: list = None) -> dict:
        """Validate transformation accuracy using known points"""
        if not self.is_calibrated:
            return {'error': 'No calibration data available for validation'}
        
        # Load test points from calibration data if not provided
        if test_points is None:
            try:
                with open(self.calibration_file, 'r') as f:
                    calib_data = json.load(f)
                test_points = calib_data['calibration_points'][:5]  # Use first 5 points
            except:
                return {'error': 'Cannot load test points'}
        
        errors = []
        
        for point in test_points:
            # Original image coordinates
            true_image = ImagePoint2D(u=int(point['image_x']), v=int(point['image_y']))
            
            # Radar coordinates
            radar_point = RadarPoint3D(
                x=point['radar_x'], 
                y=point['radar_y'], 
                z=point['radar_z']
            )
            
            # Predicted image coordinates
            pred_image = self.radar_to_image(radar_point)
            
            # Calculate error
            error = np.sqrt((true_image.u - pred_image.u)**2 + (true_image.v - pred_image.v)**2)
            errors.append(error)
        
        return {
            'mean_error_pixels': float(np.mean(errors)),
            'max_error_pixels': float(np.max(errors)),
            'min_error_pixels': float(np.min(errors)),
            'std_error_pixels': float(np.std(errors)),
            'num_test_points': len(errors),
            'errors': errors
        }


class LegacyCoordinateTransform:
    """Legacy coordinate transformation for comparison"""
    
    def __init__(self):
        self.scale_x = 60  # pixels per meter lateral
        self.scale_y = 30  # pixels per meter longitudinal
    
    def radar_to_image(self, radar_point: RadarPoint3D, 
                      img_width: int = 1280, img_height: int = 720) -> ImagePoint2D:
        """Legacy simple scaling transformation"""
        
        img_x = int(img_width / 2 + radar_point.x * self.scale_x)
        img_y = int(img_height - 100 - radar_point.y * self.scale_y)
        
        # Clamp to image bounds
        img_x = max(0, min(img_width - 1, img_x))
        img_y = max(0, min(img_height - 1, img_y))
        
        return ImagePoint2D(u=img_x, v=img_y)


def test_coordinate_transformation():
    """Test the coordinate transformation system"""
    print("🧪 Testing Coordinate Transformation System")
    print("=" * 50)
    
    # Initialize transformations
    calibrated = CalibratedCoordinateTransform()
    legacy = LegacyCoordinateTransform()
    
    # Print transformation info
    info = calibrated.get_transformation_info()
    print("📊 Transformation Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test points
    test_radar_points = [
        RadarPoint3D(x=0.0, y=5.0, z=0.0),    # Straight ahead, 5m
        RadarPoint3D(x=-2.0, y=3.0, z=0.0),   # Left, 3m
        RadarPoint3D(x=2.0, y=3.0, z=0.0),    # Right, 3m
        RadarPoint3D(x=0.0, y=10.0, z=1.0),   # Far ahead, elevated
    ]
    
    print(f"\n🎯 Comparing Transformations:")
    print("Radar Point (x,y,z) -> Calibrated (u,v) | Legacy (u,v)")
    print("-" * 60)
    
    for i, radar_point in enumerate(test_radar_points):
        calib_img = calibrated.radar_to_image(radar_point)
        legacy_img = legacy.radar_to_image(radar_point)
        
        print(f"Point {i+1}: ({radar_point.x:+.1f}, {radar_point.y:+.1f}, {radar_point.z:+.1f}) -> "
              f"({calib_img.u:4d}, {calib_img.v:4d}) | ({legacy_img.u:4d}, {legacy_img.v:4d})")
    
    # Validation if calibrated
    if calibrated.is_calibrated:
        print(f"\n🔍 Validation Results:")
        validation = calibrated.validate_transformation()
        if 'error' not in validation:
            print(f"   Mean error: {validation['mean_error_pixels']:.2f} pixels")
            print(f"   Max error: {validation['max_error_pixels']:.2f} pixels")
            print(f"   Test points: {validation['num_test_points']}")
        else:
            print(f"   {validation['error']}")
    
    print("\n✅ Coordinate transformation test complete")


if __name__ == "__main__":
    test_coordinate_transformation()
