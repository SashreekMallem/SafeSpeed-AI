#!/usr/bin/env python3

"""
Standalone OCR test for license plate text extraction
This script tests the OCR functionality without the full DeepStream pipeline
"""

import cv2
import numpy as np
import easyocr
import pytesseract
import sys

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

def test_easyocr(image):
    """Test EasyOCR on the image"""
    try:
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for testing
        print("EasyOCR initialized")
        
        results = reader.readtext(image, 
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                paragraph=False,
                                width_ths=0.9,
                                height_ths=0.7)
        
        print(f"EasyOCR Results: {len(results)} detections")
        for i, (bbox, text, confidence) in enumerate(results):
            print(f"  {i+1}. Text: '{text}' (confidence: {confidence:.3f})")
            
        return results
        
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return []

def test_tesseract(image):
    """Test Tesseract OCR on the image"""
    try:
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(image, config=config)
        text = text.strip().replace(' ', '').replace('\n', '').upper()
        
        print(f"Tesseract Result: '{text}'")
        return text
        
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""

def create_test_license_plate():
    """Create a synthetic license plate image for testing"""
    # Create a white background
    img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    
    # Add some text that looks like a license plate
    cv2.putText(img, 'ABC123', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add some noise for realism
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.subtract(img, noise)
    
    return img

def main():
    print("License Plate OCR Testing Script")
    print("=" * 40)
    
    # Create a test license plate image
    print("Creating synthetic license plate image...")
    test_image = create_test_license_plate()
    
    # Save the test image
    cv2.imwrite('test_license_plate.png', test_image)
    print("Test image saved as 'test_license_plate.png'")
    
    # Enhance the image
    print("\nEnhancing image...")
    enhanced_image = enhance_license_plate_image(test_image)
    cv2.imwrite('test_license_plate_enhanced.png', enhanced_image)
    print("Enhanced image saved as 'test_license_plate_enhanced.png'")
    
    # Test both OCR engines
    print("\n" + "-" * 40)
    print("Testing OCR Engines")
    print("-" * 40)
    
    print("\n1. Testing EasyOCR...")
    easyocr_results = test_easyocr(enhanced_image)
    
    print("\n2. Testing Tesseract...")
    tesseract_result = test_tesseract(enhanced_image)
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Expected text: 'ABC123'")
    
    if easyocr_results:
        best_easyocr = max(easyocr_results, key=lambda x: x[2])
        print(f"EasyOCR best result: '{best_easyocr[1]}' (conf: {best_easyocr[2]:.3f})")
    else:
        print("EasyOCR: No results")
        
    print(f"Tesseract result: '{tesseract_result}'")
    
    print("\nTest images created:")
    print("- test_license_plate.png (original)")
    print("- test_license_plate_enhanced.png (processed)")

if __name__ == "__main__":
    main()
