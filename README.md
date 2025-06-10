# SafeSpeed AI - Complete Vehicle & License Plate Analysis System
**Powered by NVIDIA DeepStream on Jetson Orin AGX**

## Overview
Ultimate computer vision system combining the best of both worlds:
- **Official NVIDIA DeepStream Test2**: Vehicle detection, make/model classification, tracking
- **Your Custom LPD/LPR Setup**: License plate detection and recognition
- **USB Camera Support**: Real-time processing from camera input

## Complete Pipeline
```
USB Camera → TrafficCamNet → VehicleMakeNet → VehicleTypeNet → LPDNet → LPRNet → Database + Display
                    ↓              ↓              ↓           ↓        ↓
               Vehicle Detection  Make Class    Type Class   LP Det   LP Text
                    ↓
                 NvDCF Tracker
```

## Models Used
1. **TrafficCamNet**: Primary vehicle detection (Official NVIDIA)
2. **VehicleMakeNet**: Vehicle make classification (Official NVIDIA) 
3. **VehicleTypeNet**: Vehicle type classification (Official NVIDIA)
4. **LPDNet**: License plate detection (Your setup)
5. **LPRNet**: License plate recognition (Your setup)

## What You Get
- **Vehicle Detection**: Cars, trucks, motorcycles, etc.
- **Vehicle Make**: Honda, Toyota, Ford, etc.  
- **Vehicle Type**: Sedan, SUV, truck, coupe, etc.
- **License Plates**: Detected regions
- **License Plate Text**: OCR recognition
- **Tracking**: Persistent object tracking across frames
- **Database Storage**: All detections with timestamps
- **Real-time Display**: Live video with all annotations

## Quick Start
```bash
# Run the complete system
./scripts/run_complete_pipeline.sh

# Query all results 
python3 scripts/query_complete_database.py

# Search specific license plate
python3 scripts/query_complete_database.py --search ABC123
```

## System Requirements
- NVIDIA Jetson Orin AGX
- JetPack 5.1+ (DeepStream 6.3+)
- USB Camera at /dev/video0
- All NVIDIA TAO models (auto-downloaded)
- Your LPD/LPR models and engines# Safespeed-AI
# SafeSpeed-AI
