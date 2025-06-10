# SafeSpeed AI - Complete Vehicle & License Plate Analysis System

## Overview
Complete DeepStream pipeline with 5 AI models:
1. **Primary Detection**: Vehicle detection (TrafficCamNet)
2. **Secondary 1**: Vehicle make classification 
3. **Secondary 2**: Vehicle type classification
4. **Secondary 3**: License plate detection (LPD)
5. **Secondary 4**: License plate recognition (LPR)

## Quick Start

### Run with USB Camera:
```bash
cd /home/projecta/SafespeedAI
python3 deepstream_test_2.py camera /dev/video0
```

### Run with Video File:
```bash
cd /home/projecta/SafespeedAI
python3 deepstream_test_2.py file /path/to/video.mp4
```

## System Components

### Models Used:
- **TrafficCamNet**: Primary vehicle detection
- **VehicleMakeNet**: Vehicle make classification (Honda, Toyota, etc.)
- **VehicleTypeNet**: Vehicle type classification (sedan, SUV, etc.)
- **LPDNet**: License plate detection
- **LPRNet**: License plate recognition (OCR)

### Configuration Files:
- `dstest2_pgie_config.txt` - Primary detection
- `dstest2_sgie1_config.txt` - Vehicle make
- `dstest2_sgie2_config.txt` - Vehicle type  
- `dstest2_sgie3_config.txt` - License plate detection
- `dstest2_sgie4_config.txt` - License plate recognition
- `dstest2_tracker_config.txt` - Object tracking

### Pipeline Flow:
```
Camera/File → Primary Detection → Tracker → Vehicle Make → Vehicle Type → LP Detection → LP Recognition → Display
```

## Requirements
- NVIDIA Jetson Orin AGX
- DeepStream 6.3+
- USB Camera (for camera mode)
- All model files in `deepstream_models/` directory
- Custom LPR parser library in `common/nvinfer_custom_lpr_parser/`

## Status
✅ All 5 AI models configured
✅ All configuration paths updated
✅ Custom LPR parser compiled
✅ Pipeline linking complete
✅ Ready to run!
