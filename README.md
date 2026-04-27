# SafeSpeed-AI — Autonomous Highway Traffic Enforcement

A real-time traffic enforcement system fusing computer vision and mmWave radar for speed detection, vehicle tracking, and license plate recognition at scale.

Built for deployment on NVIDIA Jetson edge hardware with a full NVIDIA DeepStream pipeline.

## Pipeline architecture

```
Camera feed → YOLO11 (primary detection) → LPD (license plate detection) → LPR (plate recognition)
                                                          ↕
AWR1843 mmWave radar → speed + behaviour classification
                                                          ↕
                              Fusion layer → violation database → web display
```

## Tech stack

- **Vision pipeline** — NVIDIA DeepStream, YOLO11 primary inference, LPDNet + LPRNet secondary models
- **Radar** — TI AWR1843 mmWave radar via custom Python interface (`awr1843_interface.py`)
- **Tracker** — NvDCF multi-object tracker
- **Backend** — Python, SQLite violation database
- **Edge deployment** — NVIDIA Jetson (setup scripts included)
- **Display** — Real-time web dashboard with live fusion output

## Key components

- `src/pipeline/safespeed_ai.py` — Core pipeline orchestrator
- `src/pipeline/safespeed_deepstream.py` — DeepStream GStreamer pipeline definition
- `awr1843_interface.py` — mmWave radar serial interface and data parser
- `fixed_web_display_fusion.py` — Live web display with sensor fusion visualisation
- `scripts/setup_jetson.sh` — Full Jetson environment setup
- `scripts/download_models.sh` — Model weight download automation
- `deepstream/configs/` — All inference, tracker, and pipeline configs

## Architecture note

The radar and vision pipelines run independently and are fused at the classification stage. mmWave provides accurate speed ground truth that camera-only approaches cannot reliably compute. License plate detection and recognition run as DeepStream secondary GIEs on the YOLO primary detections — only vehicles detected by the primary model trigger the LPD/LPR chain, keeping compute efficient.
