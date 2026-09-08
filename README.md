# football_analysis

A football broadcast video → per-frame tactical analysis pipeline. The project processes camera footage to detect players, track them across frames, and convert pixel coordinates to real-world meters using pitch geometry calibration.

## What it does

1. **Scene detection** - Identifies whether the camera is in BROADCAST_WIDE, LOW_ANGLE_CORNER, CLOSE_UP, or UNKNOWN mode
2. **Keypoint calibration** - Detects 29 keypoint positions on the pitch and computes a homography transform
3. **Camera motion tracking** - Propagates calibration across frames to handle zoom/occlusion gaps
4. **Object detection** - Detects players and the ball in each frame
5. **Tracking** - Maintains stable IDs for tracked objects across frames
6. **Team classification** - Assigns teams based on jersey color (green mask + EMA smoothing)

## Project layout

```
football_analysis/
  config/pitch_config.py          Pitch dimensions (FIFA-standard), validated via pydantic
  schemas/                        Output contract: FrameResult, TrackedObject, CalibrationInfo
  calibration/                    Scene routing, keypoint detection, homography computation
    orchestrator.py                Main entry point for video processing
  tracking/                       Object detection, ByteTracker, jersey-based team assignment
    __init__.py                    Exports detector, tracker, classifier
  coordinates/                    (Phase 3 - coordinate transforms)
  api/                            (Phase 5 - FastAPI microservice)
  tools/distance_tool.py          Interactive tool: click two points to measure real distance
  examples/demo_schema_usage.py   Example showing schema usage
  tests/                          pytest suite
```

## Technologies

- **Python** - main language
- **Pydantic** - data validation and JSON serialization for all schemas
- **OpenCV (cv2)** - homography, optical flow, color segmentation, k-means clustering
- **Ultralytics YOLO** - keypoint detection and object detection (lazy-loaded)
- **NumPy** - numerical operations

## Running the project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the demo

Shows how schemas work together:

```bash
python examples/demo_schema_usage.py
```

### Run tests

```bash
pytest tests/ -v
```

## Main entry point

Process video frames through the full pipeline:

```python
from calibration.orchestrator import VideoCalibrationOrchestrator

# Create orchestrator (uses FIFA defaults, or custom PitchConfig)
orchestrator = VideoCalibrationOrchestrator()

# Process a frame and get calibrated world positions + tracked objects
result = orchestrator.process_frame(frame)
```

`result` contains:
- `calibration`: homography for pixel→world conversion, calibration status
- `tracked_objects`: list of TrackedObject with position (meters), team assignment, provenance

## Output schema

All phases emit pydantic models that fit together:

- **FrameResult** - per-frame output combining calibration info + tracked objects
- **TrackedObject** - player/ball with world position, track ID, team, provenance
- **CalibrationInfo** - homography, status (calibrated/propagated/not_calibrated)

## Conventions

- Schemas in `schemas/` are the only thing downstream code imports from
- Every class implements a duck-typed interface (e.g., `BaseTracker`)
- Provenance is tracked separately for calibration trust and object identity
- Confidence thresholds are tuned against synthetic data; verify on real footage

## Next phases

| Phase | What it does | Status |
|-------|-------------|--------|
| 3 | Coordinate transforms, CSV/Parquet export | Not started |
| 4 | Stats: radar views, Voronoi diagrams, pressing heatmaps | Not started |
| 5 | FastAPI microservice for serving results | Not started |

See `status.md` for live test counts and `CLAUDE.md` for project guidance.
