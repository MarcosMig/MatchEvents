# Architecture

## Goal

Build a modular football video analysis system focused on broadcast feeds.

The system should support:

- video ingestion
- object detection
- multi-object tracking
- pitch projection / homography
- event derivation
- metrics extraction
- annotated video rendering
- structured export for downstream analysis

## High-level pipeline

```text
Broadcast Video
    ↓
Frame Reader
    ↓
Detector
    ↓
Tracker
    ↓
Pitch Mapper
    ↓
Event Engine
    ↓
Metrics Engine
    ↓
Exports + Visualizations
```

## Modules

### `io`
Responsible for:
- reading video metadata
- iterating through frames
- writing outputs

### `detectors`
Responsible for:
- object detection interface
- detector adapters (baseline, YOLO, Eagle-compatible, FootAndBall, etc.)

Expected classes:
- player
- goalkeeper
- referee
- ball

### `tracking`
Responsible for:
- frame-to-frame association
- track IDs
- smoothing and track continuity

Candidate implementations:
- baseline stub
- ByteTrack adapter
- Eagle-compatible tracker wrapper

### `homography`
Responsible for:
- mapping image coordinates to pitch coordinates
- homography estimation and transformation
- minimap projection

### `events`
Responsible for deriving higher-level football events from tracks and ball state.

Examples:
- ball possession
- pass candidates
- carries
- interceptions
- duels (later)

### `metrics`
Responsible for:
- distance covered
- speed estimates
- heatmaps / occupancy
- team shape metrics

### `visualization`
Responsible for:
- overlay rendering on video
- minimap rendering
- debug drawings

### `pipeline`
Responsible for:
- orchestrating all modules
- config-driven execution
- output persistence

## Data contracts

### Detection
Minimum expected fields per detection:
- `frame_idx`
- `class_name`
- `confidence`
- `bbox_xyxy`

### Track
Minimum expected fields per track row:
- `frame_idx`
- `track_id`
- `class_name`
- `confidence`
- `bbox_xyxy`
- `image_center_xy`
- `pitch_xy` (optional until homography exists)

## Development sequence

### Phase 1
- working video reader
- detector interface
- tracker interface
- config-driven pipeline
- CSV export of detections/tracks

### Phase 2
- baseline detector adapter
- baseline tracker adapter
- simple annotations

### Phase 3
- pitch mapping
- ball-specific improvements
- benchmark against open datasets

### Phase 4
- event engine
- tactical and physical metrics

## External references to integrate later

- Eagle: broadcast-to-tracking backbone
- SkillCorner Open Data: validation and benchmark
- SoccerNet Tracking: detection/tracking evaluation
- ByteTrack: robust MOT association
- FootAndBall: ball-focused detector candidate
