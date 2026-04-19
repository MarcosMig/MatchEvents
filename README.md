# MatchEvents

Open-source repository for building a broadcast-video football tracking and match-events pipeline.

## Objective

Create a modular system that:

- ingests broadcast football video
- detects players, referees, goalkeepers, and ball
- tracks entities over time
- maps coordinates to the pitch
- derives match events and tactical/physical metrics
- exports annotated video and structured data

## Current capabilities

The repository already supports:

- reading a local video file
- running a detector selected by config
- exporting `detections.csv`
- exporting `tracks.csv`
- exporting `video_metadata.json`
- rendering `annotated.mp4` with bounding boxes, labels, and track IDs

## Planned stack

- **Backbone / inspiration:** Eagle
- **Benchmark / validation:** SkillCorner Open Data
- **Tracking evaluation / training:** SoccerNet Tracking
- **Tracker candidate:** ByteTrack
- **Ball-focused detector candidate:** FootAndBall
- **Core language:** Python

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/MarcosMig/MatchEvents.git
cd MatchEvents
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

### 2. Place a video file

Put a test video at:

```text
data/raw/sample.mp4
```

### 3. Run with stub detector

```bash
make run
```

### 4. Run with YOLO

Edit `configs/base.yaml`:

```yaml
pipeline:
  detector: yolo
  tracker: baseline_stub
  pitch_mapper: none

models:
  yolo:
    model_path: data/external/yolo/best.pt
    confidence: 0.25
```

Then place your YOLO weights at:

```text
data/external/yolo/best.pt
```

And run:

```bash
make run
```

## Outputs

A successful run writes files into the configured output directory, by default:

```text
outputs/run_001/
├── annotated.mp4
├── detections.csv
├── tracks.csv
└── video_metadata.json
```

## Proposed repository structure

```text
MatchEvents/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   └── match_events/
│       ├── detectors/
│       ├── tracking/
│       ├── homography/
│       ├── metrics/
│       ├── events/
│       ├── io/
│       ├── visualization/
│       └── pipeline/
└── tests/
```

## Roadmap

### Milestone 1 — MVP
- read a short broadcast video
- run baseline detection
- generate simple tracks
- export detections/tracks to CSV or Parquet
- render annotated video

### Milestone 2 — Pitch space
- add homography / pitch mapping
- project detections to field coordinates
- render minimap

### Milestone 3 — Benchmark
- compare against SkillCorner Open Data
- evaluate tracking stability and ball continuity

### Milestone 4 — Event layer
- infer simple match events
- build event schema and export

## Notes

This repository is being structured as a modular research-to-production project rather than a single notebook prototype.
