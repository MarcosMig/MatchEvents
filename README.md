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

## Planned stack

- **Backbone / inspiration:** Eagle
- **Benchmark / validation:** SkillCorner Open Data
- **Tracking evaluation / training:** SoccerNet Tracking
- **Tracker candidate:** ByteTrack
- **Ball-focused detector candidate:** FootAndBall
- **Core language:** Python

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

## Next implementation steps

1. Add project packaging and dependencies
2. Add config-driven pipeline entrypoint
3. Implement video IO
4. Add detector/tracker interfaces
5. Plug first baseline model

## Notes

This repository is being structured as a modular research-to-production project rather than a single notebook prototype.
