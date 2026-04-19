from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from match_events.detectors.base import BaseDetector, StubDetector
from match_events.io.video import VideoReader
from match_events.tracking.base import BaseTracker, StubTracker
from match_events.visualization import VideoWriter, draw_tracks


class MatchEventsPipeline:
    def __init__(
        self,
        detector: BaseDetector | None = None,
        tracker: BaseTracker | None = None,
    ) -> None:
        self.detector = detector or StubDetector()
        self.tracker = tracker or StubTracker()

    def run(self, video_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        video_reader = VideoReader(video_path)
        metadata = video_reader.get_metadata()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        detections_rows: list[dict] = []
        tracks_rows: list[dict] = []

        annotated_video_path = output_path / "annotated.mp4"
        writer = VideoWriter(
            output_path=annotated_video_path,
            fps=max(metadata.fps, 1.0),
            width=metadata.width,
            height=metadata.height,
        )

        try:
            for frame_idx, frame in video_reader.frames():
                detections = self.detector.predict(frame, frame_idx)
                tracks = self.tracker.update(detections, frame_idx)

                for det in detections:
                    row = asdict(det)
                    row["center_xy"] = det.center_xy
                    detections_rows.append(row)

                for track in tracks:
                    row = asdict(track)
                    row["center_xy"] = track.center_xy
                    tracks_rows.append(row)

                annotated_frame = draw_tracks(frame, tracks)
                writer.write(annotated_frame)
        finally:
            writer.release()

        detections_csv = output_path / "detections.csv"
        tracks_csv = output_path / "tracks.csv"
        metadata_json = output_path / "video_metadata.json"

        pd.DataFrame(detections_rows).to_csv(detections_csv, index=False)
        pd.DataFrame(tracks_rows).to_csv(tracks_csv, index=False)
        pd.Series(asdict(metadata)).to_json(metadata_json, indent=2)

        return {
            "detections_csv": str(detections_csv),
            "tracks_csv": str(tracks_csv),
            "video_metadata_json": str(metadata_json),
            "annotated_video": str(annotated_video_path),
        }
