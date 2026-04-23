from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from match_events.analytics import PossessionEstimator
from match_events.detectors.base import BaseDetector, StubDetector
from match_events.io.video import VideoReader
from match_events.postprocessing import (
    BallClassCorrector,
    FieldRegionFilter,
    FrameRoleCorrector,
    TrackTeamAssigner,
)
from match_events.tracking.base import BaseTracker, StubTracker
from match_events.visualization import VideoWriter, draw_tracks


class MatchEventsPipeline:
    def __init__(
        self,
        detector: BaseDetector | None = None,
        tracker: BaseTracker | None = None,
        field_filter: FieldRegionFilter | None = None,
        team_assigner: TrackTeamAssigner | None = None,
        possession_estimator: PossessionEstimator | None = None,
        role_corrector: FrameRoleCorrector | None = None,
        ball_corrector: BallClassCorrector | None = None,
        interpolate_ball_tracks: bool = False,
        ball_interpolation_max_gap: int = 12,
    ) -> None:
        self.detector = detector or StubDetector()
        self.tracker = tracker or StubTracker()
        self.field_filter = field_filter or FieldRegionFilter()
        self.team_assigner = team_assigner or TrackTeamAssigner()
        self.possession_estimator = possession_estimator or PossessionEstimator()
        self.role_corrector = role_corrector or FrameRoleCorrector()
        self.ball_corrector = ball_corrector or BallClassCorrector()
        self.interpolate_ball_tracks = interpolate_ball_tracks
        self.ball_interpolation_max_gap = ball_interpolation_max_gap

    def run(self, video_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        video_reader = VideoReader(
            video_path,
            start_frame=getattr(self, "start_frame", None),
            end_frame=getattr(self, "end_frame", None),
            start_time_seconds=getattr(self, "start_time_seconds", None),
            end_time_seconds=getattr(self, "end_time_seconds", None),
        )
        metadata = video_reader.get_metadata()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        detections_rows: list[dict] = []
        tracks_rows: list[dict] = []
        tracks_by_frame: dict[int, list] = {}

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
                detections = self.field_filter.apply(frame, detections)
                detections = self.ball_corrector.apply(detections)
                detections = self.role_corrector.apply(frame, detections)
                tracks = self.tracker.update(detections, frame_idx)
                if self.interpolate_ball_tracks:
                    tracks = self._with_interpolated_ball_tracks(tracks, frame_idx)
                tracks = self.team_assigner.apply(frame, tracks)
                tracks_by_frame[frame_idx] = tracks

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
        possession_csv = output_path / "possession.csv"

        pd.DataFrame(detections_rows).to_csv(detections_csv, index=False)
        pd.DataFrame(tracks_rows).to_csv(tracks_csv, index=False)
        pd.Series(asdict(metadata)).to_json(metadata_json, indent=2)
        possession_rows = self.possession_estimator.estimate(tracks_by_frame)
        if possession_rows:
            pd.DataFrame(possession_rows).to_csv(possession_csv, index=False)

        outputs = {
            "detections_csv": str(detections_csv),
            "tracks_csv": str(tracks_csv),
            "video_metadata_json": str(metadata_json),
            "annotated_video": str(annotated_video_path),
        }
        if possession_rows:
            outputs["possession_csv"] = str(possession_csv)
        return outputs

    def _with_interpolated_ball_tracks(
        self,
        tracks: list,
        frame_idx: int,
    ) -> list:
        ball_tracks = [track for track in tracks if track.class_name == "ball"]
        if ball_tracks:
            self._last_ball_track = ball_tracks[0]
            return tracks

        last_ball_track = getattr(self, "_last_ball_track", None)
        if last_ball_track is None:
            return tracks

        gap = frame_idx - last_ball_track.frame_idx
        if gap <= 0 or gap > self.ball_interpolation_max_gap:
            return tracks

        from dataclasses import replace

        interpolated_ball = replace(
            last_ball_track,
            frame_idx=frame_idx,
            confidence=min(last_ball_track.confidence, 0.10),
        )
        return [*tracks, interpolated_ball]
