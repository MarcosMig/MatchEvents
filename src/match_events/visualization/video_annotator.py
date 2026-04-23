from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from match_events.tracking import Track


def draw_tracks(frame, tracks: Iterable[Track]):
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
        team_suffix = f" [{track.team_id}]" if track.team_id else ""
        label = f"{track.class_name}{team_suffix} #{track.track_id} {track.confidence:.2f}"

        color = (0, 255, 0)
        if track.team_id == "team_a":
            color = (255, 80, 80)
        elif track.team_id == "team_b":
            color = (80, 160, 255)
        elif track.class_name == "referee":
            color = (0, 255, 255)
        elif track.class_name == "ball":
            color = (255, 255, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.output_path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
