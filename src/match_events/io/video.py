from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2


@dataclass(slots=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


class VideoReader:
    def __init__(
        self,
        video_path: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_time_seconds: float | None = None,
        end_time_seconds: float | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_time_seconds = start_time_seconds
        self.end_time_seconds = end_time_seconds

    def _resolved_frame_range(self, fps: float, frame_count: int) -> tuple[int, int]:
        start_frame = int(self.start_frame or 0)
        end_frame = int(self.end_frame if self.end_frame is not None else frame_count)

        if self.start_time_seconds is not None:
            start_frame = max(start_frame, int(self.start_time_seconds * fps))
        if self.end_time_seconds is not None:
            end_frame = min(end_frame, int(self.end_time_seconds * fps))

        start_frame = max(0, min(start_frame, frame_count))
        end_frame = max(start_frame, min(end_frame, frame_count))
        return start_frame, end_frame

    def get_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        start_frame, end_frame = self._resolved_frame_range(fps=fps, frame_count=frame_count)

        return VideoMetadata(
            path=str(self.video_path),
            fps=fps,
            frame_count=max(0, end_frame - start_frame),
            width=width,
            height=height,
        )

    def frames(self) -> Generator[tuple[int, object], None, None]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_frame, end_frame = self._resolved_frame_range(fps=fps, frame_count=frame_count)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        try:
            while frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()
