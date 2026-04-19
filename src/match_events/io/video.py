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
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def get_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        return VideoMetadata(
            path=str(self.video_path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    def frames(self) -> Generator[tuple[int, object], None, None]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()
