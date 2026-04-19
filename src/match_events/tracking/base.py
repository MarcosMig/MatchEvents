from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from match_events.detectors.base import Detection


@dataclass(slots=True)
class Track:
    frame_idx: int
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        raise NotImplementedError


class StubTracker(BaseTracker):
    def __init__(self) -> None:
        self._next_track_id = 1

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        tracks: list[Track] = []
        for det in detections:
            tracks.append(
                Track(
                    frame_idx=frame_idx,
                    track_id=self._next_track_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                )
            )
            self._next_track_id += 1
        return tracks
