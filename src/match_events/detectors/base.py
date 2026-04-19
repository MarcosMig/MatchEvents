from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Detection:
    frame_idx: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame: Any, frame_idx: int) -> list[Detection]:
        raise NotImplementedError


class StubDetector(BaseDetector):
    def predict(self, frame: Any, frame_idx: int) -> list[Detection]:
        return []
