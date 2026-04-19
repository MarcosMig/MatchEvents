from __future__ import annotations

from typing import Any

from match_events.detectors.base import BaseDetector, Detection


class YoloDetector(BaseDetector):
    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the YOLO model.

        This is a placeholder adapter. In the next step, this method should:
        - import the chosen inference backend
        - load weights from model_path
        - bind model to device
        """
        self._model = None

    def predict(self, frame: Any, frame_idx: int) -> list[Detection]:
        if self._model is None:
            return []

        # Placeholder for real inference-to-Detection conversion.
        return []
