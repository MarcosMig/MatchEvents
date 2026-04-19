from __future__ import annotations

from pathlib import Path
from typing import Any

from match_events.detectors.base import BaseDetector, Detection


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        weights_path = Path(self.model_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self._model = YOLO(str(weights_path))

    def predict(self, frame: Any, frame_idx: int) -> list[Detection]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {})
            if boxes is None:
                continue

            xyxy_list = boxes.xyxy.cpu().tolist()
            conf_list = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            cls_list = boxes.cls.cpu().tolist() if boxes.cls is not None else []

            for bbox_xyxy, confidence, class_idx in zip(xyxy_list, conf_list, cls_list):
                class_name = names.get(int(class_idx), str(int(class_idx)))
                detections.append(
                    Detection(
                        frame_idx=frame_idx,
                        class_name=str(class_name),
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(v) for v in bbox_xyxy),
                    )
                )

        return detections
