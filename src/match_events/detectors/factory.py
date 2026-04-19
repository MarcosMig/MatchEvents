from __future__ import annotations

from match_events.detectors.base import BaseDetector, StubDetector
from match_events.detectors.yolo_adapter import YoloDetector


def build_detector(config: dict) -> BaseDetector:
    detector_name = config["pipeline"]["detector"]
    runtime_device = config.get("runtime", {}).get("device", "cpu")

    if detector_name == "baseline_stub":
        return StubDetector()

    if detector_name == "yolo":
        yolo_cfg = config.get("models", {}).get("yolo", {})
        model_path = yolo_cfg.get("model_path")
        confidence = float(yolo_cfg.get("confidence", 0.25))
        if not model_path:
            raise ValueError("models.yolo.model_path must be set when pipeline.detector=yolo")
        return YoloDetector(
            model_path=model_path,
            device=runtime_device,
            confidence=confidence,
        )

    raise ValueError(f"Unsupported detector: {detector_name}")
