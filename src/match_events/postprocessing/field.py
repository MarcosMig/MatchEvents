from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from match_events.detectors.base import Detection


@dataclass(slots=True)
class FieldMaskConfig:
    enabled: bool = False
    min_area_ratio: float = 0.08
    margin_px: int = 10
    top_ignore_ratio: float = 0.18
    close_kernel_size: int = 21
    open_kernel_size: int = 9
    min_green_saturation: int = 35
    min_green_value: int = 35
    hue_min: int = 28
    hue_max: int = 95
    perspective_filter_enabled: bool = False
    near_min_height_ratio: float = 0.06
    far_min_height_ratio: float = 0.015
    near_max_height_ratio: float = 0.32
    far_max_height_ratio: float = 0.08
    min_width_to_height_ratio: float = 0.18
    max_width_to_height_ratio: float = 0.95


class FieldRegionFilter:
    PERSON_CLASSES = {"player", "referee", "goalkeeper"}

    def __init__(self, config: FieldMaskConfig | None = None) -> None:
        self.config = config or FieldMaskConfig()

    @classmethod
    def from_config(cls, raw_config: dict) -> "FieldRegionFilter":
        field_cfg = raw_config.get("postprocessing", {}).get("field_mask", {})
        return cls(
            FieldMaskConfig(
                enabled=bool(field_cfg.get("enabled", False)),
                min_area_ratio=float(field_cfg.get("min_area_ratio", 0.08)),
                margin_px=int(field_cfg.get("margin_px", 10)),
                top_ignore_ratio=float(field_cfg.get("top_ignore_ratio", 0.18)),
                close_kernel_size=int(field_cfg.get("close_kernel_size", 21)),
                open_kernel_size=int(field_cfg.get("open_kernel_size", 9)),
                min_green_saturation=int(field_cfg.get("min_green_saturation", 35)),
                min_green_value=int(field_cfg.get("min_green_value", 35)),
                hue_min=int(field_cfg.get("hue_min", 28)),
                hue_max=int(field_cfg.get("hue_max", 95)),
                perspective_filter_enabled=bool(
                    field_cfg.get("perspective_filter_enabled", False)
                ),
                near_min_height_ratio=float(field_cfg.get("near_min_height_ratio", 0.06)),
                far_min_height_ratio=float(field_cfg.get("far_min_height_ratio", 0.015)),
                near_max_height_ratio=float(field_cfg.get("near_max_height_ratio", 0.32)),
                far_max_height_ratio=float(field_cfg.get("far_max_height_ratio", 0.08)),
                min_width_to_height_ratio=float(
                    field_cfg.get("min_width_to_height_ratio", 0.18)
                ),
                max_width_to_height_ratio=float(
                    field_cfg.get("max_width_to_height_ratio", 0.95)
                ),
            )
        )

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if not self.config.enabled or not detections:
            return detections

        mask = self._field_mask(frame)
        if mask is None:
            return detections

        return [
            detection
            for detection in detections
            if self._keep_detection(detection, mask)
            and self._passes_perspective_size(detection, frame.shape[0])
        ]

    def _field_mask(self, frame: np.ndarray) -> np.ndarray | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                self.config.hue_min,
                self.config.min_green_saturation,
                self.config.min_green_value,
            ],
            dtype=np.uint8,
        )
        upper = np.array([self.config.hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        height, width = mask.shape[:2]
        top_ignore = int(height * self.config.top_ignore_ratio)
        if top_ignore > 0:
            mask[:top_ignore, :] = 0

        close_kernel = np.ones(
            (self.config.close_kernel_size, self.config.close_kernel_size), dtype=np.uint8
        )
        open_kernel = np.ones(
            (self.config.open_kernel_size, self.config.open_kernel_size), dtype=np.uint8
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        min_area = int(height * width * self.config.min_area_ratio)
        component_mask = np.zeros_like(mask)
        kept_components = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component_mask[labels == label_idx] = 255
            kept_components += 1

        if kept_components == 0:
            return None

        # Expand slightly so we do not cut objects touching the field boundary.
        margin = max(1, self.config.margin_px)
        dilate_kernel = np.ones((margin, margin), dtype=np.uint8)
        return cv2.dilate(component_mask, dilate_kernel, iterations=1)

    def _keep_detection(self, detection: Detection, mask: np.ndarray) -> bool:
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        if detection.class_name == "ball":
            px = int(round((x1 + x2) / 2.0))
            py = int(round((y1 + y2) / 2.0))
        else:
            px = int(round((x1 + x2) / 2.0))
            py = int(round(y2))

        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        return bool(mask[py, px] > 0)

    def _passes_perspective_size(self, detection: Detection, frame_height: int) -> bool:
        if not self.config.perspective_filter_enabled:
            return True
        if detection.class_name not in self.PERSON_CLASSES:
            return True

        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if height <= 1.0 or frame_height <= 1:
            return False

        foot_y = max(0.0, min(float(frame_height - 1), y2))
        foot_ratio = foot_y / max(float(frame_height - 1), 1.0)

        min_height_ratio = _lerp(
            self.config.far_min_height_ratio,
            self.config.near_min_height_ratio,
            foot_ratio,
        )
        max_height_ratio = _lerp(
            self.config.far_max_height_ratio,
            self.config.near_max_height_ratio,
            foot_ratio,
        )
        height_ratio = height / float(frame_height)
        if height_ratio < min_height_ratio or height_ratio > max_height_ratio:
            return False

        width_to_height = width / max(height, 1e-6)
        return (
            self.config.min_width_to_height_ratio
            <= width_to_height
            <= self.config.max_width_to_height_ratio
        )


def _lerp(start: float, end: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return start + (end - start) * alpha
