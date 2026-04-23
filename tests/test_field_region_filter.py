from __future__ import annotations

import numpy as np

from match_events.detectors.base import Detection
from match_events.postprocessing import FieldRegionFilter


def test_field_region_filter_removes_detections_outside_field() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    frame[30:, :] = (40, 160, 40)

    detections = [
        Detection(frame_idx=0, class_name="player", confidence=0.8, bbox_xyxy=(60, 50, 90, 110)),
        Detection(frame_idx=0, class_name="player", confidence=0.8, bbox_xyxy=(20, 5, 50, 25)),
        Detection(frame_idx=0, class_name="ball", confidence=0.3, bbox_xyxy=(100, 70, 108, 78)),
        Detection(frame_idx=0, class_name="ball", confidence=0.3, bbox_xyxy=(120, 10, 128, 18)),
    ]

    filter_ = FieldRegionFilter.from_config(
        {
            "postprocessing": {
                "field_mask": {
                    "enabled": True,
                    "min_area_ratio": 0.05,
                    "margin_px": 6,
                    "top_ignore_ratio": 0.1,
                }
            }
        }
    )

    filtered = filter_.apply(frame, detections)

    assert [d.class_name for d in filtered] == ["player", "ball"]
    assert [d.bbox_xyxy for d in filtered] == [(60, 50, 90, 110), (100, 70, 108, 78)]


def test_field_region_filter_rejects_person_boxes_with_implausible_perspective_size() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[:] = (40, 160, 40)

    detections = [
        Detection(frame_idx=0, class_name="player", confidence=0.8, bbox_xyxy=(40, 140, 70, 198)),
        Detection(frame_idx=0, class_name="player", confidence=0.8, bbox_xyxy=(120, 30, 150, 120)),
        Detection(frame_idx=0, class_name="player", confidence=0.8, bbox_xyxy=(200, 150, 260, 198)),
    ]

    filter_ = FieldRegionFilter.from_config(
        {
            "postprocessing": {
                "field_mask": {
                    "enabled": True,
                    "perspective_filter_enabled": True,
                    "near_min_height_ratio": 0.20,
                    "far_min_height_ratio": 0.10,
                    "near_max_height_ratio": 0.45,
                    "far_max_height_ratio": 0.20,
                    "min_width_to_height_ratio": 0.20,
                    "max_width_to_height_ratio": 0.80,
                }
            }
        }
    )

    filtered = filter_.apply(frame, detections)

    assert [d.bbox_xyxy for d in filtered] == [(40, 140, 70, 198)]
