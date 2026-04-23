from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass

from match_events.detectors.base import Detection


@dataclass(slots=True)
class Track:
    frame_idx: int
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    team_id: str | None = None

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


@dataclass(slots=True)
class _ActiveTrack:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    last_frame_idx: int
    missed_frames: int
    class_votes: deque[str]


class CentroidTracker(BaseTracker):
    """Small online tracker for football detections.

    It deliberately associates people class-agnostically, so an object can keep
    the same ID even if the detector flips between player/referee/goalkeeper.
    """

    PERSON_CLASSES = {"player", "referee", "goalkeeper"}

    def __init__(
        self,
        max_missed_frames: int = 15,
        max_center_distance: float = 80.0,
        min_iou: float = 0.05,
        class_history: int = 15,
        ball_max_missed_frames: int = 25,
        ball_max_center_distance: float = 140.0,
        ball_min_confidence: float = 0.20,
        ball_min_size: float = 3.0,
        ball_max_size: float = 28.0,
        ball_max_aspect_ratio: float = 2.0,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.max_center_distance = max_center_distance
        self.min_iou = min_iou
        self.class_history = class_history
        self.ball_max_missed_frames = ball_max_missed_frames
        self.ball_max_center_distance = ball_max_center_distance
        self.ball_min_confidence = ball_min_confidence
        self.ball_min_size = ball_min_size
        self.ball_max_size = ball_max_size
        self.ball_max_aspect_ratio = ball_max_aspect_ratio
        self._next_track_id = 1
        self._tracks: dict[int, _ActiveTrack] = {}
        self._ball_track_id: int | None = None

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        output_tracks: list[Track] = []

        ball_detections = [
            detection
            for detection in detections
            if detection.class_name == "ball"
            and detection.confidence >= self.ball_min_confidence
            and self._is_plausible_ball(detection)
        ]
        non_ball_detections = [
            detection
            for detection in detections
            if detection.class_name != "ball" or not self._is_plausible_ball(detection)
        ]

        candidate_pairs = self._build_candidate_pairs(non_ball_detections)
        for score, track_id, det_index in candidate_pairs:
            if track_id in matched_track_ids or det_index in matched_detection_indices:
                continue

            det = non_ball_detections[det_index]
            active = self._tracks[track_id]
            active.bbox_xyxy = det.bbox_xyxy
            active.confidence = det.confidence
            active.last_frame_idx = frame_idx
            active.missed_frames = 0
            active.class_votes.append(det.class_name)
            active.class_name = self._smoothed_class_name(active.class_votes)
            matched_track_ids.add(track_id)
            matched_detection_indices.add(det_index)
            output_tracks.append(self._to_track(active, frame_idx))

        for det_index, det in enumerate(non_ball_detections):
            if det_index in matched_detection_indices:
                continue
            active = _ActiveTrack(
                track_id=self._next_track_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox_xyxy=det.bbox_xyxy,
                last_frame_idx=frame_idx,
                missed_frames=0,
                class_votes=deque([det.class_name], maxlen=self.class_history),
            )
            self._tracks[active.track_id] = active
            self._next_track_id += 1
            output_tracks.append(self._to_track(active, frame_idx))

        ball_track = self._update_ball(ball_detections, frame_idx)
        if ball_track is not None:
            output_tracks.append(ball_track)

        stale_ids: list[int] = []
        for track_id, active in self._tracks.items():
            if track_id in matched_track_ids or active.last_frame_idx == frame_idx:
                continue
            active.missed_frames += 1
            if active.missed_frames > self.max_missed_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._tracks[track_id]

        return output_tracks

    def _update_ball(self, detections: list[Detection], frame_idx: int) -> Track | None:
        active = self._tracks.get(self._ball_track_id) if self._ball_track_id is not None else None
        detection = self._select_ball_detection(detections, active)

        if detection is None:
            if active is None:
                return None
            active.missed_frames += 1
            if active.missed_frames > self.ball_max_missed_frames:
                if self._ball_track_id in self._tracks:
                    del self._tracks[self._ball_track_id]
                self._ball_track_id = None
            return None

        if active is None:
            active = _ActiveTrack(
                track_id=self._next_track_id,
                class_name="ball",
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
                last_frame_idx=frame_idx,
                missed_frames=0,
                class_votes=deque(["ball"], maxlen=self.class_history),
            )
            self._tracks[active.track_id] = active
            self._ball_track_id = active.track_id
            self._next_track_id += 1
        else:
            active.bbox_xyxy = detection.bbox_xyxy
            active.confidence = detection.confidence
            active.last_frame_idx = frame_idx
            active.missed_frames = 0

        return self._to_track(active, frame_idx)

    def _select_ball_detection(
        self,
        detections: list[Detection],
        active: _ActiveTrack | None,
    ) -> Detection | None:
        if not detections:
            return None
        if active is None:
            return max(detections, key=lambda detection: detection.confidence)

        candidates: list[tuple[float, Detection]] = []
        for detection in detections:
            center_distance = _center_distance(active.bbox_xyxy, detection.bbox_xyxy)
            if center_distance > self.ball_max_center_distance:
                continue
            score = detection.confidence - (center_distance / max(self.ball_max_center_distance, 1.0))
            candidates.append((score, detection))

        if not candidates:
            return max(detections, key=lambda detection: detection.confidence)
        return max(candidates, key=lambda item: item[0])[1]

    def _is_plausible_ball(self, detection: Detection) -> bool:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width < self.ball_min_size or height < self.ball_min_size:
            return False
        if width > self.ball_max_size or height > self.ball_max_size:
            return False
        aspect_ratio = max(width / max(height, 1e-6), height / max(width, 1e-6))
        return aspect_ratio <= self.ball_max_aspect_ratio

    def _build_candidate_pairs(self, detections: list[Detection]) -> list[tuple[float, int, int]]:
        pairs: list[tuple[float, int, int]] = []
        for track_id, active in self._tracks.items():
            for det_index, det in enumerate(detections):
                if not self._classes_can_match(active.class_name, det.class_name):
                    continue
                center_distance = _center_distance(active.bbox_xyxy, det.bbox_xyxy)
                iou = _iou(active.bbox_xyxy, det.bbox_xyxy)
                if center_distance > self.max_center_distance and iou < self.min_iou:
                    continue
                score = iou - (center_distance / max(self.max_center_distance, 1.0))
                pairs.append((score, track_id, det_index))
        return sorted(pairs, reverse=True)

    def _classes_can_match(self, track_class: str, detection_class: str) -> bool:
        if track_class == detection_class:
            return True
        return track_class in self.PERSON_CLASSES and detection_class in self.PERSON_CLASSES

    @staticmethod
    def _smoothed_class_name(class_votes: deque[str]) -> str:
        return Counter(class_votes).most_common(1)[0][0]

    @staticmethod
    def _to_track(active: _ActiveTrack, frame_idx: int) -> Track:
        return Track(
            frame_idx=frame_idx,
            track_id=active.track_id,
            class_name=active.class_name,
            confidence=active.confidence,
            bbox_xyxy=active.bbox_xyxy,
        )


def _center_distance(
    first_bbox: tuple[float, float, float, float],
    second_bbox: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    first_center = ((first_x1 + first_x2) / 2.0, (first_y1 + first_y2) / 2.0)
    second_center = ((second_x1 + second_x2) / 2.0, (second_y1 + second_y2) / 2.0)
    return (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5


def _iou(
    first_bbox: tuple[float, float, float, float],
    second_bbox: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0.0:
        return 0.0

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
