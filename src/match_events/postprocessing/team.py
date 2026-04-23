from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from match_events.tracking.base import Track


@dataclass(slots=True)
class TeamAssignerConfig:
    enabled: bool = False
    player_observations_required: int = 40
    player_max_distance: float = 0.16
    goalkeeper_max_distance: float = 0.22
    saturation_threshold: int = 35
    value_threshold: int = 35
    memory_size: int = 3
    referee_cluster_max_share: float = 0.22
    green_hue_min: int = 28
    green_hue_max: int = 95
    green_suppression_saturation: int = 30
    green_suppression_value: int = 30


@dataclass(slots=True)
class _TeamCluster:
    center: np.ndarray
    observations: int


class TrackTeamAssigner:
    TEAM_IDS = ("team_a", "team_b")

    def __init__(self, config: TeamAssignerConfig | None = None) -> None:
        self.config = config or TeamAssignerConfig()
        self._clusters: list[_TeamCluster] = []

    @classmethod
    def from_config(cls, raw_config: dict) -> "TrackTeamAssigner":
        team_cfg = raw_config.get("postprocessing", {}).get("team_assignment", {})
        return cls(
            TeamAssignerConfig(
                enabled=bool(team_cfg.get("enabled", False)),
                player_observations_required=int(team_cfg.get("player_observations_required", 40)),
                player_max_distance=float(team_cfg.get("player_max_distance", 0.16)),
                goalkeeper_max_distance=float(team_cfg.get("goalkeeper_max_distance", 0.22)),
                saturation_threshold=int(team_cfg.get("saturation_threshold", 35)),
                value_threshold=int(team_cfg.get("value_threshold", 35)),
                memory_size=int(team_cfg.get("memory_size", 3)),
                referee_cluster_max_share=float(team_cfg.get("referee_cluster_max_share", 0.22)),
                green_hue_min=int(team_cfg.get("green_hue_min", 28)),
                green_hue_max=int(team_cfg.get("green_hue_max", 95)),
                green_suppression_saturation=int(
                    team_cfg.get("green_suppression_saturation", 30)
                ),
                green_suppression_value=int(team_cfg.get("green_suppression_value", 30)),
            )
        )

    def apply(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        if not self.config.enabled or not tracks:
            return tracks

        corrected = list(tracks)

        player_features: list[tuple[int, int, np.ndarray]] = []
        for index, track in enumerate(tracks):
            if track.class_name != "player":
                continue
            feature = self._shirt_feature(frame, track.bbox_xyxy)
            if feature is None:
                continue
            cluster_id = self._observe_player_feature(feature)
            player_features.append((index, cluster_id, feature))

        reliable_clusters = self._reliable_clusters()
        if len(reliable_clusters) < 2:
            return corrected

        cluster_to_team, referee_clusters = self._cluster_roles(reliable_clusters)

        for index, cluster_id, _feature in player_features:
            if cluster_id in referee_clusters:
                corrected[index] = replace(corrected[index], class_name="referee", team_id=None)
                continue
            if cluster_id not in cluster_to_team:
                continue
            corrected[index] = replace(corrected[index], team_id=cluster_to_team[cluster_id])

        for index, track in enumerate(corrected):
            if track.class_name != "goalkeeper":
                continue
            feature = self._shirt_feature(frame, track.bbox_xyxy)
            if feature is None:
                continue
            cluster_id, distance = self._nearest_cluster(feature)
            if cluster_id is None or cluster_id in referee_clusters or cluster_id not in cluster_to_team:
                corrected[index] = self._assign_goalkeeper_by_neighbors(track, corrected)
                continue
            if distance > self.config.goalkeeper_max_distance:
                corrected[index] = self._assign_goalkeeper_by_neighbors(track, corrected)
                continue
            corrected[index] = replace(corrected[index], team_id=cluster_to_team[cluster_id])

        return corrected

    def _assign_goalkeeper_by_neighbors(self, goalkeeper: Track, tracks: list[Track]) -> Track:
        teammates = [
            track
            for track in tracks
            if track.class_name == "player" and track.team_id is not None
        ]
        if not teammates:
            return goalkeeper

        nearest = min(
            teammates,
            key=lambda track: _center_distance(track.center_xy, goalkeeper.center_xy),
        )
        return replace(goalkeeper, team_id=nearest.team_id)

    def _observe_player_feature(self, feature: np.ndarray) -> int:
        cluster_id, distance = self._nearest_cluster(feature)
        if cluster_id is None:
            self._clusters.append(_TeamCluster(center=feature.copy(), observations=1))
            return 0

        if (
            distance > self.config.player_max_distance
            and len(self._clusters) < self.config.memory_size
        ):
            self._clusters.append(_TeamCluster(center=feature.copy(), observations=1))
            return len(self._clusters) - 1

        cluster = self._clusters[cluster_id]
        cluster.observations += 1
        learning_rate = 1.0 / min(cluster.observations, 200)
        cluster.center = (1.0 - learning_rate) * cluster.center + learning_rate * feature
        return cluster_id

    def _nearest_cluster(self, feature: np.ndarray) -> tuple[int | None, float]:
        if not self._clusters:
            return None, float("inf")
        distances = [
            float(np.linalg.norm(cluster.center - feature)) for cluster in self._clusters
        ]
        cluster_id = int(np.argmin(distances))
        return cluster_id, distances[cluster_id]

    def _reliable_clusters(self) -> list[int]:
        reliable = [
            cluster_id
            for cluster_id, cluster in enumerate(self._clusters)
            if cluster.observations >= self.config.player_observations_required
        ]
        return reliable[: self.config.memory_size]

    def _cluster_roles(self, cluster_ids: list[int]) -> tuple[dict[int, str], set[int]]:
        if len(cluster_ids) <= 2:
            return {
                cluster_id: team_id for cluster_id, team_id in zip(cluster_ids[:2], self.TEAM_IDS)
            }, set()

        total_observations = sum(self._clusters[cluster_id].observations for cluster_id in cluster_ids)
        sorted_clusters = sorted(
            cluster_ids,
            key=lambda cluster_id: self._clusters[cluster_id].observations,
            reverse=True,
        )
        candidate_referee = sorted_clusters[-1]
        share = self._clusters[candidate_referee].observations / max(total_observations, 1)
        referee_clusters: set[int] = set()
        team_clusters = sorted_clusters
        if share <= self.config.referee_cluster_max_share:
            referee_clusters.add(candidate_referee)
            team_clusters = sorted_clusters[:-1]

        cluster_to_team = {
            cluster_id: team_id for cluster_id, team_id in zip(team_clusters[:2], self.TEAM_IDS)
        }
        return cluster_to_team, referee_clusters

    def _shirt_feature(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 + 2 or y2 <= y1 + 4:
            return None

        box_width = x2 - x1
        box_height = y2 - y1
        torso_x1 = x1 + int(box_width * 0.28)
        torso_x2 = x2 - int(box_width * 0.28)
        torso_y1 = y1 + int(box_height * 0.18)
        torso_y2 = y1 + int(box_height * 0.48)
        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        saturation = pixels[:, 1]
        value = pixels[:, 2]
        useful_pixels = pixels[
            (saturation > self.config.saturation_threshold)
            & (value > self.config.value_threshold)
        ]
        non_field_pixels = useful_pixels[
            ~(
                (useful_pixels[:, 0] >= self.config.green_hue_min)
                & (useful_pixels[:, 0] <= self.config.green_hue_max)
                & (useful_pixels[:, 1] >= self.config.green_suppression_saturation)
                & (useful_pixels[:, 2] >= self.config.green_suppression_value)
            )
        ]
        if len(non_field_pixels) >= 8:
            useful_pixels = non_field_pixels
        if len(useful_pixels) < 8:
            useful_pixels = pixels

        median_hsv = np.median(useful_pixels, axis=0)
        hue_std = float(np.std(useful_pixels[:, 0] / 179.0))
        return np.array(
            [
                median_hsv[0] / 179.0,
                median_hsv[1] / 255.0,
                median_hsv[2] / 255.0,
                hue_std,
            ],
            dtype=np.float32,
        )


def _center_distance(
    first_center: tuple[float, float],
    second_center: tuple[float, float],
) -> float:
    return (
        (first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
