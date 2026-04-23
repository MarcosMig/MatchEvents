from __future__ import annotations

from dataclasses import dataclass

from match_events.tracking.base import Track


@dataclass(slots=True)
class PossessionConfig:
    enabled: bool = False
    max_ball_distance: float = 140.0
    carry_frames: int = 8


class PossessionEstimator:
    PLAYER_CLASSES = {"player", "goalkeeper"}

    def __init__(self, config: PossessionConfig | None = None) -> None:
        self.config = config or PossessionConfig()

    @classmethod
    def from_config(cls, raw_config: dict) -> "PossessionEstimator":
        possession_cfg = raw_config.get("analytics", {}).get("possession", {})
        return cls(
            PossessionConfig(
                enabled=bool(possession_cfg.get("enabled", False)),
                max_ball_distance=float(possession_cfg.get("max_ball_distance", 140.0)),
                carry_frames=int(possession_cfg.get("carry_frames", 8)),
            )
        )

    def estimate(self, tracks_by_frame: dict[int, list[Track]]) -> list[dict]:
        if not self.config.enabled:
            return []

        rows: list[dict] = []
        last_team_id: str | None = None
        last_track_id: int | None = None
        last_frame_idx: int | None = None

        for frame_idx in sorted(tracks_by_frame):
            tracks = tracks_by_frame[frame_idx]
            ball_tracks = [track for track in tracks if track.class_name == "ball"]
            candidates = [
                track
                for track in tracks
                if track.class_name in self.PLAYER_CLASSES and track.team_id is not None
            ]
            best = self._best_candidate(ball_tracks, candidates)
            possession_team_id: str | None = None
            possession_track_id: int | None = None
            ball_distance: float | None = None
            source = "none"

            if best is not None:
                possessor, ball_track, ball_distance = best
                possession_team_id = possessor.team_id
                possession_track_id = possessor.track_id
                last_team_id = possession_team_id
                last_track_id = possession_track_id
                last_frame_idx = frame_idx
                source = "direct"
            elif (
                last_team_id is not None
                and last_frame_idx is not None
                and frame_idx - last_frame_idx <= self.config.carry_frames
            ):
                possession_team_id = last_team_id
                possession_track_id = last_track_id
                source = "carry"

            rows.append(
                {
                    "frame_idx": frame_idx,
                    "possession_team_id": possession_team_id,
                    "possessor_track_id": possession_track_id,
                    "ball_distance": ball_distance,
                    "source": source,
                }
            )

        return rows

    def _best_candidate(
        self,
        ball_tracks: list[Track],
        candidates: list[Track],
    ) -> tuple[Track, Track, float] | None:
        if not ball_tracks or not candidates:
            return None

        best_pair: tuple[Track, Track, float] | None = None
        for ball_track in ball_tracks:
            for candidate in candidates:
                distance = _center_distance(ball_track.center_xy, candidate.center_xy)
                if distance > self.config.max_ball_distance:
                    continue
                if best_pair is None or distance < best_pair[2]:
                    best_pair = (candidate, ball_track, distance)
        return best_pair


def _center_distance(
    first_center: tuple[float, float],
    second_center: tuple[float, float],
) -> float:
    return (
        (first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
