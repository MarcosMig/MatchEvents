from __future__ import annotations

import numpy as np

from match_events.postprocessing import TrackTeamAssigner
from match_events.tracking.base import Track


def _make_frame() -> np.ndarray:
    frame = np.full((160, 240, 3), (30, 140, 30), dtype=np.uint8)
    frame[30:100, 20:50] = (20, 20, 220)
    frame[30:100, 70:100] = (25, 25, 215)
    frame[30:100, 130:160] = (220, 30, 30)
    frame[30:100, 180:210] = (215, 35, 35)
    frame[25:105, 100:120] = (30, 220, 220)
    frame[30:100, 150:180] = (215, 35, 35)
    return frame


def test_team_assigner_excludes_referee_and_labels_players_and_goalkeeper() -> None:
    assigner = TrackTeamAssigner.from_config(
        {
            "postprocessing": {
                "team_assignment": {
                    "enabled": True,
                    "player_observations_required": 2,
                    "player_max_distance": 0.20,
                    "goalkeeper_max_distance": 0.25,
                    "memory_size": 3,
                    "referee_cluster_max_share": 0.30,
                }
            }
        }
    )
    frame = _make_frame()

    result: list[Track] = []
    for frame_idx in range(3):
        tracks = [
            Track(frame_idx, 1, "player", 0.8, (20, 30, 50, 100)),
            Track(frame_idx, 2, "player", 0.8, (70, 30, 100, 100)),
            Track(frame_idx, 3, "player", 0.8, (130, 30, 160, 100)),
            Track(frame_idx, 4, "player", 0.8, (180, 30, 210, 100)),
            Track(frame_idx, 5, "referee", 0.8, (100, 25, 120, 105)),
            Track(frame_idx, 6, "goalkeeper", 0.8, (150, 30, 180, 100)),
        ]
        result = assigner.apply(frame, tracks)

    by_id = {track.track_id: track for track in result}
    assert by_id[1].team_id == "team_a"
    assert by_id[2].team_id == "team_a"
    assert by_id[3].team_id == "team_b"
    assert by_id[4].team_id == "team_b"
    assert by_id[5].team_id is None
    assert by_id[6].class_name == "goalkeeper"
    assert by_id[6].team_id == "team_b"
