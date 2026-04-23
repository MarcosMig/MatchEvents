from __future__ import annotations

from match_events.analytics import PossessionEstimator
from match_events.tracking.base import Track


def test_possession_estimator_assigns_nearest_player_and_carries_short_gaps() -> None:
    estimator = PossessionEstimator.from_config(
        {
            "analytics": {
                "possession": {
                    "enabled": True,
                    "max_ball_distance": 30.0,
                    "carry_frames": 2,
                }
            }
        }
    )

    tracks_by_frame = {
        0: [
            Track(0, 1, "player", 0.8, (10, 10, 30, 50), "team_a"),
            Track(0, 2, "player", 0.8, (100, 10, 120, 50), "team_b"),
            Track(0, 3, "ball", 0.8, (18, 40, 24, 46)),
        ],
        1: [
            Track(1, 1, "player", 0.8, (10, 10, 30, 50), "team_a"),
            Track(1, 2, "player", 0.8, (100, 10, 120, 50), "team_b"),
        ],
        2: [
            Track(2, 1, "player", 0.8, (10, 10, 30, 50), "team_a"),
            Track(2, 2, "player", 0.8, (100, 10, 120, 50), "team_b"),
        ],
        3: [
            Track(3, 1, "player", 0.8, (10, 10, 30, 50), "team_a"),
            Track(3, 2, "player", 0.8, (100, 10, 120, 50), "team_b"),
            Track(3, 4, "ball", 0.8, (102, 40, 108, 46)),
        ],
    }

    rows = estimator.estimate(tracks_by_frame)

    assert rows[0]["possession_team_id"] == "team_a"
    assert rows[0]["source"] == "direct"
    assert rows[1]["possession_team_id"] == "team_a"
    assert rows[1]["source"] == "carry"
    assert rows[2]["possession_team_id"] == "team_a"
    assert rows[2]["source"] == "carry"
    assert rows[3]["possession_team_id"] == "team_b"
    assert rows[3]["source"] == "direct"
