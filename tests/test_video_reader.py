from __future__ import annotations

from pathlib import Path

from match_events.io import VideoReader


SAMPLE_VIDEO = Path("data/raw/sample.mp4")


def test_video_reader_limits_frames_by_index() -> None:
    full_reader = VideoReader(SAMPLE_VIDEO)
    full_metadata = full_reader.get_metadata()

    reader = VideoReader(SAMPLE_VIDEO, start_frame=2, end_frame=6)

    metadata = reader.get_metadata()
    frame_indices = [frame_idx for frame_idx, _frame in reader.frames()]

    assert full_metadata.frame_count >= 6
    assert metadata.frame_count == 4
    assert frame_indices == [2, 3, 4, 5]


def test_video_reader_limits_frames_by_time() -> None:
    full_reader = VideoReader(SAMPLE_VIDEO)
    full_metadata = full_reader.get_metadata()

    reader = VideoReader(SAMPLE_VIDEO, start_time_seconds=0.4, end_time_seconds=1.4)

    metadata = reader.get_metadata()
    frame_indices = [frame_idx for frame_idx, _frame in reader.frames()]

    assert full_metadata.fps == 25.0
    assert metadata.frame_count == 25
    assert frame_indices == list(range(10, 35))
