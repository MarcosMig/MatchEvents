from match_events.detectors import StubDetector, YoloDetector, build_detector
from match_events.io import VideoReader
from match_events.pipeline import MatchEventsPipeline
from match_events.postprocessing import FieldRegionFilter, TrackTeamAssigner
from match_events.tracking import StubTracker


def test_imports() -> None:
    assert StubDetector is not None
    assert YoloDetector is not None
    assert build_detector is not None
    assert StubTracker is not None
    assert VideoReader is not None
    assert MatchEventsPipeline is not None
    assert FieldRegionFilter is not None
    assert TrackTeamAssigner is not None
