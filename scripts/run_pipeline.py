from pathlib import Path
import argparse
import yaml

from match_events.detectors import build_detector
from match_events.pipeline import MatchEventsPipeline
from match_events.postprocessing import BallClassCorrector, FrameRoleCorrector
from match_events.tracking import build_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MatchEvents pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_path = config["input"]["video_path"]
    output_dir = config["output"]["output_dir"]

    detector = build_detector(config)
    tracker = build_tracker(config)
    role_corrector = FrameRoleCorrector.from_config(config)
    ball_corrector = BallClassCorrector.from_config(config)
    pipeline = MatchEventsPipeline(
        detector=detector,
        tracker=tracker,
        role_corrector=role_corrector,
        ball_corrector=ball_corrector,
        interpolate_ball_tracks=bool(
            config.get("tracking", {}).get("ball", {}).get("interpolate", False)
        ),
        ball_interpolation_max_gap=int(
            config.get("tracking", {}).get("ball", {}).get("max_gap_frames", 12)
        ),
    )
    pipeline.start_frame = config.get("input", {}).get("start_frame")
    pipeline.end_frame = config.get("input", {}).get("end_frame")
    pipeline.start_time_seconds = config.get("input", {}).get("start_time_seconds")
    pipeline.end_time_seconds = config.get("input", {}).get("end_time_seconds")
    outputs = pipeline.run(video_path=video_path, output_dir=output_dir)

    print("[MatchEvents] Run completed")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
