from pathlib import Path
import argparse
import yaml

from match_events.detectors import build_detector
from match_events.pipeline import MatchEventsPipeline


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
    pipeline = MatchEventsPipeline(detector=detector)
    outputs = pipeline.run(video_path=video_path, output_dir=output_dir)

    print("[MatchEvents] Run completed")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
