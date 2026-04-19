from pathlib import Path
import argparse
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MatchEvents pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[MatchEvents] Pipeline bootstrap")
    print(f"Config: {config_path}")
    print(f"Input video: {config['input']['video_path']}")
    print(f"Output dir: {config['output']['output_dir']}")
    print("Next step: wire detector, tracker, and pitch mapper modules.")


if __name__ == "__main__":
    main()
