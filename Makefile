.PHONY: install test run lint format

install:
	pip install -U pip
	pip install -e .[dev]

test:
	pytest -q

run:
	python scripts/run_pipeline.py --config configs/base.yaml

lint:
	ruff check src tests scripts

format:
	black src tests scripts
