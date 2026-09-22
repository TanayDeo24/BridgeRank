.PHONY: install test lint serve index evaluate gate docker
install:
	pip install -e ".[dev,ml,dashboard]"
test:
	pytest -q
lint:
	ruff check src tests
serve:
	uvicorn support_search.api:app --reload
index:
	python -m support_search.cli build-index
evaluate:
	python -m support_search.cli evaluate
gate: evaluate
	python -m support_search.cli quality-gate
docker:
	docker compose up --build
