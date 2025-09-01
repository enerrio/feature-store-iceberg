.PHONY: help lint format test
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

help:
	@echo "Available targets:"
	@echo "  lint    - Run ruff linter"
	@echo "  format  - Format code using ruff"
	@echo "  test    - Run tests with coverage"

lint:
	$(RUFF) check

format:
	$(RUFF) check --fix
	$(RUFF) format .

test:
	$(PYTEST) tests/ --cov=. --cov-report=term
