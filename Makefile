# ------------------------------------------------------------------------------
NAMESPACE ?= default
N_EVENTS  ?= 1000
N_USERS   ?= 100
DAYS      ?= 30
SEED      ?= 7
BATCHES   ?= 2
ICEBERG_REST_URI ?= http://localhost:8181
FEATURES_SQL ?= sql/gapfilled_7day_spend.sql
# ------------------------------------------------------------------------------

.PHONY: help lint format test data features
PY := .venv/bin/python
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

help:
	@echo "Available targets:"
	@echo "  lint     - Run ruff linter"
	@echo "  format   - Format code using ruff"
	@echo "  test     - Run tests with coverage"
	@echo "  data     - Generate synthetic raw events and ingest into Iceberg"
	@echo "  features - Compute and materialize features with pure SQL via DuckDB+Iceberg"

lint:
	$(RUFF) check

format:
	$(RUFF) check --fix
	$(RUFF) format .

test:
	$(PYTEST) tests/ --cov=. --cov-report=term

data:
	$(PY) -m src.data_generator generate \
		--namespace $(NAMESPACE) \
		--n-events $(N_EVENTS) \
		--n-users $(N_USERS) \
		--days $(DAYS) \
		--seed $(SEED) \
		--batches $(BATCHES) \
		$(if $(ANOMALIES_JSON),--anomalies-json '$(ANOMALIES_JSON)',) \
		$(if $(ANOM_FILE),--anomaly-file '$(ANOM_FILE)',)

features:
	$(PY) -m src.feature_engineer \
		--namespace $(NAMESPACE) \
		--sql $(FEATURES_SQL) \
		--rest-uri $(ICEBERG_REST_URI) \
