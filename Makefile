# ------------------------------------------------------------------------------
NAMESPACE ?= default
N_EVENTS  ?= 1000
N_USERS   ?= 100
DAYS      ?= 30
SEED      ?= 7
BATCHES   ?= 2
ICEBERG_REST_URI ?= http://localhost:8181
ICEBERG_WAREHOUSE ?= file:///tmp/warehouse
FEATURES_SQL ?= sql/gapfilled_7day_spend.sql
START_DATE ?=
END_DATE   ?=
WINDOW_DAYS ?= 30
FEATURE_NAME ?= spending_mean_7d
QUANTILE ?= 0.995
MODEL_VERSION ?=
USER_ID ?= 42
AMOUNT ?= 250.0
AS_OF ?=
LOOKBACK_DAYS ?= 14
# ------------------------------------------------------------------------------

.PHONY: help lint format test data features train-model detect debug
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
	@echo "  train-model - Calibrate anomaly detector threshold and register metadata"
	@echo "  detect   - Score a single transaction using a registered model"
	@echo "  debug    - Explain a transaction with model snapshot provenance"

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

train-model:
	$(PY) -m src.anomaly_detector --namespace $(NAMESPACE) train \
		$(if $(MODEL_VERSION),--model-version $(MODEL_VERSION),) \
		$(if $(START_DATE),--start-date $(START_DATE),) \
		$(if $(END_DATE),--end-date $(END_DATE),) \
		--window-days $(WINDOW_DAYS) \
		--feature-name $(FEATURE_NAME) \
		--quantile $(QUANTILE)

detect:
	@if [ -z "$(MODEL_VERSION)" ]; then \
		echo "Error: MODEL_VERSION is required (e.g. make detect MODEL_VERSION=v20250101...)" ; \
		exit 1; \
	fi
	$(PY) -m src.anomaly_detector --namespace $(NAMESPACE) detect \
		--model-version $(MODEL_VERSION) \
		--user-id $(USER_ID) \
		--amount $(AMOUNT) \
		$(if $(AS_OF),--as-of $(AS_OF),)

debug:
	@if [ -z "$(MODEL_VERSION)" ]; then \
		echo "Error: MODEL_VERSION is required (e.g. make debug MODEL_VERSION=v20250101...)" ; \
		exit 1; \
	fi
	$(PY) -m src.anomaly_detector --namespace $(NAMESPACE) debug \
		--model-version $(MODEL_VERSION) \
		--user-id $(USER_ID) \
		--amount $(AMOUNT) \
		--lookback-days $(LOOKBACK_DAYS) \
		$(if $(AS_OF),--as-of $(AS_OF),)
