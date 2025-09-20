# Feature Store with Iceberg

This repo accompanies the [What's a Feature Store?](TODO) blog post. The repo implements a feature store using Apache Iceberg to demonstrate their practical use. It imagines a data table of raw financial transaction data with anomalies injected. The data is transformed into features that can be used to train a predictive model. Both raw transactions and features are stored in separate relational tables managed by Apache Iceberg. Finally, a model is trained (utilizing a global z-score) and its relevant materials, like exact training data snapshot, are stored in another table. The feature store can be used to get the **exact** training data used to train a specific model and even used for debugging models.

Here's a short description of different parts of the repo:
* anomaly_files/ - JSON files that dictate anomalies to inject during data generation
* notebooks/ - Exploratory notebooks for querying Apache Iceberg tables
* sql/ - SQL code for feature engineering
* src/ - Main source code (data generation, feature engineering, anomaly detector calibration, feature store implementation)
* tests/ - Integration tests
* docker-compose.yml - Config file for creating a Docker container with Apache Iceberg running inside it
* docker-compose.test.yml - Similar to above but only for use with integration tests
* Makefile - Shortcuts for different stages of the pipeline (see `Usage` section)

The next few sections talk briefly about some background information for each stage of the pipeline and ends with some usage instructions.

## Data generation
Data is generated according to a lognormal distribution to simulate customer spending. The result is stored in a simple `raw_events` table with the following schema:

| Column Name     | Data Type      | Description                          |
| --------------- | -------------- | ------------------------------------ |
| id              | BIGINT         | Unique identifier for the event      |
| user_id         | BIGINT         | ID of the user associated with event |
| amount          | DECIMAL(15, 2) | Dollar amount spent                  |
| vendor_id       | BIGINT         | Vendor where transaction occurred    |
| event_timestamp | TIMESTAMP      | When the event occurred in UTC       |

## Feature computation
Now we need a data table that stores the features used to train the model. The schema for this `user_features` table looks like this but can evolve over time:

| Column Name      | Data Type      | Description                               |
| ---------------- | -------------- | ----------------------------------------- |
| user_id          | BIGINT         | ID of the user associated with event      |
| dt               | DATE           | date of last day in 7 day time period     |
| spending_mean_7d | DECIMAL(15, 2) | Average spending over a 7 day time period |

## Feature store
We also need a table to track the models that are trained and the data they are trained on. It will be called `model_training_metadata` and the schema is:

| Column Name            | Data Type | Description                                                 |
| ---------------------- | --------- | ----------------------------------------------------------- |
| model_version          | VARCHAR   | Model version number                                        |
| trained_at             | TIMESTAMP | Time that model completed training                          |
| feature_snapshot_id    | BIGINT    | Snapshot ID of `user_features` used for training            |
| raw_events_snapshot_id | BIGINT    | Snapshot ID of `raw_events` used for creating features      |
| feature_name           | VARCHAR   | Feature column used when calibrating the threshold          |
| decision_threshold     | DOUBLE    | Final \|z\|-score threshold chosen from the training window |
| training_window_start  | DATE      | First day of the training window                            |
| training_window_end    | DATE      | Last day of the training window                             |
| quantile               | DOUBLE    | Quantile used to pick the threshold (optional)              |

## Usage
A Makefile is included to make it easier to run different steps. Before getting started make sure you have [uv](https://docs.astral.sh/uv/) installed and create your virtual environment via `uv sync`. Also make sure you have Docker installed, we set up an Apache Iceberg server inside a docker container. First startup the Docker container:
```bash
# Launch Iceberg in a docker container. Detach so you can run other commands
docker compose up -d

# Tear down the docker container when you're done. Also remove any attached volumes
docker compose down -v
```

Here are some Makefile usage examples.
```bash
# Generate fresh raw events
make data

# Generate fresh raw events with some anomalies injected for a single user
make data ANOM_FILE=anomaly_files/single_user.json

# Create the user feature table and populate with transformed raw events
make features

# Calibrate and register a model
make train-model

# Make a prediction for a specific transaction amount (get model version from `make train-model` output)
make detect AMOUNT=50 MODEL_VERSION=v20250916224417

# Explain a specific transaction
make debug USER_ID=42 AMOUNT=275 MODEL_VERSION=v20250916224417
```

## Pyiceberg CLI Cheat Sheet

```bash
# List namespaces
uv run pyiceberg --uri http://localhost:8181 list

# Get full details including snapshot ids
uv run pyiceberg --uri http://localhost:8181 describe default.raw_events

# Get schema details
uv run pyiceberg --uri http://localhost:8181 schema

# Get partition info
uv run pyiceberg --uri http://localhost:8181 spec

# Delete a table
uv run pyiceberg --uri http://localhost:8181 drop table default.raw_events

# Delete a namespace
uv run pyiceberg --uri http://localhost:8181 drop namespace default
```
