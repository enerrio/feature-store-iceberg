import os
from datetime import datetime
from pathlib import Path

import pytest

from src.anomaly_detector import ThresholdAnomalyDetector
from src.data_generator import generate_raw_events, ingest_raw_events
from src.feature_engineer import create_features
from src.feature_store import FeatureStore

TEST_CONFIG = {
    "NAMESPACE": "test",
    "N_EVENTS": 100,
    "N_USERS": 10,
    "DAYS": 7,
    "SEED": 42,
    "MODEL_VERSION": "vdebug",
    "REST_URI": "http://localhost:8182",
}


@pytest.fixture
def initialized_feature_store(test_iceberg_container, test_catalog):
    data = generate_raw_events(
        n_events=TEST_CONFIG["N_EVENTS"],
        n_users=TEST_CONFIG["N_USERS"],
        days=TEST_CONFIG["DAYS"],
        seed=TEST_CONFIG["SEED"],
    )
    ingest_raw_events(test_catalog, TEST_CONFIG["NAMESPACE"], data)

    sql_path = Path(__file__).parent.parent.parent / "sql" / "gapfilled_7day_spend.sql"
    create_features(
        namespace=TEST_CONFIG["NAMESPACE"],
        sql_path=str(sql_path),
        rest_uri=os.environ["ICEBERG_REST_URI"],
    )

    return FeatureStore(TEST_CONFIG["NAMESPACE"])


class TestAnomalyDebug:
    def test_debug_reports_training_snapshot(self, test_catalog, initialized_feature_store):
        feature_store = initialized_feature_store
        detector = ThresholdAnomalyDetector(feature_store)

        current_features = feature_store.feature_table.scan().to_arrow()
        dates = current_features.column("dt").to_pylist()
        first_dt = min(dates)
        last_dt = max(dates)
        target_user = current_features.column("user_id")[0].as_py()

        start = datetime.combine(first_dt, datetime.min.time())
        end = datetime.combine(last_dt, datetime.min.time())

        model_version = detector.train(
            start,
            end,
            model_version=TEST_CONFIG["MODEL_VERSION"],
        )

        metadata = feature_store.get_model_metadata(model_version)
        training_snapshot = metadata["feature_snapshot_id"]

        flagged_date = datetime.combine(last_dt, datetime.min.time())
        training_slice = feature_store.feature_table.scan(
            f"user_id = '{target_user}' and dt <= '{flagged_date.date()}'",
            snapshot_id=training_snapshot,
        ).to_arrow()
        expected_values = [
            float(x) for x in training_slice.column("spending_mean_7d").to_pylist()
        ]

        baseline = detector.detect(
            user_id=target_user,
            amount=expected_values[-1],
            as_of=flagged_date,
            model_version=model_version,
        )
        std = baseline.historical_std if baseline.historical_std > 0 else 1.0
        amount = baseline.historical_mean + (baseline.threshold_used + 1) * std

        additional = generate_raw_events(
            n_events=TEST_CONFIG["N_EVENTS"],
            n_users=TEST_CONFIG["N_USERS"],
            days=TEST_CONFIG["DAYS"],
            seed=TEST_CONFIG["SEED"] + 1,
        )
        ingest_raw_events(test_catalog, TEST_CONFIG["NAMESPACE"], additional)
        create_features(
            namespace=TEST_CONFIG["NAMESPACE"],
            sql_path=str(Path(__file__).parent.parent.parent / "sql" / "gapfilled_7day_spend.sql"),
            rest_uri=TEST_CONFIG["REST_URI"],
        )
        feature_store.feature_table.refresh()

        assert (
            feature_store.feature_table.current_snapshot().snapshot_id
            != training_snapshot
        )

        report = detector.debug_detection(
            user_id=target_user,
            flagged_date=flagged_date,
            actual_amount=amount,
            model_version=model_version,
            lookback_days=len(expected_values),
        )

        assert report["model_card"]["feature_snapshot_id"] == training_snapshot
        assert report["decision"]["is_anomaly"] is True
        assert report["context"]["as_of_feature_value"] == pytest.approx(
            expected_values[-1]
        )
        assert report["context"]["last_n_feature_values"] == pytest.approx(
            expected_values
        )
