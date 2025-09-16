import os
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.compute as pc
import pytest

from src.data_generator import generate_raw_events, ingest_raw_events
from src.feature_engineer import create_features
from src.feature_store import FeatureStore

TEST_CONFIG = {
    "NAMESPACE": "test",
    "N_EVENTS": 100,
    "N_USERS": 10,
    "DAYS": 7,
    "SEED": 42,
    "MODEL_VERSION": "v1.2.3",
    "REST_URI": "http://localhost:8182",
}


@pytest.fixture
def initialized_feature_store(test_iceberg_container, test_catalog):
    """Set up tables with data before creating FeatureStore"""
    # Create raw events
    data = generate_raw_events(
        n_events=TEST_CONFIG["N_EVENTS"],
        n_users=TEST_CONFIG["N_USERS"],
        days=TEST_CONFIG["DAYS"],
        seed=TEST_CONFIG["SEED"],
    )
    ingest_raw_events(test_catalog, TEST_CONFIG["NAMESPACE"], data)

    # Create features
    sql_path = Path(__file__).parent.parent.parent / "sql" / "gapfilled_7day_spend.sql"
    create_features(
        namespace=TEST_CONFIG["NAMESPACE"],
        sql_path=str(sql_path),
        rest_uri=os.environ["ICEBERG_REST_URI"],
    )

    # Now safe to create FeatureStore
    return FeatureStore(TEST_CONFIG["NAMESPACE"])


class TestTimeTravel:
    def test_invalid_model_version(self, initialized_feature_store):
        """Verify that no data is returned for a nonexistent model version."""
        with pytest.raises(ValueError):
            initialized_feature_store.get_training_data(
                "v1.3.4", datetime.now() - timedelta(days=29), datetime.now()
            )

    def test_time_travel_reproducibility(self, test_catalog, initialized_feature_store):
        """Verify that get_training_data returns same data despite new appends."""
        current_snapshot = initialized_feature_store.get_current_snapshot_ids()
        window_end = datetime.now()
        window_start = window_end - timedelta(days=29)
        initialized_feature_store.register_model_training(
            TEST_CONFIG["MODEL_VERSION"],
            datetime.now().astimezone(),
            current_snapshot["features"],
            current_snapshot["raw_events"],
            feature_name="spending_mean_7d",
            decision_threshold=0.6,
            training_window_start=window_start.date(),
            training_window_end=window_end.date(),
            quantile=0.995,
        )
        # Get training data - save for comparison
        initial_data = initialized_feature_store.get_training_data(
            TEST_CONFIG["MODEL_VERSION"],
            window_start,
            window_end,
        )
        initial_row_count = initial_data.num_rows

        # Add MORE data to both tables
        data = generate_raw_events(
            n_events=TEST_CONFIG["N_EVENTS"],
            n_users=TEST_CONFIG["N_USERS"],
            days=TEST_CONFIG["DAYS"],
            seed=21,
        )
        ingest_raw_events(test_catalog, "test", data)
        create_features(
            namespace=TEST_CONFIG["NAMESPACE"],
            sql_path=Path(__file__).parent.parent.parent
            / "sql"
            / "gapfilled_7day_spend.sql",
            rest_uri=TEST_CONFIG["REST_URI"],
        )

        # Get training data again with same model version
        later_data = initialized_feature_store.get_training_data(
            TEST_CONFIG["MODEL_VERSION"],
            window_start,
            window_end,
        )

        assert later_data.num_rows == initial_row_count
        initial_sum = initial_data.column("spending_mean_7d").to_numpy().sum()
        later_sum = later_data.column("spending_mean_7d").to_numpy().sum()
        assert initial_sum == later_sum

        initial_user_7 = (
            initial_data.filter(pc.equal(initial_data.column("user_id"), 7))
            .column("spending_mean_7d")[0]
            .as_py()
        )
        later_user_7 = (
            later_data.filter(pc.equal(later_data.column("user_id"), 7))
            .column("spending_mean_7d")[0]
            .as_py()
        )
        assert initial_user_7 == later_user_7

        # Get ALL data
        current_features = initialized_feature_store.feature_table.scan().to_arrow()
        assert current_features.num_rows > initial_row_count
