from datetime import date, datetime
from typing import Optional

import pyarrow as pa
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform
from pyiceberg.types import (
    DateType,
    FloatType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)
from rich import print

from .catalog import get_catalog


class FeatureStore:
    def __init__(self, namespace: str):
        catalog = get_catalog()
        self.schema = Schema(
            NestedField(1, "model_version", StringType(), required=True),
            NestedField(2, "trained_at", TimestamptzType(), required=True),
            NestedField(3, "feature_snapshot_id", LongType(), required=True),
            NestedField(4, "raw_events_snapshot_id", LongType(), required=True),
            NestedField(5, "feature_name", StringType(), required=True),
            NestedField(6, "decision_threshold", FloatType(), required=True),
            NestedField(7, "training_window_start", DateType(), required=True),
            NestedField(8, "training_window_end", DateType(), required=True),
            NestedField(9, "quantile", FloatType(), required=False),
        )

        ts_id = self.schema.find_field("trained_at").field_id
        spec = PartitionSpec(
            PartitionField(
                source_id=ts_id, field_id=100, transform=DayTransform(), name="dt"
            )
        )

        model_table_name = f"{namespace}.model_training_metadata"
        self.model_table = catalog.create_table_if_not_exists(
            model_table_name, schema=self.schema, partition_spec=spec
        )
        # Reload table and schema to get canonical version of table/schema. Avoids any issues with appending
        self.model_table = catalog.load_table(model_table_name)
        self.schema = self.model_table.schema()

        features_name = f"{namespace}.user_features"
        self.feature_table = catalog.load_table(features_name)
        raw_events_name = f"{namespace}.raw_events"
        self.raw_events_table = catalog.load_table(raw_events_name)

    def register_model_training(
        self,
        model_version: str,
        trained_at: datetime,
        feature_snapshot_id: int,
        raw_events_snapshot_id: int,
        *,
        feature_name: str,
        decision_threshold: float,
        training_window_start: date,
        training_window_end: date,
        quantile: Optional[float] = None,
    ):
        """Record that model X was trained with snapshot Y and calibrated threshold."""
        model_snapshot_table = pa.Table.from_pylist(
            [
                {
                    "model_version": model_version,
                    "trained_at": trained_at,
                    "feature_snapshot_id": feature_snapshot_id,
                    "raw_events_snapshot_id": raw_events_snapshot_id,
                    "feature_name": feature_name,
                    "decision_threshold": float(decision_threshold),
                    "training_window_start": training_window_start,
                    "training_window_end": training_window_end,
                    "quantile": float(quantile) if quantile is not None else None,
                }
            ],
            schema=self.schema.as_arrow(),
        )
        self.model_table.append(model_snapshot_table)

    def _get_model_snapshot(self, model_version: str):
        """Look up snapshot from model metadata table for a given model version."""
        # Look up features table snapshot from metadata
        model_metadata_filtered = self.model_table.scan(
            row_filter=f"model_version = '{model_version}'",
            selected_fields=("feature_snapshot_id",),
        ).to_arrow()
        if model_metadata_filtered.num_rows == 0:
            raise ValueError(f"Model version {model_version} not found")
        if model_metadata_filtered.num_rows > 1:
            # Take the latest? Raise error? What's your strategy?
            raise ValueError(
                f"Expected one model, found {model_metadata_filtered.num_rows}"
            )
        feature_snapshot_id = model_metadata_filtered.column("feature_snapshot_id")[
            0
        ].as_py()
        snapshot = self.feature_table.snapshot_by_id(feature_snapshot_id)
        return snapshot

    def get_training_data(
        self, model_version: str, start_date: datetime, end_date: datetime
    ) -> pa.Table:
        """Reproduce exact training dataset for a model."""
        snapshot = self._get_model_snapshot(model_version)
        # Query features using that snapshot
        start_date_fmt = start_date.strftime("%Y-%m-%d")
        end_date_fmt = end_date.strftime("%Y-%m-%d")
        scan = self.feature_table.scan(
            f"dt >= '{start_date_fmt}' and dt <= '{end_date_fmt}'",
            snapshot_id=snapshot.snapshot_id,
        )
        return scan.to_arrow()

    def get_current_snapshot_ids(self):
        """Get latest snapshot IDs for both tables."""
        return {
            "features": self.feature_table.current_snapshot().snapshot_id,
            "raw_events": self.raw_events_table.current_snapshot().snapshot_id,
        }

    def get_model_metadata(self, model_version: str) -> pa.Table:
        """Get metadata for a specific model version."""
        scan = self.model_table.scan(
            row_filter=f"model_version = '{model_version}'"
        ).to_arrow()
        if scan.num_rows == 0:
            raise ValueError(f"Model {model_version} not found")
        return scan.to_pylist()[0]

    def get_features_for_inference(
        self, model_version: str, user_id: int, as_of: datetime
    ) -> pa.Table:
        """Get features for real-time scoring using the same snapshot as training."""
        # Use model's feature_snapshot_id
        snapshot = self._get_model_snapshot(model_version)
        # But filter for specific user and recent time window
        as_of_fmt = as_of.strftime("%Y-%m-%d")
        scan = self.feature_table.scan(
            f"user_id = '{user_id}' and dt <= '{as_of_fmt}'",
            snapshot_id=snapshot.snapshot_id,
        )
        return scan.to_arrow()


if __name__ == "__main__":
    fs = FeatureStore("default")
    current_snapshot = fs.get_current_snapshot_ids()
    model_version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    fs.register_model_training(
        model_version,
        datetime.now().astimezone(),
        current_snapshot["features"],
        current_snapshot["raw_events"],
        feature_name="spending_mean_7d",
        decision_threshold=0.6,
        training_window_start=datetime(2025, 8, 29).date(),
        training_window_end=datetime(2025, 9, 3).date(),
        quantile=0.995,
    )
    table = fs.get_training_data(model_version, datetime(2025, 8, 29), datetime(2025, 9, 3))
    print(f"Model version: {model_version}")
    print(f"Got training data from snapshot: {table.shape}")
    print(table.column_names)
    print("Testing getting features for inference")
    table = fs.get_features_for_inference("v1.2.3", "42", datetime.now().astimezone())
    print(f"Got features for inference: {table.shape}")
    print(table.column_names)

    a = fs.get_model_metadata(model_version)
    print(a)
