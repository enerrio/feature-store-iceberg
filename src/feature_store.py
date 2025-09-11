from datetime import datetime

import pyarrow as pa
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform
from pyiceberg.types import (
    FloatType,
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
    StructType,
    TimestamptzType,
)

from .catalog import get_catalog


class FeatureStore:
    def __init__(self, namespace: str):
        catalog = get_catalog()
        self.schema = Schema(
            NestedField(1, "model_version", StringType(), required=True),
            NestedField(2, "trained_at", TimestamptzType(), required=True),
            NestedField(3, "feature_snapshot_id", LongType(), required=True),
            NestedField(4, "raw_events_snapshot_id", LongType(), required=True),
            NestedField(
                5,
                "feature_columns",
                ListType(
                    element_id=10, element_type=StringType(), element_required=True
                ),
                required=True,
            ),
            NestedField(
                6,
                "training_params",
                StructType(
                    NestedField(11, "seed", IntegerType(), required=True),
                    NestedField(12, "window", FloatType(), required=False),
                    NestedField(13, "threshold", FloatType(), required=False),
                ),
                required=False,
            ),
            NestedField(
                7,
                "model_metrics",
                StructType(
                    NestedField(14, "accuracy", FloatType(), required=True),
                    NestedField(15, "precision", FloatType(), required=True),
                    NestedField(16, "recall", FloatType(), required=True),
                ),
                required=True,
            ),
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
        **metadata,
    ):
        """Record that model X was trained with snapshot Y."""
        # Create/append to model_training_metadata table
        model_snapshot_table = pa.Table.from_pylist(
            [
                {
                    "model_version": model_version,
                    "trained_at": trained_at,
                    "feature_snapshot_id": feature_snapshot_id,
                    "raw_events_snapshot_id": raw_events_snapshot_id,
                    "feature_columns": metadata["feature_columns"],
                    "training_params": metadata["training_params"],
                    "model_metrics": metadata["model_metrics"],
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
        """Get latest snapshot IDs for both tables"""
        return {
            "features": self.feature_table.current_snapshot().snapshot_id,
            "raw_events": self.raw_events_table.current_snapshot().snapshot_id,
        }

    def get_features_for_inference(
        self, model_version: str, user_id: int, as_of: datetime
    ) -> pa.Table:
        """Get features for real-time scoring using the same snapshot as training"""
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
    metadata = {
        "feature_columns": ["spending_mean_7d"],
        "training_params": {"seed": 21, "window": 3.0, "threshold": 0.6},
        "model_metrics": {"accuracy": 0.84, "precision": 0.74, "recall": 0.49},
    }
    current_snapshot = fs.get_current_snapshot_ids()
    fs.register_model_training(
        "v1.2.3",
        datetime.now().astimezone(),
        current_snapshot["features"],
        current_snapshot["raw_events"],
        **metadata,
    )
    table = fs.get_training_data("v1.2.3", datetime(2025, 8, 29), datetime(2025, 9, 3))
    print(f"Got training data from snapshot: {table.shape}")
    print(table.column_names)
    print("Testing getting features for inference")
    table = fs.get_features_for_inference("v1.2.3", "42", datetime.now().astimezone())
    print(f"Got features for inference: {table.shape}")
    print(table.column_names)
