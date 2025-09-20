from datetime import date, datetime
from typing import Any, Optional

import pyarrow as pa
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.snapshots import Snapshot
from pyiceberg.transforms import DayTransform
from pyiceberg.types import (
    DateType,
    FloatType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)

from .catalog import get_catalog


class FeatureStore:
    """Convenience wrapper around the Iceberg catalog used in the demos."""

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
        feature_name: str,
        decision_threshold: float,
        training_window_start: date,
        training_window_end: date,
        quantile: Optional[float] = None,
    ) -> None:
        """Persist a model training event and its calibrated threshold.

        Args:
            model_version: Identifier assigned to the trained model.
            trained_at: Timestamp when training finished.
            feature_snapshot_id: Snapshot ID from the `user_features` table.
            raw_events_snapshot_id: Snapshot ID from the `raw_events` table.
            feature_name: Feature used to calibrate the decision threshold.
            decision_threshold: Absolute z-score threshold picked for the model.
            training_window_start: First day included in the training window.
            training_window_end: Last day included in the training window.
            quantile: Optional quantile used when computing the threshold.
        """
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

    def _get_model_snapshot(self, model_version: str) -> Snapshot:
        """Return the feature snapshot associated with `model_version`.

        Args:
            model_version: Registered model identifier.

        Returns:
            Snapshot: Iceberg snapshot representing the feature table state.

        Raises:
            ValueError: If the model version is missing or duplicated.
        """
        model_metadata_filtered = self.model_table.scan(
            row_filter=f"model_version = '{model_version}'",
            selected_fields=("feature_snapshot_id",),
        ).to_arrow()
        if model_metadata_filtered.num_rows == 0:
            raise ValueError(f"Model version {model_version} not found")
        if model_metadata_filtered.num_rows > 1:
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
        """Return the training data window used for `model_version`.

        Args:
            model_version: Registered model identifier.
            start_date: Inclusive lower bound of the training window.
            end_date: Inclusive upper bound of the training window.

        Returns:
            pa.Table: Arrow table containing the feature rows for the window.
        """
        snapshot = self._get_model_snapshot(model_version)
        # Query features using that snapshot
        start_date_fmt = start_date.strftime("%Y-%m-%d")
        end_date_fmt = end_date.strftime("%Y-%m-%d")
        scan = self.feature_table.scan(
            f"dt >= '{start_date_fmt}' and dt <= '{end_date_fmt}'",
            snapshot_id=snapshot.snapshot_id,
        )
        return scan.to_arrow()

    def get_current_snapshot_ids(self) -> dict[str, int]:
        """Return the latest snapshot IDs for the feature and raw tables.

        Returns:
            dict[str, int]: Mapping with `features` and `raw_events` keys.
        """
        return {
            "features": self.feature_table.current_snapshot().snapshot_id,
            "raw_events": self.raw_events_table.current_snapshot().snapshot_id,
        }

    def get_model_metadata(self, model_version: str) -> dict[str, Any]:
        """Fetch the metadata row registered for `model_version`.

        Args:
            model_version: Registered model identifier.

        Returns:
            dict: Row from the metadata table converted to a Python dict.

        Raises:
            ValueError: If the model version does not exist.
        """
        scan = self.model_table.scan(
            row_filter=f"model_version = '{model_version}'"
        ).to_arrow()
        if scan.num_rows == 0:
            raise ValueError(f"Model {model_version} not found")
        return scan.to_pylist()[0]

    def get_features_for_inference(
        self, model_version: str, user_id: int, as_of: datetime
    ) -> pa.Table:
        """Retrieve features for inference using the model's training snapshot.

        Args:
            model_version: Registered model identifier.
            user_id: Identifier of the user to fetch features for.
            as_of: Inclusive upper bound on the feature dates.

        Returns:
            pa.Table: Arrow table containing the filtered feature rows.
        """
        # Use model's feature_snapshot_id
        snapshot = self._get_model_snapshot(model_version)
        # But filter for specific user and recent time window
        as_of_fmt = as_of.strftime("%Y-%m-%d")
        scan = self.feature_table.scan(
            f"user_id = '{user_id}' and dt <= '{as_of_fmt}'",
            snapshot_id=snapshot.snapshot_id,
        )
        return scan.to_arrow()

