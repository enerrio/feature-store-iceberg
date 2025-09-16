from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from rich import print

from .feature_store import FeatureStore


@dataclass(frozen=True)
class DetectionResult:
    is_anomaly: bool
    z_score: float
    historical_mean: float
    historical_std: float
    threshold_used: float


class ThresholdAnomalyDetector:
    """Minimal detector that leans on the FeatureStore for provenance."""

    def __init__(self, feature_store: FeatureStore):
        self.fs = feature_store

    def train(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        model_version: Optional[str] = None,
        feature_name: str = "spending_mean_7d",
        quantile: float = 0.995,
    ) -> str:
        """Calibrate a z-score threshold and register the run in the feature store."""

        window_start = start_date.strftime("%Y-%m-%d")
        window_end = end_date.strftime("%Y-%m-%d")
        features = self.fs.feature_table.scan(
            f"dt >= '{window_start}' AND dt <= '{window_end}'"
        ).to_arrow()

        if features.num_rows == 0:
            raise ValueError("No training features found in the selected window.")

        values = self._to_float_np(features[feature_name])
        if np.all(np.isnan(values)):
            raise ValueError("Training data contained only null values.")

        valid_values = values[~np.isnan(values)]
        mean = float(valid_values.mean())
        std = float(valid_values.std(ddof=0))
        if std == 0.0:
            threshold = 3.0
        else:
            z_scores = np.abs((valid_values - mean) / std)
            threshold = float(np.quantile(z_scores, quantile))

        if model_version is None:
            model_version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

        snapshots = self.fs.get_current_snapshot_ids()
        self.fs.register_model_training(
            model_version,
            datetime.now().astimezone(),
            snapshots["features"],
            snapshots["raw_events"],
            feature_name=feature_name,
            decision_threshold=threshold,
            training_window_start=start_date.date(),
            training_window_end=end_date.date(),
            quantile=quantile,
        )
        return model_version

    def detect(
        self,
        user_id: int,
        amount: float,
        as_of: datetime,
        *,
        model_version: str,
        override_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Score a transaction using the threshold stored for `model_version`."""

        metadata = self.fs.get_model_metadata(model_version)
        feature_name = metadata["feature_name"]
        threshold = (
            float(override_threshold)
            if override_threshold is not None
            else float(metadata["decision_threshold"])
        )

        user_features = self.fs.get_features_for_inference(
            model_version, user_id, as_of
        )
        if user_features.num_rows == 0:
            raise ValueError("No feature rows available for this user up to as_of.")

        values = self._to_float_np(user_features[feature_name])
        if np.all(np.isnan(values)):
            raise ValueError("Feature history for user is empty.")
        mean, std = self._safe_stats(values)

        denom = std if std > 0 else 1e-6
        z_score = float((amount - mean) / denom)
        is_anomaly = abs(z_score) > threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            z_score=z_score,
            historical_mean=mean,
            historical_std=std,
            threshold_used=threshold,
        )

    def debug_detection(
        self,
        user_id: int,
        flagged_date: datetime,
        actual_amount: float,
        *,
        model_version: str,
        lookback_days: int = 14,
    ) -> dict:
        """Return decision details plus the exact feature slice used for scoring."""

        metadata = self.fs.get_model_metadata(model_version)
        decision = self.detect(
            user_id=user_id,
            amount=actual_amount,
            as_of=flagged_date,
            model_version=model_version,
        )

        feature_name = metadata["feature_name"]
        user_features = self.fs.get_features_for_inference(
            model_version, user_id, flagged_date
        )
        values = self._to_float_np(user_features[feature_name])
        dates = user_features["dt"].to_pylist()

        last_vals = values[-lookback_days:].tolist()
        last_dates = [str(d) for d in dates[-lookback_days:]]
        as_of_feature = (
            float(values[~np.isnan(values)][-1]) if (~np.isnan(values)).any() else None
        )

        why = (
            f"|z-score| {abs(decision.z_score):.2f} exceeds threshold {decision.threshold_used:.2f}"
            if decision.is_anomaly
            else "within normal range"
        )

        return {
            "model_version": model_version,
            "decision": decision.__dict__,
            "why": why,
            "context": {
                "as_of": str(flagged_date.date()),
                "user_id": user_id,
                "as_of_feature_value": as_of_feature,
                "last_n_feature_values": last_vals,
                "last_n_feature_dates": last_dates,
            },
            "model_card": {
                "trained_at": str(metadata["trained_at"]),
                "feature_snapshot_id": metadata["feature_snapshot_id"],
                "raw_events_snapshot_id": metadata["raw_events_snapshot_id"],
                "feature_name": metadata["feature_name"],
                "decision_threshold": metadata["decision_threshold"],
                "training_window_start": str(metadata["training_window_start"]),
                "training_window_end": str(metadata["training_window_end"]),
                "quantile": metadata["quantile"],
            },
        }

    @staticmethod
    def _to_float_np(array) -> np.ndarray:
        """Convert a PyArrow array to a dense NumPy array of floats (NaNs preserved)."""

        return np.array(
            [float(x) if x is not None else np.nan for x in array.to_pylist()],
            dtype=float,
        )

    @staticmethod
    def _safe_stats(values: np.ndarray) -> tuple[float, float]:
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            return 0.0, 0.0
        return float(valid.mean()), float(valid.std(ddof=0))


if __name__ == "__main__":
    feature_store = FeatureStore("default")
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 9, 13)
    user_id = 42
    amount = 3.59
    as_of = datetime(2025, 9, 14)

    detector = ThresholdAnomalyDetector(feature_store)

    model_version = detector.train(start_date, end_date)

    result = detector.detect(user_id, amount, as_of, model_version=model_version)
    print(result)

    report = detector.debug_detection(
        user_id=user_id,
        flagged_date=as_of,
        actual_amount=amount,
        model_version=model_version,
    )
    print(report)
