import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform
from pyiceberg.types import DecimalType, LongType, NestedField, TimestamptzType

from .catalog import get_catalog


def _choose_rng(seed: Optional[int]):
    return random.Random(seed) if seed is not None else random


def generate_raw_events(
    n_events: int = 10_000,
    n_users: int = 1_000,
    days: int = 30,
    base_date: Optional[datetime] = None,
    seed: Optional[int] = None,
    anomalies: Optional[list[dict]] = None,
) -> dict[str, pa.Array]:
    """Generate synthetic user events with optional anomaly spikes.

    Args:
        n_events: total number of events to generate.
        n_users: number of distinct users (user_id in [0, n_users-1]).
        days: number of days in the window (0-based day index in [0, days-1]).
        base_date: start of the window; defaults to (today - (days-1)) at 00:00 UTC.
        seed: optional seed for deterministic generation.
        anomalies: optional list of anomaly specs. Each spec is a dict with keys:
            - user_id (int, required): which user to spike.
            - day (int, required): 0-based day index since base_date (e.g., 5 == day 5).
            - multiplier (float, optional, default 5.0): multiply amounts by this factor.
            - proportion (float, optional, default 0.5): fraction of that user's events
                on that day to spike (0.0-1.0). If `n` is provided, it overrides proportion.
            - n (int, optional): exact number of events to spike for that (user, day).

    Returns:
        A dict of column -> pyarrow arrays with columns:
            id (int), user_id (int), amount (str), vendor_id (int), event_timestamp (datetime)
    """
    rng = _choose_rng(seed)

    if base_date is None:
        # Start window so that the *latest* day aligns to today (UTC) by default
        today_utc = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        base_date = today_utc - timedelta(days=days - 1)
    else:
        # Normalize to midnight UTC for clean day boundaries
        base_date = base_date.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # Pre-allocate lists
    ids: list[int] = []
    user_ids: list[int] = []
    amounts: list[float] = []
    vendor_ids: list[int] = []
    timestamps: list[datetime] = []

    # Pick a starting id so IDs look large but remain unique/sequential
    start_id = rng.randrange(10**8, 10**12)

    for i in range(n_events):
        uid = rng.randrange(n_users)
        day_offset = rng.randrange(days)  # 0..days-1 uniformly
        # Random time within the chosen day
        seconds_into_day = rng.randrange(24 * 60 * 60)
        ts = base_date + timedelta(days=day_offset, seconds=seconds_into_day)

        # Baseline spend: log-normal to mimic long tail; clamp small values
        amt = max(0.5, rng.lognormvariate(3.0, 0.75))  # median ~ e^3 ≈ 20.1

        vid = rng.randint(1, 1_000)

        ids.append(start_id + i)
        user_ids.append(uid)
        amounts.append(amt)
        vendor_ids.append(vid)
        timestamps.append(ts)

    # Apply anomaly specs, if provided
    if anomalies:
        # index events by (user_id, day)
        per_key: dict[tuple, list[int]] = {}
        for idx, (uid, ts) in enumerate(zip(user_ids, timestamps, strict=True)):
            day_idx = (ts.date() - base_date.date()).days
            per_key.setdefault((uid, day_idx), []).append(idx)

        for spec in anomalies:
            uid = int(spec["user_id"])  # required
            day = int(spec["day"])  # required
            multiplier = float(spec.get("multiplier", 5.0))
            proportion = spec.get("proportion", 0.5)
            n_override = spec.get("n")

            idxs = per_key.get((uid, day), [])
            if not idxs:
                continue  # no events for that (user, day)

            if n_override is not None:
                k = max(0, min(int(n_override), len(idxs)))
            else:
                p = max(0.0, min(float(proportion), 1.0))
                k = max(0, min(int(round(p * len(idxs))), len(idxs)))

            chosen = rng.sample(idxs, k) if k and len(idxs) > 0 else []
            for j in chosen:
                amounts[j] *= multiplier

    return {
        "id": pa.array(ids, type=pa.int64()),
        "user_id": pa.array(user_ids, type=pa.int64()),
        "amount": pa.array(
            [Decimal(f"{amt:.2f}") for amt in amounts], type=pa.decimal128(15, 2)
        ),
        "vendor_id": pa.array(vendor_ids, type=pa.int64()),
        "event_timestamp": pa.array(timestamps, type=pa.timestamp("ms", tz="UTC")),
    }


def ingest_raw_events(catalog: Catalog, namespace: str, data: dict[str, pa.Array]):
    """Ingest raw events data to Iceberg."""
    raw_events_table_name = f"{namespace}.raw_events"
    catalog.create_namespace_if_not_exists(namespace)
    # catalog.drop_table(raw_events_table_name)
    # Define an explicit Iceberg schema using Iceberg types (including TIMESTAMPTZ)
    iceberg_schema = Schema(
        NestedField(1, "id", LongType(), required=True),
        NestedField(2, "user_id", LongType(), required=True),
        NestedField(3, "amount", DecimalType(15, 2), required=True),
        NestedField(4, "vendor_id", LongType(), required=True),
        NestedField(5, "event_timestamp", TimestamptzType(), required=True),
    )
    raw_events_data = pa.Table.from_pydict(data, schema=iceberg_schema.as_arrow())

    # Build a partition spec off the timestamp column using built-in transforms
    ts_id = iceberg_schema.find_field("event_timestamp").field_id
    partition_spec = PartitionSpec(
        PartitionField(
            source_id=ts_id, field_id=100, transform=DayTransform(), name="dt"
        ),
        # Uncomment next line if you also want hour-level pruning (epoch hours)
        # PartitionField(source_id=ts_id, field_id=200, transform=HourTransform(), name="hour"),
    )
    raw_data_table = catalog.create_table_if_not_exists(
        raw_events_table_name,
        schema=iceberg_schema,
        partition_spec=partition_spec,
    )
    raw_data_table.append(raw_events_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data generator & features for Iceberg"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    gen = subparsers.add_parser("generate", help="Generate and ingest raw events")
    gen.add_argument("--namespace", default="default")
    gen.add_argument("--n-events", type=int, default=1000)
    gen.add_argument("--n-users", type=int, default=100)
    gen.add_argument("--days", type=int, default=14)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument(
        "--batches",
        type=int,
        default=1,
        help="How many batches to ingest (seeds will increment)",
    )
    gen.add_argument(
        "--anomaly",
        action="append",
        default=[],
        help='JSON object per anomaly spec; may be repeated. Example: \'{"user_id":42,"day":5,"multiplier":10.0,"n":1}\'',
    )
    gen.add_argument(
        "--anomalies-json",
        type=str,
        default=None,
        help='JSON array of anomaly specs. Example: \'[{"user_id":42,"day":5,"multiplier":10.0,"n":1}]\'',
    )
    gen.add_argument(
        "--anomaly-file",
        type=str,
        default=None,
        help="Path to JSON or NDJSON file of anomaly specs",
    )

    args = parser.parse_args()

    catalog = get_catalog()

    if args.cmd == "generate":
        for i in range(args.batches):
            seed = (args.seed + i) if args.seed is not None else None
            # Load anomaly specs (from --anomaly multiple, --anomalies-json array, or --anomaly-file)
            anomaly_specs = []
            # Individual JSON objects via --anomaly (can repeat)
            for spec in args.anomaly or []:
                spec_obj = json.loads(spec)
                anomaly_specs.append(spec_obj)
            # A whole JSON array via --anomalies-json
            if args.anomalies_json:
                arr = json.loads(args.anomalies_json)
                if isinstance(arr, list):
                    anomaly_specs.extend(arr)
                else:
                    raise ValueError("--anomalies-json must be a JSON array")
            # From a file: either a JSON array or NDJSON (one object per line)
            if args.anomaly_file:
                with open(args.anomaly_file) as fh:
                    content = fh.read().strip()
                if content:
                    if content.lstrip().startswith("["):
                        arr = json.loads(content)
                        if not isinstance(arr, list):
                            raise ValueError(
                                "--anomaly-file JSON must be an array or NDJSON"
                            )
                        anomaly_specs.extend(arr)
                    else:
                        for line in content.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            anomaly_specs.append(json.loads(line))
            data = generate_raw_events(
                n_events=args.n_events,
                n_users=args.n_users,
                days=args.days,
                seed=seed,
                anomalies=anomaly_specs,
            )
            ingest_raw_events(catalog, args.namespace, data)
            if anomaly_specs:
                print(
                    f"Ingested {args.n_events} raw events to {args.namespace}.raw_events (with anomalies)"
                )
            else:
                print(
                    f"Ingested {args.n_events} raw events to {args.namespace}.raw_events (no anomalies)"
                )
