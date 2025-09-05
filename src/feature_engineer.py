import argparse

import duckdb
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import DateType, DecimalType, LongType, NestedField

from .catalog import get_catalog


def create_features(namespace: str, sql_path: str, rest_uri: str) -> int:
    """Compute gap-filled 7d mean via DuckDB (read-only) and append to Iceberg via PyIceberg.

    - Reads SELECT-only SQL from `sql_path` (must yield columns: user_id (int), dt (date), spending_mean_7d (decimal(15,2))).
    - Uses DuckDB Iceberg extension to query the REST catalog (read-only) and fetch an Arrow table.
    - Writes results to `<namespace>.user_features` using PyIceberg (create-if-not-exists, partitioned by dt).
    Returns number of rows written.
    """
    conn = duckdb.connect()
    conn.execute("INSTALL iceberg; LOAD iceberg;")
    conn.execute(f"""
    ATTACH 'warehouse' as icecat (
        TYPE ICEBERG,
        ENDPOINT '{rest_uri}',
        AUTHORIZATION_TYPE 'none'
    );
    """)
    conn.execute(f"USE icecat.{namespace};")

    with open(sql_path) as f:
        sql = f.read()

    arrow_table = conn.execute(sql).fetch_arrow_table()

    schema = Schema(
        NestedField(1, "user_id", LongType(), required=True),
        NestedField(2, "dt", DateType(), required=True),
        NestedField(3, "spending_mean_7d", DecimalType(15, 2), required=True),
    )
    arrow_table = arrow_table.cast(schema.as_arrow())

    catalog = get_catalog()
    features_name = f"{namespace}.user_features"
    dt_id = schema.find_field("dt").field_id
    spec = PartitionSpec(
        PartitionField(
            source_id=dt_id, field_id=100, transform=IdentityTransform(), name="dt"
        )
    )
    table = catalog.create_table_if_not_exists(
        features_name, schema=schema, partition_spec=spec
    )

    # Append computed features
    table.append(arrow_table)

    print(f"Created/updated {features_name} with {arrow_table.num_rows} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineer -> Iceberg table")
    parser.add_argument(
        "--namespace", default="default", help="Iceberg namespace (schema) to use"
    )
    parser.add_argument(
        "--sql",
        default="sql/gapfilled_7day_spend.sql",
        help="Path to SELECT-only SQL file",
    )
    parser.add_argument("--rest-uri", default=None, help="REST catalog endpoint")
    parser.add_argument("--warehouse", default=None, help="Warehouse URI")
    args = parser.parse_args()
    create_features(namespace=args.namespace, sql_path=args.sql, rest_uri=args.rest_uri)
