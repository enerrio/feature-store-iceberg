import os

from pyiceberg.catalog import Catalog, load_catalog


def get_catalog() -> Catalog:
    """Single source of truth for catalog configuration."""
    return load_catalog(
        "rest",
        **{
            "type": "rest",
            "uri": os.getenv("ICEBERG_REST_URI", "http://localhost:8181/"),
            "warehouse": os.getenv("ICEBERG_WAREHOUSE", "file:///tmp/warehouse"),
        },
    )
