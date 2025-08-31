from pyiceberg.catalog import load_catalog


def get_catalog():
    """Single source of truth for catalog configuration."""
    warehouse_path = "/tmp/warehouse"
    return load_catalog(
        "local",
        **{
            "type": "sql",
            "uri": f"sqlite:///{warehouse_path}/catalog.db",
            "warehouse": f"file://{warehouse_path}",
        },
    )
