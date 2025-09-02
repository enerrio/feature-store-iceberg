from pyiceberg.catalog import load_catalog


def get_catalog():
    """Single source of truth for catalog configuration."""
    return load_catalog(
        "rest",
        **{
            "type": "rest",
            "uri": "http://localhost:8181/",
            "warehouse": "file:///tmp/warehouse",
        },
    )
