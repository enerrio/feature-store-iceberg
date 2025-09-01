from pyiceberg.catalog import load_catalog


def get_catalog():
    """Single source of truth for catalog configuration."""
    return load_catalog(
        "local",
        **{
            "type": "rest",
            "uri": "http://localhost:8181/",
        },
    )
