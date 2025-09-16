import os
from pathlib import Path

import pytest
from pyiceberg.catalog import load_catalog
from testcontainers.compose import DockerCompose


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    monkeypatch.setenv("ICEBERG_REST_URI", "http://localhost:8182/")
    monkeypatch.setenv("ICEBERG_WAREHOUSE", "file:///tmp/test-warehouse")


@pytest.fixture(scope="session")
def test_iceberg_container():
    project_root = Path(__file__).parent.parent.parent
    with DockerCompose(
        context=str(project_root),
        compose_file_name="docker-compose.test.yml",
        pull=True,
    ) as compose:
        # Wait for REST catalog to be ready
        compose.wait_for("http://localhost:8182/v1/config")
        yield compose


@pytest.fixture
def test_catalog():
    """Create a test catalog pointing to test container"""

    return load_catalog(
        "rest",
        **{
            "type": "rest",
            "uri": os.getenv("ICEBERG_REST_URI"),
            "warehouse": os.getenv("ICEBERG_WAREHOUSE"),
        },
    )
