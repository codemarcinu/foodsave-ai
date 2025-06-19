from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app  # Import main FastAPI instance

client = TestClient(app)


@pytest.fixture(scope="module")
def test_app():
    # Setup test database, mock external services if needed
    yield client
    # Cleanup after tests


def test_full_orchestration_flow(test_app):
    # Simulate API request that uses the orchestrator
    response = test_app.post(
        "/process_query",
        json={"session_id": "test_session_123", "query": "Find a recipe for spaghetti"},
    )
    assert response.status_code == 200
    assert "spaghetti" in response.json().get("response").lower()


def test_add_item_to_list_flow(test_app):
    response = test_app.post(
        "/process_query",
        json={"session_id": "test_session_123", "query": "add milk to shopping list"},
    )
    assert response.status_code == 200
    assert "milk" in response.json().get("response").lower()


def test_invalid_input_query(test_app):
    response = test_app.post(
        "/process_query", json={"session_id": "test_session", "query": ""}
    )  # Empty query
    assert response.status_code == 400
    assert "error_code" in response.json()
    assert response.json()["error_code"] == "INVALID_INPUT"


@patch(
    "backend.main.db_session_dependency.get_db",
    side_effect=Exception("DB connection error"),
)
def test_database_connection_failure(mock_get_db, test_app):
    # Simulate DB connection failure
    response = test_app.post(
        "/process_query", json={"session_id": "test_session", "query": "some query"}
    )
    assert response.status_code == 500
    assert response.json()["error_code"] == "INTERNAL_SERVER_ERROR"
