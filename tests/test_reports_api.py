import sqlite3

from fastapi.testclient import TestClient

from app.api.dependencies import get_settings
from app.api.main import app
from app.config import Settings


def _create_test_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE measurements (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            batchID TEXT,
            loading_step_id INTEGER,
            loading_step_type TEXT,
            component_1 INTEGER,
            mass_1 REAL,
            startpH REAL,
            endpH REAL,
            duration_minutes REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE Testing_Protocols (
            protocol_id INTEGER PRIMARY KEY,
            product_id INTEGER,
            test_date TEXT,
            ph TEXT,
            chlorides TEXT,
            viscosity TEXT,
            batch_id INTEGER,
            is_compliant INTEGER,
            compliance_percent REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE testing_protocol_values (
            protocol_id INTEGER PRIMARY KEY,
            ph_value REAL,
            viscosity_value REAL,
            chlorides_value REAL
        )
        """
    )
    cur.execute(
        "INSERT INTO measurements VALUES (1, NULL, '1', 101, 'water', 501, 10.0, 25.0, 5.6, 10.0)"
    )
    cur.execute(
        "INSERT INTO Testing_Protocols VALUES (1001, 10, '2026-07-21', '5,9', '1,2', '3200', 1, 1, 98.0)"
    )
    cur.execute(
        "INSERT INTO testing_protocol_values VALUES (1001, 5.9, 3200.0, 1.2)"
    )
    conn.commit()
    conn.close()


def test_data_quality_report_can_store_and_list_issues(tmp_path):
    db_path = tmp_path / "api_test.db"
    _create_test_db(db_path)

    def override_settings():
        return Settings(db_path=db_path, ml_storage_path=tmp_path / "ml_storage.db")

    app.dependency_overrides[get_settings] = override_settings
    client = TestClient(app)

    report_response = client.get("/api/v1/reports/data-quality?store=true")
    assert report_response.status_code == 200
    report_payload = report_response.json()
    assert "sample_issues" in report_payload
    assert "stored_issues" in report_payload
    assert report_payload["stored_issues"] >= 1

    issues_response = client.get("/api/v1/reports/data-quality/issues")
    assert issues_response.status_code == 200
    issues_payload = issues_response.json()
    assert issues_payload["count"] >= 1
    assert issues_payload["items"][0]["issue_type"] in {"missing", "out_of_range"}
    app.dependency_overrides.clear()
