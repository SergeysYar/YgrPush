import sqlite3

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.dependencies import get_settings
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
        CREATE TABLE Batches (
            batch_id INTEGER PRIMARY KEY,
            product_id INTEGER,
            batch_number TEXT,
            production_date TEXT,
            status TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE Products (
            product_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            base TEXT,
            base_code TEXT,
            viscosity_thickener TEXT,
            viscosity_softener TEXT,
            ph_corrector TEXT,
            viscosity_adjustment TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE Loading_Process (
            loading_step_id INTEGER PRIMARY KEY,
            batch_id INTEGER,
            step_order INTEGER,
            stage TEXT,
            status TEXT,
            type_id INTEGER
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
        "INSERT INTO Batches VALUES (1, 10, 'B-001', '2026-07-20', 'completed')"
    )
    cur.execute(
        "INSERT INTO Products VALUES (10, 'Test Shampoo', 'daily', 'water', 'W1', 'x', 'y', 'z', 'adj')"
    )
    cur.execute(
        "INSERT INTO Loading_Process VALUES (101, 1, 1, 'mix', 'done', 1)"
    )
    cur.execute(
        "INSERT INTO Loading_Process VALUES (102, 1, 2, 'measure', 'done', 2)"
    )
    cur.execute(
        "INSERT INTO measurements VALUES (1, '2026-07-20T10:00:00', '1', 101, 'water', 501, 12.5, 5.6, 5.8, 15.0)"
    )
    cur.execute(
        "INSERT INTO measurements VALUES (2, '2026-07-20T10:20:00', '1', 102, 'salt', NULL, NULL, 5.8, 6.0, 20.0)"
    )
    cur.execute(
        "INSERT INTO Testing_Protocols VALUES (1001, 10, '2026-07-21', '5,9', '1,2', '3200', 1, 1, 98.0)"
    )
    cur.execute(
        "INSERT INTO testing_protocol_values VALUES (1001, 5.9, 3200.0, 1.2)"
    )
    conn.commit()
    conn.close()


def test_data_summary_route(tmp_path):
    db_path = tmp_path / "api_test.db"
    _create_test_db(db_path)

    def override_settings():
        return Settings(db_path=db_path, ml_storage_path=tmp_path / "ml_storage.db")

    app.dependency_overrides[get_settings] = override_settings
    client = TestClient(app)

    response = client.get("/api/v1/data/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_measurements"] == 2
    assert payload["total_batches"] == 1
    assert payload["labeled_batches"]["ph"] == 1
    assert payload["protocol_count"] == 1
    app.dependency_overrides.clear()


def test_batch_details_route(tmp_path):
    db_path = tmp_path / "api_test.db"
    _create_test_db(db_path)

    def override_settings():
        return Settings(db_path=db_path, ml_storage_path=tmp_path / "ml_storage.db")

    app.dependency_overrides[get_settings] = override_settings
    client = TestClient(app)

    response = client.get("/api/v1/batches/1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["batch"]["batch_id"] == 1
    assert payload["product"]["product_id"] == 10
    assert len(payload["measurements"]) == 2
    assert len(payload["protocols"]) == 1
    assert payload["targets"][0]["ph"] == 5.9
    app.dependency_overrides.clear()
