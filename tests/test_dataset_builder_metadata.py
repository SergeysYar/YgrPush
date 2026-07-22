import sqlite3

from app.data.dataset_builder import DatasetBuilder


def test_build_batch_features_dataset_includes_batch_and_product_metadata(tmp_path):
    db_path = tmp_path / "metadata_test.db"
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
        "INSERT INTO measurements VALUES (1, '2026-07-20T10:00:00', '1', 101, 'water', 501, 10.0, 5.5, 5.6, 10.0)"
    )
    cur.execute(
        "INSERT INTO Testing_Protocols VALUES (1001, 10, '2026-07-21', '5,9', '1,2', '3200', 1, 1, 98.0)"
    )
    cur.execute(
        "INSERT INTO testing_protocol_values VALUES (1001, 5.9, 3200.0, 1.2)"
    )
    conn.commit()
    conn.close()

    dataset = DatasetBuilder(db_path).build_batch_features_dataset()

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["batch_number"] == "B-001"
    assert row["product_category"] == "daily"
    assert row["product_base"] == "water"
    assert row["product_base_code"] == "W1"
    assert row["target_ph"] == 5.9
