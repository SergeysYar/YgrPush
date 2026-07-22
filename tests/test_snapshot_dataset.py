import sqlite3

from app.data.dataset_builder import DatasetBuilder


def _create_snapshot_db(db_path):
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
        CREATE TABLE testing_protocol_values (
            protocol_id INTEGER PRIMARY KEY,
            ph_value REAL,
            viscosity_value REAL,
            chlorides_value REAL
        )
        """
    )
    cur.execute(
        "INSERT INTO measurements VALUES (1, '2026-07-20T10:00:00', '1', 101, 'water', 501, 10.0, 5.5, 5.6, 10.0)"
    )
    cur.execute(
        "INSERT INTO measurements VALUES (2, '2026-07-20T10:15:00', '1', 102, 'salt', 601, 3.0, 5.7, 5.9, 20.0)"
    )
    cur.execute(
        "INSERT INTO Testing_Protocols VALUES (1001, 10, '2026-07-21', '5,9', '1,2', '3200', 1, 1, 98.0)"
    )
    cur.execute(
        "INSERT INTO testing_protocol_values VALUES (1001, 5.9, 3200.0, 1.2)"
    )
    cur.execute(
        "INSERT INTO Batches VALUES (1, 10, 'B-001', '2026-07-20', 'completed')"
    )
    cur.execute(
        "INSERT INTO Products VALUES (10, 'Test Shampoo', 'daily', 'water', 'W1', 'x', 'y', 'z', 'adj')"
    )
    conn.commit()
    conn.close()


def test_build_snapshot_features_dataset(tmp_path):
    db_path = tmp_path / "snapshot_test.db"
    _create_snapshot_db(db_path)

    dataset = DatasetBuilder(db_path).build_snapshot_features_dataset()

    assert len(dataset) == 2
    assert dataset["batch_id"].tolist() == [1, 1]
    assert dataset["checkpoint_order"].tolist() == [1, 2]
    assert dataset["completed_steps"].tolist() == [1, 2]
    assert dataset["is_final_snapshot"].tolist() == [False, True]
    assert dataset["snapshot_weight"].tolist() == [0.5, 0.5]
    assert dataset["target_ph"].tolist() == [5.9, 5.9]
    assert dataset["product_category"].tolist() == ["daily", "daily"]
    assert dataset["product_base"].tolist() == ["water", "water"]
    assert dataset["batch_number"].tolist() == ["B-001", "B-001"]
