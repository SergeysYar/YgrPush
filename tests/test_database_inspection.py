import sqlite3
import tempfile

from app.database.source import DatabaseInspector


def test_inspect_database_with_minimal_schema(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE measurements (id INTEGER PRIMARY KEY, batchID TEXT, timestamp TEXT)")
    cur.execute("CREATE TABLE Batches (batch_id INTEGER PRIMARY KEY, product_id INTEGER, batch_number TEXT, production_date TEXT, status TEXT)")
    conn.commit()
    conn.close()

    inspector = DatabaseInspector(db_path)
    assert inspector.has_table("measurements")
    assert inspector.has_table("Batches")
    tables = inspector.inspect_tables()
    assert "measurements" in tables
    assert "Batches" in tables
