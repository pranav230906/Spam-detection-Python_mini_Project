# Placeholder for test_db.py
# tests/test_db.py
import tempfile
from db import Database

def test_db_insert_and_query():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    id1 = db.insert_message("hello", None, "ham", 0.9, "TestModel")
    rows = db.query_messages()
    assert len(rows) >= 1
    assert rows[0][0] == id1
