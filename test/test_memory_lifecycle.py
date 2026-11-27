"""Memory operations close SQLite handles on success and rollback on failure."""
import sqlite3

import pytest

from telepathy_receptor.memory import AivaMemory


def test_transactions_close_and_rollback_without_partial_exchange(tmp_path):
    memory = AivaMemory(str(tmp_path / 'chat.sqlite'))
    with memory._connect() as connection:
        connection.execute("INSERT INTO conversations(id,title) VALUES ('saved','Saved')")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute('SELECT 1')
    with pytest.raises(RuntimeError):
        with memory._connect() as failed:
            failed.execute("INSERT INTO conversations(id,title) VALUES ('rolled-back','Rollback')")
            raise RuntimeError('interrupted')
    with pytest.raises(sqlite3.ProgrammingError):
        failed.execute('SELECT 1')
    with memory._connect() as check:
        assert check.execute("SELECT id FROM conversations WHERE id='rolled-back'").fetchone() is None
        assert check.execute("SELECT id FROM conversations WHERE id='saved'").fetchone() == ('saved',)
