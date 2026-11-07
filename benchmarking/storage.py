from __future__ import annotations

import fcntl
import gzip
import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarking.common import canonical, digest


class Store:
    def __init__(self, directory: Path, *, readonly: bool = False):
        self.directory = directory
        self.readonly = readonly
        self.lock = None
        if readonly:
            self.db = sqlite3.connect(f"file:{directory / 'checkpoints.sqlite3'}?mode=ro", uri=True)
            self.db.execute("BEGIN")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.lock = (directory / ".lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.lock.close()
                raise ValueError(f"Каталог уже используется другим сборщиком: {directory}") from exc
            self.db = sqlite3.connect(directory / "checkpoints.sqlite3")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    key TEXT PRIMARY KEY, identity TEXT NOT NULL, status TEXT NOT NULL,
                    payload BLOB, error TEXT, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, job_key TEXT NOT NULL, started REAL NOT NULL,
                    finished REAL, error TEXT, details BLOB
                );
            """)

    def bind(self, config: dict) -> None:
        value = canonical(config)
        existing = self.metadata("config")
        if existing is not None and existing != config:
            raise ValueError(
                "Настройки, исходные данные или backend изменились. Чекпоинты сохранены. "
                "Для другого эксперимента укажите новый --output-dir."
            )
        if existing is None:
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES ('config', ?)", (value,))

    def metadata(self, name: str) -> Any:
        row = self.db.execute("SELECT value FROM metadata WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def is_done(self, key: str) -> bool:
        row = self.db.execute("SELECT status FROM jobs WHERE key=?", (key,)).fetchone()
        return bool(row and row[0] == "complete")

    def start(self, key: str, identity: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO jobs VALUES (?, ?, 'running', NULL, NULL, ?) "
                "ON CONFLICT(key) DO UPDATE SET status='running', error=NULL, updated=excluded.updated",
                (key, canonical(identity), time.time()),
            )
            self.db.execute("INSERT INTO attempts(job_key, started) VALUES (?, ?)", (key, time.time()))

    def complete(self, key: str, payload: dict) -> None:
        packed = gzip.compress(canonical(payload).encode(), mtime=0)
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='complete',payload=?,error=NULL,updated=? WHERE key=?",
                (packed, time.time(), key),
            )
            self.db.execute(
                "UPDATE attempts SET finished=? WHERE id=(SELECT MAX(id) FROM attempts WHERE job_key=?)",
                (time.time(), key),
            )

    def fail(self, key: str, error: str, *, details: Any = None) -> None:
        packed = gzip.compress(canonical(details).encode(), mtime=0) if details is not None else None
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',error=?,payload=?,updated=? WHERE key=?", (error, packed, time.time(), key))
            self.db.execute(
                "UPDATE attempts SET finished=?,error=?,details=? WHERE id=(SELECT MAX(id) FROM attempts WHERE job_key=?)",
                (time.time(), error, packed, key),
            )

    def records(self, *, failures: bool = False):
        for key, identity, status, payload, error in self.db.execute(
            "SELECT key,identity,status,payload,error FROM jobs ORDER BY key"
        ):
            if status == "complete":
                yield json.loads(gzip.decompress(payload))
            elif failures:
                yield {
                    "job_key": key, **json.loads(identity), "status": status, "error": error, "score": None,
                    "details": json.loads(gzip.decompress(payload)) if payload else None,
                }

    def content_digest(self, progress: Callable[[int], None] | None = None) -> str:
        import hashlib

        hasher = hashlib.sha256()
        for count, (key, payload) in enumerate(
            self.db.execute("SELECT key,payload FROM jobs WHERE status='complete' ORDER BY key"), start=1,
        ):
            hasher.update(canonical([key, digest(json.loads(gzip.decompress(payload)))]).encode())
            if progress is not None and count % 100 == 0:
                progress(count)
        return hasher.hexdigest()

    def counts(self) -> dict[str, int]:
        return dict(self.db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))

    def close(self) -> None:
        self.db.close()
        if self.lock:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
