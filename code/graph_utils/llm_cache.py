"""
LLM response caches for an incremental graph build.

A new build of a changed corpus repeats every LLM prompt of the unchanged chunks. The earlier build's cache answers
them, but it may belong to another user and be read-only, and copying it (tens of gigabytes) is wasteful.
`LayeredCache` reads the build's own cache first and then the earlier one; it writes only to its own cache.
`ReadOnlyDiskCache` reads a `diskcache.Index` directory without opening it for writing: diskcache itself creates
tables and settings on open, which fails on a read-only directory.
"""
import os
import sqlite3
import threading
from collections.abc import Iterator, MutableMapping
from typing import Any

from diskcache.core import Disk

_MISSING = object()


class ReadOnlyDiskCache:
    """Lookups in the `Cache` table of a diskcache directory through a read-only SQLite connection."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self._disk = Disk(directory)
        # A closed WAL database (no cache.db-wal) opened with plain mode=ro in a read-only directory fails: SQLite
        # tries to create cache.db-shm there. The earlier build is finished, so it is opened as immutable: no locks,
        # no -shm. If a -wal file is still there, part of the data is only in it, which immutable mode does not read.
        has_wal = os.path.exists(os.path.join(directory, 'cache.db-wal'))
        mode = 'mode=ro' if has_wal else 'mode=ro&immutable=1'
        self._connection = sqlite3.connect(f'file:{directory}/cache.db?{mode}', uri=True, check_same_thread=False)
        self._lock = threading.Lock()

    def _lookup(self, key: Any) -> Any:
        db_key, raw = self._disk.put(key)
        with self._lock:
            row = self._connection.execute(
                'SELECT mode, filename, value FROM Cache WHERE key = ? AND raw = ?', (db_key, raw)
            ).fetchone()
        if row is None:
            return _MISSING
        mode, filename, value = row
        return self._disk.fetch(mode, filename, value, False)

    def get(self, key: Any, default: Any = None) -> Any:
        value = self._lookup(key)
        return default if value is _MISSING else value

    def __contains__(self, key: Any) -> bool:
        return self._lookup(key) is not _MISSING


class LayeredCache(MutableMapping):
    """The build's own cache (read and written) over an earlier, read-only one (read only)."""

    def __init__(self, primary: MutableMapping, fallback: ReadOnlyDiskCache) -> None:
        self.primary = primary
        self.fallback = fallback

    def __getitem__(self, key: Any) -> Any:
        value = self.primary.get(key, _MISSING)
        if value is _MISSING:
            value = self.fallback.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: Any, value: Any) -> None:
        self.primary[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.primary[key]

    def __iter__(self) -> Iterator:
        return iter(self.primary)

    def __len__(self) -> int:
        return len(self.primary)
