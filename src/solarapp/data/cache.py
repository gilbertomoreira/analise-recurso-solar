"""Tiny on-disk JSON cache keyed by request parameters."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from solarapp.config import CACHE_DIR, CACHE_TTL_DAYS


class JsonCache:
    """Stores raw API responses as JSON files named by the SHA-256 of the request key."""

    def __init__(self, directory: Path | None = None, ttl_days: float = CACHE_TTL_DAYS) -> None:
        self.directory = Path(directory) if directory else CACHE_DIR
        self.ttl_s = ttl_days * 86400.0

    def _path(self, key: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: dict[str, Any]) -> Any | None:
        """Return the cached payload, or None if absent/expired/corrupted."""
        path = self._path(key)
        if self.ttl_s <= 0 or not path.exists():
            return None
        if time.time() - path.stat().st_mtime >= self.ttl_s:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, key: dict[str, Any], payload: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(payload), encoding="utf-8")

    def clear(self) -> int:
        """Delete all cache files; return how many were removed."""
        n = 0
        if self.directory.exists():
            for p in self.directory.glob("*.json"):
                p.unlink()
                n += 1
        return n
