"""
Persistent history store for analysis runs.

Saves session history to disk so it survives Streamlit reruns / browser
refreshes. The store is a single JSON file under the user's home dir by
default (override with MARKET_ANALYST_HISTORY_PATH).

Schema (v1):
{
  "version": 1,
  "items": [
    {
      "topic": "Figma",
      "mode": "CrewAI Orchestrator",
      "markdown": "# Market Intelligence Report: ...",
      "scores": {"market_opportunity": 72, ...},
      "timestamp": "2026-06-05 10:30"
    }
  ]
}
"""
import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_HISTORY_VERSION = 1
_DEFAULT_MAX_ITEMS = 20
_DEFAULT_PATH = Path.home() / ".market_analyst" / "history.json"

# Module-level lock so concurrent Streamlit threads don't corrupt the file.
_lock = threading.Lock()


def _resolve_path() -> Path:
    """Resolve the on-disk path, honoring the env override."""
    env = os.environ.get("MARKET_ANALYST_HISTORY_PATH")
    return Path(env) if env else _DEFAULT_PATH


def load_history(max_items: int = _DEFAULT_MAX_ITEMS) -> list[dict[str, Any]]:
    """
    Return the persisted history list, newest-first.

    Returns an empty list if the file is missing, unreadable, or has an
    unexpected schema — never raises, so a corrupt store can't break the
    Streamlit app.
    """
    path = _resolve_path()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, dict) or data.get("version") != _HISTORY_VERSION:
        return []
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    # Newest first, capped
    return list(reversed(items[-max_items:]))


def save_history(items: list[dict[str, Any]]) -> None:
    """
    Atomically write the history list to disk.

    Uses a temp-file + rename so a crash mid-write can't corrupt the
    store. Older entries are pruned to the same cap that load_history
    applies, keeping the file size bounded.
    """
    path = _resolve_path()
    cap = _DEFAULT_MAX_ITEMS
    pruned = list(items)[-cap:]

    payload = {
        "version": _HISTORY_VERSION,
        "items": pruned,
    }

    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same dir, then os.replace
        fd, tmp_path = tempfile.mkstemp(
            prefix=".history.", suffix=".json.tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


def append_history(entry: dict[str, Any], max_items: int = _DEFAULT_MAX_ITEMS) -> list[dict[str, Any]]:
    """
    Append a new entry and return the updated history (newest-first).

    Loads the existing store, appends, saves, and returns the new list.
    The whole read-modify-write cycle is serialized on the module lock
    so concurrent appends from multiple Streamlit threads can't lose
    updates.
    """
    with _lock:
        existing = load_history(max_items=max_items)
        # Reverse to chronological (oldest-first) for the append
        chronological = list(reversed(existing))
        chronological.append(entry)
        # Inline save to avoid re-acquiring _lock (threading.Lock is not reentrant)
        path = _resolve_path()
        payload = {
            "version": _HISTORY_VERSION,
            "items": chronological[-max_items:],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".history.", suffix=".json.tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
    return load_history(max_items=max_items)


def clear_history() -> None:
    """Delete the on-disk history file. Safe to call when the file is absent."""
    path = _resolve_path()
    with _lock:
        if path.exists():
            path.unlink()
