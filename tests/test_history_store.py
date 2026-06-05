"""Tests for history_store.py — atomic persistence."""

import json
import threading
from pathlib import Path

import pytest

from history_store import (
    _resolve_path,
    append_history,
    clear_history,
    load_history,
    save_history,
)


@pytest.fixture
def tmp_history_path(tmp_path, monkeypatch):
    """Redirect the store to a tmp file for the duration of each test."""
    p = tmp_path / "history.json"
    monkeypatch.setenv("MARKET_ANALYST_HISTORY_PATH", str(p))
    return p


# ──────────────────────────────────────────────────────────────
#  load / save
# ──────────────────────────────────────────────────────────────
class TestLoadSave:
    def test_load_empty_when_file_missing(self, tmp_history_path):
        assert load_history() == []

    def test_save_creates_file_and_parent(self, tmp_history_path, monkeypatch):
        # The parent might already exist (tmp_path fixture creates it),
        # so use a deeply-nested path to verify nested-dir creation.
        nested = tmp_history_path.parent / "deeply" / "nested" / "history.json"
        monkeypatch.setenv("MARKET_ANALYST_HISTORY_PATH", str(nested))
        assert not nested.parent.exists()
        save_history(
            [{"topic": "Figma", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "now"}]
        )
        assert nested.exists()

    def test_save_then_load_roundtrip(self, tmp_history_path):
        items = [
            {"topic": "A", "mode": "x", "markdown": "m1", "scores": {}, "timestamp": "t1"},
            {"topic": "B", "mode": "y", "markdown": "m2", "scores": {}, "timestamp": "t2"},
        ]
        save_history(items)
        loaded = load_history()
        # load returns newest-first
        assert len(loaded) == 2
        assert loaded[0]["topic"] == "B"
        assert loaded[1]["topic"] == "A"

    def test_load_returns_empty_for_corrupt_file(self, tmp_history_path):
        tmp_history_path.write_text("not valid json {{{", encoding="utf-8")
        assert load_history() == []

    def test_load_returns_empty_for_wrong_schema(self, tmp_history_path):
        tmp_history_path.write_text(json.dumps({"version": 99, "items": []}), encoding="utf-8")
        # version mismatch → ignored
        assert load_history() == []

    def test_load_returns_empty_if_items_not_list(self, tmp_history_path):
        tmp_history_path.write_text(
            json.dumps({"version": 1, "items": "not a list"}), encoding="utf-8"
        )
        assert load_history() == []

    def test_save_caps_items(self, tmp_history_path):
        # 50 items → should be pruned to default cap (20)
        big = [
            {"topic": f"T{i}", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t"}
            for i in range(50)
        ]
        save_history(big)
        loaded = load_history()
        assert len(loaded) == 20  # default cap

    def test_save_writes_valid_json(self, tmp_history_path):
        save_history(
            [
                {
                    "topic": "X",
                    "mode": "x",
                    "markdown": "m",
                    "scores": {"a": 1},
                    "timestamp": "t",
                    "unicode": "résumé",
                }
            ]
        )
        data = json.loads(tmp_history_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["items"][0]["unicode"] == "résumé"

    def test_save_is_atomic_no_leftover_tmp(self, tmp_history_path):
        save_history([{"topic": "X", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t"}])
        # No leftover temp files in the parent dir
        leftovers = list(tmp_history_path.parent.glob(".history.*.tmp"))
        assert leftovers == []


# ──────────────────────────────────────────────────────────────
#  append_history
# ──────────────────────────────────────────────────────────────
class TestAppend:
    def test_append_to_empty(self, tmp_history_path):
        result = append_history(
            {"topic": "Figma", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t"}
        )
        assert len(result) == 1
        assert result[0]["topic"] == "Figma"

    def test_append_preserves_order_newest_first(self, tmp_history_path):
        append_history(
            {"topic": "A", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t1"}
        )
        append_history(
            {"topic": "B", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t2"}
        )
        result = append_history(
            {"topic": "C", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t3"}
        )
        assert [r["topic"] for r in result] == ["C", "B", "A"]

    def test_append_caps_at_max(self, tmp_history_path):
        for i in range(25):
            append_history(
                {"topic": f"T{i}", "mode": "x", "markdown": "m", "scores": {}, "timestamp": f"t{i}"}
            )
        result = load_history(max_items=20)
        assert len(result) == 20
        # Newest first
        assert result[0]["topic"] == "T24"
        assert result[-1]["topic"] == "T5"

    def test_concurrent_appends_dont_corrupt(self, tmp_history_path):
        """100 appends across 10 threads should all land safely."""
        n_threads = 10
        n_per_thread = 10

        def worker(tid):
            for i in range(n_per_thread):
                append_history(
                    {
                        "topic": f"T{tid}_{i}",
                        "mode": "x",
                        "markdown": "m",
                        "scores": {},
                        "timestamp": "t",
                    }
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Default cap is 20 — exactly 20 items should be in the file
        result = load_history()
        assert len(result) == 20
        # Each entry should be valid (no truncated JSON)
        for r in result:
            assert "topic" in r
            assert "timestamp" in r


# ──────────────────────────────────────────────────────────────
#  clear_history
# ──────────────────────────────────────────────────────────────
class TestClear:
    def test_clear_removes_file(self, tmp_history_path):
        save_history([{"topic": "X", "mode": "x", "markdown": "m", "scores": {}, "timestamp": "t"}])
        assert tmp_history_path.exists()
        clear_history()
        assert not tmp_history_path.exists()

    def test_clear_when_no_file_is_safe(self, tmp_history_path):
        # Should not raise
        clear_history()
        clear_history()


# ──────────────────────────────────────────────────────────────
#  Path resolution
# ──────────────────────────────────────────────────────────────
class TestPathResolution:
    def test_default_path_is_in_home(self, monkeypatch):
        monkeypatch.delenv("MARKET_ANALYST_HISTORY_PATH", raising=False)
        p = _resolve_path()
        assert str(p).startswith(str(Path.home()))

    def test_env_override_takes_precedence(self, monkeypatch):
        # Use a path the OS can actually represent (Windows-safe)
        custom = str(Path.home() / "custom_subdir" / "h.json")
        monkeypatch.setenv("MARKET_ANALYST_HISTORY_PATH", custom)
        p = _resolve_path()
        assert str(p) == custom
