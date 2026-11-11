"""Checks for the standalone-build support: the self-test and the AI-extra gate."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")


def test_self_test_passes_without_gl(tmp_path, monkeypatch):
    from genparallax.selftest import run

    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    report = tmp_path / "report.txt"
    assert run([str(report), "--no-gl"]) == 0
    text = report.read_text(encoding="utf-8")
    assert "SELF-TEST PASSED" in text and "glb:" in text


def test_ai_available_reflects_installed_modules(monkeypatch):
    import importlib.util

    from genparallax.reconstruction import estimator

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert estimator.ai_available() is False


def test_ai_missing_messages_are_translated(monkeypatch):
    import sys

    from genparallax.app import i18n, main_window

    i18n.set_language("ja")
    try:
        for frozen in (False, True):
            monkeypatch.setattr(sys, "frozen", frozen, raising=False)
            assert "torch" in main_window._ai_missing_text()
            assert "深度マップ" in main_window._ai_missing_text()
    finally:
        i18n.set_language("ja")
