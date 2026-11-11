"""Entry point for the standalone (PyInstaller) builds."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _log_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "GenParallax3D"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "GenParallax3D"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "GenParallax3D"
    base.mkdir(parents=True, exist_ok=True)
    return base / "GenParallax3D.log"


def _redirect_missing_stdio() -> None:
    """A windowed executable has no console: send output to a log file instead of ``None``."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        stream = open(_log_path(), "w", encoding="utf-8", buffering=1)  # noqa: SIM115 - lives for the process
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


if __name__ == "__main__":
    # Must come first: spawned worker processes re-run this executable.
    multiprocessing.freeze_support()
    if "--self-test" not in sys.argv:
        _redirect_missing_stdio()
    from genparallax.app.main_window import main

    sys.exit(main())
