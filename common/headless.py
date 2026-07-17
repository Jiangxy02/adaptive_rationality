"""Headless runtime guards shared by command line entrypoints."""

import os
import sys
from pathlib import Path


def apply_headless_guard() -> None:
    """Configure Qt/matplotlib defaults only when no display server exists."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        str(Path(sys.prefix) / "plugins" / "platforms"),
    )
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
