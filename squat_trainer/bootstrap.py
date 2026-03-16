"""Project-local dependency bootstrap."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_local_dependencies() -> None:
    """Add the project-local dependency directory to sys.path if it exists."""
    project_root = Path(__file__).resolve().parent.parent
    deps_dir = project_root / ".deps"
    cache_dir = project_root / ".cache"
    mplconfig_dir = project_root / ".mplconfig"
    deps_str = str(deps_dir)
    cache_dir.mkdir(exist_ok=True)
    mplconfig_dir.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig_dir))
    if deps_dir.exists() and deps_str not in sys.path:
        sys.path.insert(0, deps_str)
