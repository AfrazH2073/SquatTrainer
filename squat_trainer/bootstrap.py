"""Project-local dependency bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_dependencies() -> None:
    """Add the project-local dependency directory to sys.path if it exists."""
    deps_dir = Path(__file__).resolve().parent.parent / ".deps"
    deps_str = str(deps_dir)
    if deps_dir.exists() and deps_str not in sys.path:
        sys.path.insert(0, deps_str)

