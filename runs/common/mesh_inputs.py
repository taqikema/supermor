"""Shared mesh inputs for the forthcoming numerical case entry points."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesh_gen.paths import DATA_ROOT, get_mesh  # noqa: E402

__all__ = ["DATA_ROOT", "get_mesh"]
