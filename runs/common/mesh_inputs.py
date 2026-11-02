"""Checkout-relative input locations shared by all numerical cases."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesh_gen.paths import DATA_ROOT, get_mesh  # noqa: E402

INPUT_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"


def get_case_mesh(name, *, data_root=None, require=True):
    """Resolve a case mesh; data_root replaces the complete data directory.

    The older get_mesh() remains available and takes a mesh_data directory.
    Keeping that distinction explicit avoids a doubled mesh_data component.
    """
    base = Path(data_root).expanduser().resolve() if data_root else INPUT_ROOT
    return get_mesh(name, require=require, data_root=base / "mesh_data")


def material_directory(case, *, data_root=None):
    base = Path(data_root).expanduser().resolve() if data_root else INPUT_ROOT
    return base / "model_data" / case


def runtime_mesh(paths, output):
    """Create an output-local view for custEM's writable mesh directory.

    Upstream creates auxiliary mesh directories even when overwrite_mesh is
    false. Only file symlinks live in this view; the input H5 and parameters
    remain in data and are never copied or rewritten by the run entry.
    """
    directory = Path(output).resolve() / "runtime_mesh"
    directory.mkdir(parents=True, exist_ok=True)
    for key, subdirectory in (("mesh", "_h5"), ("parameters", "para")):
        target = directory / subdirectory
        target.mkdir(exist_ok=True)
        sources = [paths[key]] if key == "mesh" else sorted(paths[key].parent.iterdir())
        for source in sources:
            if not source.is_file():
                continue
            destination = target / source.name
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            try:
                destination.symlink_to(source.resolve())
            except FileExistsError:
                if not destination.is_symlink() or destination.resolve() != source.resolve():
                    raise
    return dict(paths, m_dir=directory,
                mesh=directory / "_h5" / paths["mesh"].name,
                parameters=directory / "para" / paths["parameters"].name)


__all__ = ["ROOT", "DATA_ROOT", "INPUT_ROOT", "OUTPUT_ROOT", "get_mesh",
           "get_case_mesh", "material_directory", "runtime_mesh"]
