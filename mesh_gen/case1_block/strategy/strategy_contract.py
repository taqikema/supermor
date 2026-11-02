"""Fixed block strategy geometry and shared mesh metadata helpers."""
from hashlib import sha256 as _sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FREQUENCIES = np.asarray(2. ** np.arange(-6, 14), dtype=np.float64)
FREQUENCIES.setflags(write=False)


def receivers():
    x, y = np.meshgrid(np.arange(1000., 5001., 50.), np.arange(-1000., 1001., 50.))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.)))


def refinement_anchors():
    x = np.arange(0., 4001., 250.)
    return np.column_stack((x, np.full(x.size, 400.), np.zeros(x.size)))


def physical_config():
    return dict(conductivity_S_per_m=dict(air=1e-8, background=.01, block=1.),
        block_bounds_m=[[1300., 3500.], [-500., 500.], [-700., -300.]],
        source=dict(kind='finite_wire', start_m=[-2500., 0., 0.],
                    end_m=[-2000., 0., 0.], current_A=1., segments=20),
        mu_H_per_m=float(4e-7 * np.pi), time_convention='+iwt',
        domain_bounds_m=[[-100000., 100000.]] * 3,
        approximation='quasistatic', boundary_condition='natural')


def sha256(path):
    result = _sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Not JSON serializable: ' + type(value).__name__)


def write_json(path, obj):
    """Replace the destination only after successfully encoding strict JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False, default=_json_value)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_directory(root=None):
    """Locate the installed custEM source; CUSTEM_SOURCE is an optional override.

    ``root`` remains accepted for compatibility with the original mesh entry.
    Source discovery never depends on an experiment output or HPC directory.
    """
    import importlib.util
    configured = os.environ.get("CUSTEM_SOURCE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / "custEM" / "__init__.py").is_file():
            raise FileNotFoundError("CUSTEM_SOURCE must contain custEM/__init__.py")
        return candidate
    specification = importlib.util.find_spec("custEM")
    if specification is None or specification.origin is None:
        raise ImportError("custEM is required; activate the documented cust_env environment")
    candidate = Path(specification.origin).resolve().parent.parent
    if not (candidate / "custEM" / "__init__.py").is_file():
        raise ImportError("The installed custEM package has no readable source directory")
    return candidate
