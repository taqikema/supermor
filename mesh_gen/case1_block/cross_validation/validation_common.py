"""Fixed block mesh geometry, shared metadata helpers and source discovery.

Geometry/array functions are retained from the verified paper generation code.
Only source discovery and packaging have changed; no forward solver is imported.
"""
from hashlib import sha256 as _sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FREQUENCIES = np.array([.1, 1., 10., 100.], dtype=np.float64)


def receivers(extra_depth=False):
    x, y = np.meshgrid(np.arange(1000., 5001., 50.), np.arange(-1000., 1001., 50.))
    result = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.)))
    if extra_depth:
        result = np.vstack((result, np.column_stack((np.arange(1000., 5001., 50.),
                                                    np.full(81, 400.), np.full(81, -.1)))))
    return result


def physical_config():
    return {
        'conductivity_S_per_m': {'air': 1e-8, 'background': .01, 'block': 1.},
        'block_bounds_m': [[1300., 3500.], [-500., 500.], [-700., -300.]],
        'source': {'kind': 'finite_wire', 'start_m': [-2500., 0., 0.],
                   'end_m': [-2000., 0., 0.], 'current_A': 1.},
        'mu_H_per_m': float(4e-7 * np.pi), 'time_convention': '+iwt',
        'domain_bounds_m': [[-100000., 100000.]] * 3,
    }


def mesh_spec(software, level):
    if isinstance(level, bool) or int(level) != level or level < 0:
        raise ValueError('level must be a nonnegative integer')
    level = int(level)
    name = software.lower()
    if name == 'simpeg':
        return {'software': name, 'level': level, 'h': 100. / 2 ** level}
    if name in ('custem', 'dfom', 'mor'):
        return {'software': 'custem', 'level': level, 'local_volume': 330000. / 3 ** level,
                'brick_volume': 44000. / 3 ** level, 'shallow_volume': 44000. / 3 ** level,
                'surface_area': 10000. / 3 ** (2 * level / 3), 'source_segments': 20 * 2 ** level}
    raise ValueError('software must be custem or simpeg')


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, complex):
        return {'real': _json_safe(value.real), 'imag': _json_safe(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(_json_safe(obj), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = _sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(obj):
    return _sha256(json.dumps(_json_safe(obj), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')).hexdigest()


def identity():
    return {'physical_hash': object_hash(physical_config()),
            'receivers_hash': object_hash(receivers()),
            'frequencies_hash': object_hash(FREQUENCIES)}


def nonnegative_level(value):
    level = int(value)
    if level < 0:
        raise ValueError("level must be a nonnegative integer")
    return level


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
