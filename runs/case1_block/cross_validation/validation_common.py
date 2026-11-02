"""Shared, fixed physical contract and auditable frequency-shard I/O."""
from hashlib import sha256 as _sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import zlib

import numpy as np


FREQUENCIES = np.array([.1, 1., 10., 100.], dtype=np.float64)


def receivers(extra_depth=False):
    x, y = np.meshgrid(np.arange(1000., 5001., 50.), np.arange(-1000., 1001., 50.))
    result = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.)))
    if extra_depth:
        result = np.vstack((result, np.column_stack((np.arange(1000., 5001., 50.),
                                                    np.full(81, 400.), np.full(81, -.1)))))
    return result


def line_indices(rx=None):
    coords = receivers() if rx is None else np.asarray(rx)
    indices = np.flatnonzero((coords[:, 0] == 3000.) & (coords[:, 2] == -1.))
    return indices[np.argsort(coords[indices, 1])]


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


def _validate(index, coords, electric, magnetic, meta):
    if isinstance(index, bool) or int(index) != index or not 0 <= index < len(FREQUENCIES):
        raise ValueError('frequency index must be 0..3')
    if coords.shape not in ((3321, 3), (3402, 3)) or not np.array_equal(coords, receivers(len(coords) == 3402)):
        raise ValueError('coordinates do not match the fixed receiver contract')
    if electric.shape != coords.shape or magnetic.shape != coords.shape:
        raise ValueError('E and H must contain all three components at every receiver')
    if not np.iscomplexobj(electric) or not np.iscomplexobj(magnetic):
        raise ValueError('E and H must be complex arrays')
    if not np.all(np.isfinite(electric)) or not np.all(np.isfinite(magnetic)):
        raise ValueError('nonfinite response')
    expected = identity()
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('incompatible ' + key)
    if meta.get('physical_hash') != expected['physical_hash']:
        raise ValueError('physical_hash is required')
    mesh_hash = meta.get('mesh_hash', '')
    if not isinstance(mesh_hash, str) or len(mesh_hash) != 64 or any(c not in '0123456789abcdef' for c in mesh_hash):
        raise ValueError('mesh_hash must be a SHA-256 digest')
    if meta.get('status') not in ('complete', 'failed'):
        raise ValueError('status must be complete or failed')
    residual = meta.get('residual')
    if meta['status'] == 'complete' and (not isinstance(residual, (float, int)) or not np.isfinite(residual) or not 0 <= residual <= 1e-6):
        raise ValueError('complete shard requires measured relative residual <= 1e-6')


def _read_shard(path):
    with Path(path).open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in ('coords', 'E', 'H')}
        result['index'] = int(data['index'])
        result['frequency'] = float(data['frequency'])
        result['meta'] = json.loads(str(data['meta']))
    _validate(result['index'], result['coords'], result['E'], result['H'], result['meta'])
    if result['frequency'] != FREQUENCIES[result['index']]:
        raise ValueError('frequency/index mismatch')
    return result


def _atomic_npz(path, **arrays):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_frequency(output, index, coords, E, H, meta):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coords, E, H = np.asarray(coords), np.asarray(E), np.asarray(H)
    meta = dict(_json_safe(meta))
    _validate(index, coords, E, H, meta)
    path = output / f'frequency_{index:02d}.npz'
    if path.exists():
        old = _read_shard(path)
        if old['meta']['mesh_hash'] != meta['mesh_hash'] or old['meta']['physical_hash'] != meta['physical_hash']:
            raise ValueError('resume rejected: mesh/physical identity changed')
        if old['meta'] == meta and np.array_equal(old['coords'], coords) and np.array_equal(old['E'], E) and np.array_equal(old['H'], H):
            return {'status': meta['status'], 'path': str(path), 'resumed': True}
        if old['meta']['status'] == 'complete':
            raise ValueError('resume rejected: completed shard differs')
        attempts = output / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt = attempts / f'frequency_{index:02d}_failed_{sha256(path)}.npz'
        if not attempt.exists():
            shutil.copyfile(path, attempt)
    _atomic_npz(path, index=int(index), frequency=FREQUENCIES[index], coords=coords, E=E, H=H,
                meta=json.dumps(meta, sort_keys=True, allow_nan=False))
    return {'status': meta['status'], 'path': str(path), 'resumed': False}


def merge_responses(output, required_indices=None):
    output = Path(output)
    required = list(range(len(FREQUENCIES))) if required_indices is None else list(required_indices)
    if any(isinstance(i, bool) or int(i) != i or not 0 <= i < len(FREQUENCIES) for i in required) or len(set(required)) != len(required):
        raise ValueError('required_indices must contain unique frequency indices in 0..3')
    valid, missing, failed, errors = {}, [], [], {}
    for index in range(len(FREQUENCIES)):
        path = output / f'frequency_{index:02d}.npz'
        if not path.exists():
            missing.append(index)
            continue
        try:
            shard = _read_shard(path)
            if shard['index'] != index:
                raise ValueError('shard filename/index mismatch')
            if shard['meta']['status'] != 'complete':
                failed.append(index)
            else:
                valid[index] = shard
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            errors[index] = str(error)
    if valid:
        first = next(iter(valid.values()))
        for index, shard in list(valid.items()):
            if shard['meta']['mesh_hash'] != first['meta']['mesh_hash'] or not np.array_equal(shard['coords'], first['coords']):
                errors[index] = 'mixed mesh identity or receiver sampling'
                del valid[index]
    complete = len(valid) == len(FREQUENCIES)
    status = {'schema_version': 1, 'status': 'complete' if complete else 'partial', 'valid_indices': sorted(valid),
              'missing_indices': missing, 'failed_indices': failed, 'errors': errors,
              'required_indices': required, 'required_complete': all(i in valid for i in required), **identity()}
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'responses.npz'
    if complete:
        shards = [valid[i] for i in range(len(FREQUENCIES))]
        arrays = {'schema_version': 1, 'frequencies_hz': FREQUENCIES, 'receiver_xyz': receivers(),
                  'frequencies': FREQUENCIES, 'coords': receivers(),
                  'rx_xyz': receivers(), 'indices': np.arange(len(FREQUENCIES)),
                  'E': np.stack([s['E'][:3321] for s in shards]),
                  'H': np.stack([s['H'][:3321] for s in shards]),
                  'meta': json.dumps([s['meta'] for s in shards], sort_keys=True),
                  **{key: value for key, value in identity().items()}}
        if len(shards[0]['coords']) == 3402:
            arrays.update(coords_extra=receivers(True)[3321:],
                          E_extra=np.stack([s['E'][3321:] for s in shards]),
                          H_extra=np.stack([s['H'][3321:] for s in shards]))
        _atomic_npz(target, **arrays)
        status['responses_sha256'] = sha256(target)
    elif target.exists():
        attempts = output / 'attempts'
        attempts.mkdir(exist_ok=True)
        target.replace(attempts / ('responses_obsolete_' + sha256(target) + '.npz'))
    write_json(output / 'merge_status.json', status)
    return status
