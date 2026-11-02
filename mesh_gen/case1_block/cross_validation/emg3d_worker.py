#!/usr/bin/env python3
"""Generate the fixed paper emg3d grid and interpolation inputs with NumPy.

The original mesh/array functions are retained verbatim. Forward solves and
native emg3d imports are intentionally outside this mesh-generation entry.
"""
from pathlib import Path
import argparse
import json
import os
import tempfile

import numpy as np

if __package__:
    from .validation_common import identity, object_hash, physical_config, receivers, sha256, write_json
else:
    from validation_common import identity, object_hash, physical_config, receivers, sha256, write_json


CORE = {
    'x': ([-3000., -2500., -2000., 0., 1000., 1300., 3500., 5000., 5500.], [4, 10, 18, 10, 6, 26, 18, 4]),
    'y': ([-1500., -1000., -500., 0., 500., 1000., 1500.], [16]*6),
    'z': ([-1500., -700., -300., -100., -64., -32., -16., -8., -4., -2., -1.,
           0., 1., 2., 4., 8., 16., 32., 64., 100., 500.], [8, 12, 4]+[1]*16+[8]),
}


def _padding(length):
    low, high = 1., 2.
    for _ in range(100):
        ratio = (low+high)/2
        if np.sum(100.*ratio**np.arange(16)) < length:
            low = ratio
        else:
            high = ratio
    distances = np.cumsum(100.*((low+high)/2)**np.arange(16))
    distances[-1] = length
    return distances


def fixed_mesh_arrays():
    result = {}
    for axis, (breaks, counts) in CORE.items():
        core = np.concatenate([np.linspace(a, b, count+1)[:-1]
                               for a, b, count in zip(breaks[:-1], breaks[1:], counts)] + [[breaks[-1]]])
        left = breaks[0]-_padding(breaks[0]+100000.)[::-1]
        right = breaks[-1]+_padding(100000.-breaks[-1])
        nodes = np.r_[left, core, right]
        # Differencing in the positive cumulative coordinate preserves exact
        # interface nodes when TensorMesh reconstructs nodes from its origin.
        widths = np.diff(nodes+100000.)
        nodes = -100000.+np.r_[0., np.cumsum(widths)]
        result['nodes_'+axis], result['h'+axis] = nodes, widths
    centers = [(result['nodes_'+axis][:-1]+result['nodes_'+axis][1:])/2 for axis in 'xyz']
    x, y, z = (centers[0][:, None, None], centers[1][None, :, None], centers[2][None, None, :])
    config = physical_config()
    conductivity = config['conductivity_S_per_m']
    shape = tuple(len(result['h'+axis]) for axis in 'xyz')
    sigma = np.full(shape, conductivity['background'], dtype=np.float64, order='F')
    sigma[:, :, centers[2] > 0] = conductivity['air']
    bounds = config['block_bounds_m']
    block = ((x > bounds[0][0]) & (x < bounds[0][1]) &
             (y > bounds[1][0]) & (y < bounds[1][1]) &
             (z > bounds[2][0]) & (z < bounds[2][1]))
    sigma[block] = conductivity['block']
    result['sigma'] = sigma
    return result


def _array_hash(array):
    import hashlib
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order='F'))
    return digest.hexdigest()


def mesh_metadata(arrays):
    nx, ny, nz = (len(arrays['h'+axis]) for axis in 'xyz')
    edges = nx*(ny+1)*(nz+1)+(nx+1)*ny*(nz+1)+(nx+1)*(ny+1)*nz
    hashes = {key: _array_hash(value) for key, value in arrays.items()}
    return {'shape_cells': [nx, ny, nz], 'cells': nx*ny*nz,
        'complex_edges': edges, 'real_equivalent_unknowns': 2*edges,
        'pec_free_complex_edges': nx*(ny-1)*(nz-1)+(nx-1)*ny*(nz-1)+(nx-1)*(ny-1)*nz,
        'mesh_hash': object_hash(hashes), 'array_sha256': hashes,
        'array_hash_order': 'Fortran', 'core': CORE, 'padding_cells_per_side': 16,
        'first_padding_width_m': 100., 'bounds_m': [[-100000., 100000.]]*3,
        'nodes': {axis: arrays['nodes_'+axis].tolist() for axis in 'xyz'},
        'widths': {axis: arrays['h'+axis].tolist() for axis in 'xyz'},
        'conductivity_counts': {str(value): int(np.count_nonzero(arrays['sigma'] == value))
                                for value in np.unique(arrays['sigma'])}, **identity()}


def component_axes(arrays, component):
    if component not in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        raise ValueError('unknown field component')
    aligned = 'xyz'.index(component[1])
    result = []
    for dimension, axis in enumerate('xyz'):
        nodes = arrays['nodes_'+axis]
        centered = (dimension == aligned) if component[0] == 'E' else (dimension != aligned)
        result.append((nodes[:-1]+nodes[1:])/2 if centered else nodes)
    return result


def linear_support(arrays, component, coords):
    coords = np.asarray(coords)
    axes = component_axes(arrays, component)
    brackets, fractions = [], []
    for dimension, axis in enumerate(axes):
        if np.any(coords[:, dimension] < axis[0]) or np.any(coords[:, dimension] > axis[-1]):
            raise ValueError('receiver outside interpolation axis')
        lower = np.clip(np.searchsorted(axis, coords[:, dimension], side='right')-1, 0, len(axis)-2)
        brackets.append(lower)
        fractions.append((coords[:, dimension]-axis[lower])/(axis[lower+1]-axis[lower]))
    indices = np.empty((len(coords), 8, 3), dtype=np.int64)
    weights = np.ones((len(coords), 8))
    for corner in range(8):
        for dimension in range(3):
            upper = (corner >> dimension) & 1
            indices[:, corner, dimension] = brackets[dimension]+upper
            weights[:, corner] *= fractions[dimension] if upper else 1-fractions[dimension]
    support_z = axes[2][indices[:, :, 2]]
    if np.any(support_z[weights != 0] >= 0):
        raise ValueError('nonzero receiver interpolation support crosses into z>=0')
    return {'indices': indices, 'weights': weights, 'z': support_z}


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _audit_mesh_arrays(arrays):
    """Audit saved physical geometry without regenerating platform-sensitive padding."""
    expected_shapes = {'sigma': (128, 128, 80)}
    for axis, count in zip('xyz', (128, 128, 80)):
        expected_shapes['nodes_' + axis] = (count + 1,)
        expected_shapes['h' + axis] = (count,)
    if (set(arrays) != set(expected_shapes) or
            any(value.dtype != np.float64 or value.shape != expected_shapes[name] or
                not np.isfinite(value).all() for name, value in arrays.items())):
        raise ValueError('fixed mesh arrays must be finite float64 on the 128x128x80 grid')
    for axis in 'xyz':
        nodes, widths = arrays['nodes_' + axis], arrays['h' + axis]
        if (nodes[0] != -100000. or nodes[-1] != 100000. or np.any(widths <= 0) or
                not np.allclose(np.diff(nodes), widths, rtol=1e-12, atol=1e-10)):
            raise ValueError('fixed mesh actual bounds or cell widths changed')
        breaks, counts = CORE[axis]
        positions = [np.flatnonzero(nodes == value) for value in breaks]
        if (any(len(position) != 1 for position in positions) or
                positions[0][0] != 16 or positions[-1][0] != len(widths) - 16 or
                not np.array_equal(np.diff([position[0] for position in positions]), counts)):
            raise ValueError('fixed mesh physical interfaces or core cell counts changed')
    centers = [(arrays['nodes_' + axis][:-1] + arrays['nodes_' + axis][1:]) / 2 for axis in 'xyz']
    x, y, z = centers[0][:, None, None], centers[1][None, :, None], centers[2][None, None, :]
    config = physical_config()
    bounds, sigma = config['block_bounds_m'], config['conductivity_S_per_m']
    material = np.full((128, 128, 80), sigma['background'])
    material[:, :, centers[2] > 0] = sigma['air']
    block = ((x > bounds[0][0]) & (x < bounds[0][1]) &
             (y > bounds[1][0]) & (y < bounds[1][1]) &
             (z > bounds[2][0]) & (z < bounds[2][1]))
    material[block] = sigma['block']
    if not np.array_equal(arrays['sigma'], material):
        raise ValueError('fixed mesh materials do not match the current physical model')
    meta = mesh_metadata(arrays)
    if (meta['cells'] != 1310720 or meta['complex_edges'] != 4006224 or
            meta['real_equivalent_unknowns'] != 8012448 or meta['pec_free_complex_edges'] != 3858768):
        raise ValueError('fixed mesh must retain 8,012,448 stored real-equivalent unknowns')
    return meta


def prepare_mesh(root, create=False):
    directory = Path(root)/'meshes/emg3d_fixed'
    path, metapath = directory/'axes_sigma.npz', directory/'mesh.json'
    if path.exists() != metapath.exists():
        raise ValueError('partial mesh files exist; refusing to replace the fixed mesh')
    if not path.exists() or not metapath.exists():
        if not create:
            raise ValueError('fixed mesh is missing; run --stage mesh first')
        expected = fixed_mesh_arrays()
        expected_meta = _audit_mesh_arrays(expected)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_npz(path, **expected)
        expected_meta['axes_sigma_npz_sha256'] = sha256(path)
        write_json(metapath, expected_meta)
        return expected, expected_meta
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(metapath.read_text(encoding='utf-8'))
    expected_meta = _audit_mesh_arrays(arrays)
    if (any(object_hash(meta.get(name)) != object_hash(value) for name, value in expected_meta.items()) or
            meta.get('axes_sigma_npz_sha256') != sha256(path)):
        raise ValueError('fixed mesh identity changed; a second mesh is prohibited')
    return arrays, meta


def prepare_supports(root, arrays, create=False):
    directory = Path(root)/'meshes/emg3d_fixed'
    supports = {key: linear_support(arrays, key, receivers())
                for key in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')}
    audit_arrays = {'coords': receivers()}
    summary = {'method': 'linear', 'coordinates_unchanged': True,
               'receivers_hash': identity()['receivers_hash'], 'components': {}}
    for key, support in supports.items():
        for name, value in support.items():
            audit_arrays[key+'_'+name] = value
        selected = support['z'][support['weights'] != 0]
        summary['components'][key] = {'support_z_min_m': float(selected.min()),
            'support_z_max_m': float(selected.max()), 'nonzero_support_z_strictly_negative': True,
            'nonzero_support_count': int(np.count_nonzero(support['weights'])),
            'weight_sum_max_error': float(np.max(abs(support['weights'].sum(axis=1)-1.))),
            'indices_sha256': _array_hash(support['indices']), 'weights_sha256': _array_hash(support['weights'])}
    path = directory/'receiver_support.npz'
    metapath = directory/'receiver_support.json'
    have_arrays, have_metadata = path.is_file(), metapath.is_file()
    if not (have_arrays and have_metadata) and not create:
        raise ValueError('receiver support audit is missing or partial; run --stage mesh')
    # Validate retained files before rebuilding a missing companion. Existing
    # inconsistent sampling data must never be silently replaced.
    if have_arrays:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(audit_arrays) or any(not np.array_equal(saved[name], value) for name, value in audit_arrays.items()):
                raise ValueError('receiver support audit changed')
    saved_meta = None
    if have_metadata:
        saved_meta = json.loads(metapath.read_text(encoding='utf-8'))
        if (saved_meta.get('components') != summary['components'] or
                saved_meta.get('receivers_hash') != summary['receivers_hash'] or
                (have_arrays and saved_meta.get('npz_sha256') != sha256(path))):
            raise ValueError('receiver support audit metadata changed')
    if not have_arrays:
        atomic_npz(path, **audit_arrays)
    if not (have_arrays and have_metadata):
        summary['npz_sha256'] = sha256(path)
        write_json(metapath, summary)
    else:
        summary = saved_meta
    return supports, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / 'data/mesh_data/case1_block/cross_validation/emg3d')
    parser.add_argument('--stage', choices=('mesh',), default='mesh')
    parser.add_argument('--plan', action='store_true', help='Print the formal grid size without generating files')
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(dict(output_root=str(args.root.resolve()), shape_cells=[128, 128, 80], cells=1310720, complex_edges=4006224, pec_free_complex_edges=3858768, core=CORE), indent=2))
        return 0
    arrays, metadata = prepare_mesh(args.root.resolve(), create=True)
    _, audit = prepare_supports(args.root.resolve(), arrays, create=True)
    print(json.dumps(dict(status='complete', stage='mesh', mesh_hash=metadata['mesh_hash'], shape_cells=metadata['shape_cells'], receiver_support=audit['components'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
