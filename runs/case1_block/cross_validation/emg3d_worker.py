"""Independent emg3d 1.9.1 worker for one immutable 128 x 128 x 80 mesh.

The project H contract is explicitly the negative of native emg3d H. No
frequency-, receiver-, or component-dependent corrections are permitted.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "core"), str(ROOT)]
from runs.common.mesh_inputs import get_case_mesh

import numpy as np

if __package__:
    from . import validation_common
    from .validation_common import (FREQUENCIES, identity, physical_config, receivers,
        save_frequency, write_json, sha256, object_hash)
else:
    import validation_common
    from validation_common import (FREQUENCIES, identity, physical_config, receivers,
        save_frequency, write_json, sha256, object_hash)


VERSION = '1.9.1'
SOURCE_COMMIT = '0624e3c9779503793d271dd940daca59e2a3fa41'
H_MAPPING = 'H_contract=-H_native; H_native=+curl(E)/(i*omega*mu); H_contract=-curl(E)/(i*omega*mu)'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'BLIS_NUM_THREADS')
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


def apply_support(values, support):
    indices = support['indices']
    return np.sum(values[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]*support['weights'], axis=1)


def _curl_terms(electric):
    hx, hy, hz = electric.grid.h
    ex, ey, ez = electric.fx, electric.fy, electric.fz
    return (((ez[:, 1:, :]-ez[:, :-1, :])/hy[None, :, None],
             (ey[:, :, 1:]-ey[:, :, :-1])/hz[None, None, :]),
            ((ex[:, :, 1:]-ex[:, :, :-1])/hz[None, None, :],
             (ez[1:, :, :]-ez[:-1, :, :])/hx[:, None, None]),
            ((ey[1:, :, :]-ey[:-1, :, :])/hx[:, None, None],
             (ex[:, 1:, :]-ex[:, :-1, :])/hy[None, :, None]))


def discrete_curl(electric):
    return tuple(term1-term2 for term1, term2 in _curl_terms(electric))


def contract_h(native):
    return -np.asarray(native, dtype=np.complex128)


def accept_solution(electric, source, info, residual):
    if info.get('exit') != 0:
        raise ValueError('native solver exit must be 0')
    if any(not np.all(np.isfinite(field.field)) for field in (electric, source, residual)):
        raise ValueError('nonfinite electric field, source, or true residual')
    norm_source = np.linalg.norm(source.field)
    measured = float(np.linalg.norm(residual.field)/norm_source) if norm_source > 0 else float('inf')
    if not np.isfinite(measured) or measured > 1e-6:
        raise ValueError(f'true residual {measured:.12g} exceeds 1e-6')
    return measured


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


def completed(output, index, checkpoint_key):
    output = Path(output)
    path = output/f'frequency_{index:02d}.npz'
    if not path.exists():
        return False
    shard = validation_common._read_shard(path)
    if shard['index'] != index:
        raise ValueError('resume rejected: shard filename/index mismatch')
    meta = shard['meta']
    if meta.get('checkpoint_key') != checkpoint_key:
        raise ValueError('resume rejected: checkpoint input/implementation/runtime key changed')
    if meta.get('status') != 'complete':
        return False
    if meta.get('native_exit') != 0:
        raise ValueError('resume rejected: native exit must be 0')
    if shard['E'].dtype != np.complex128 or shard['H'].dtype != np.complex128:
        raise ValueError('resume rejected: all six components must be complex128')
    sidecar = output/f'frequency_{index:02d}_native_H.npz'
    if not sidecar.is_file() or meta.get('native_H_sha256') != sha256(sidecar):
        raise ValueError('resume rejected: native H sidecar missing or checksum changed')
    with np.load(sidecar, allow_pickle=False) as data:
        native, coords = data['H_native'], data['coords']
        side_meta = json.loads(str(data['meta']))
    normalization_fields = ('normalization_policy', 'normalization_applied', 'source_norm',
        'normalized_norm', 'restoration_norm', 'original_source_sha256', 'solver_source_sha256',
        'native_exit', 'residual')
    if (meta.get('normalization_policy') != normalization_policy() or
            any(key not in meta or key not in side_meta or meta[key] != side_meta[key]
                for key in normalization_fields)):
        raise ValueError('resume rejected: normalization/native exit/residual metadata differs from native H sidecar')
    if (side_meta.get('checkpoint_key') != checkpoint_key or
            side_meta.get('index') != index or side_meta.get('frequency_hz') != float(FREQUENCIES[index]) or
            side_meta.get('H_contract_mapping') != H_MAPPING or meta.get('H_contract_mapping') != H_MAPPING or
            native.dtype != np.complex128 or native.shape != (3321, 3) or
            not np.array_equal(coords, receivers()) or not np.all(np.isfinite(native)) or
            not np.array_equal(contract_h(native), shard['H'])):
        raise ValueError('resume rejected: invalid native H sidecar or sign mapping')
    return True


def execution_key(parts):
    return object_hash(parts)


def solver_attempts():
    fixed = {'cycle': 'F', 'semicoarsening': True, 'linerelaxation': True,
        'tol': 1e-6, 'nu_init': 0, 'nu_pre': 2, 'nu_coarse': 1, 'nu_post': 2,
        'clevel': -1, 'return_info': True, 'verb': 3}
    return [{**fixed, 'sslsolver': method, 'maxit': maximum}
            for method, maximum in [('bicgstab', 100), ('bicgstab', 400), ('gcrotmk', 400)]]


def normalization_policy():
    return {'name': 'gcrot_rhs_l2_v1', 'bicgstab': 'original_physical_source',
        'gcrotmk': 'positive_real_l2_rhs_normalization_then_electric_restoration',
        'scale_definition': 'norm(original_source.field,2)', 'applies_to_attempts': [3],
        'residual_source': 'original_physical_source', 'fitted_gain_or_phase': False}


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
    if create:
        raise ValueError('generate mesh inputs with mesh_gen/generate.py, not a runs entry')
    directory = Path(root)
    path, metapath = directory/'axes_sigma.npz', directory/'mesh.json'
    if path.exists() != metapath.exists():
        raise ValueError('partial mesh files exist; refusing to replace the fixed mesh')
    if not path.exists() or not metapath.exists():
        if not create:
            raise ValueError('fixed mesh is missing; run mesh_gen/generate.py block_emg3d')
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


def single_cpu_environment(environ=None, affinity=None, threadpools=None, numba_threads=None):
    environ = os.environ if environ is None else environ
    required = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')
    if any(environ.get(name) != '1' for name in required):
        raise ValueError('single thread environment requires OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1')
    if any(environ.get(name, '1') != '1' for name in THREAD_VARIABLES):
        raise ValueError('all configured native thread limits must be 1')
    for name in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE', 'SLURM_NTASKS'):
        if int(environ.get(name, '1')) != 1:
            raise ValueError(f'{name} must be 1')
    if affinity is None:
        if not hasattr(os, 'sched_getaffinity'):
            raise ValueError('actual CPU affinity audit requires Linux sched_getaffinity')
        affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise ValueError('actual CPU affinity must contain exactly one CPU')
    if threadpools is None:
        from threadpoolctl import threadpool_info
        threadpools = threadpool_info()
    if any(pool.get('num_threads') != 1 for pool in threadpools):
        raise ValueError('actual BLAS/OpenMP thread count must be 1')
    if numba_threads is None:
        import numba
        numba_threads = numba.get_num_threads()
    if numba_threads != 1:
        raise ValueError('actual numba thread count must be 1')
    return {'env': {name: environ.get(name) for name in THREAD_VARIABLES},
            'affinity': list(affinity), 'threadpools': threadpools, 'numba_threads': numba_threads}


def _resources():
    result = {'process_cpu_seconds': time.process_time()}
    if hasattr(os, 'sched_getaffinity'):
        result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(max_rss_kib=usage.ru_maxrss, user_seconds=usage.ru_utime,
                      system_seconds=usage.ru_stime)
    except ImportError:
        pass
    return result


def _utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def frequency_lock(output, index):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'.frequency_{index:02d}.lock'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'frequency lock already exists: {path}; inspect the owning job before removing it') from error
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'host': platform.node(), 'created_utc': _utc(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID')}, stream)
        yield
    finally:
        path.unlink()


def _native_samples(field, coords):
    orientations = [(0., 0.), (90., 0.), (0., 90.)]
    return np.column_stack([field.get_receiver((*coords.T, azimuth, elevation), method='linear')
                            for azimuth, elevation in orientations]).astype(np.complex128)


def _curl_check(electric, native_h, mu, frequency):
    terms = _curl_terms(electric)
    components = {}
    factor = 2j*np.pi*frequency*mu
    epsilon = np.finfo(np.float64).eps
    for dimension, ((term1, term2), magnetic) in enumerate(zip(terms, (native_h.fx, native_h.fy, native_h.fz))):
        interior = [slice(None)]*3
        interior[dimension] = slice(1, -1)
        interior = tuple(interior)
        term1, term2 = term1[interior], term2[interior]
        expected = (term1-term2)/factor
        actual = magnetic[interior]
        norm1, norm2 = float(np.linalg.norm(term1)), float(np.linalg.norm(term2))
        reference_norm = float(np.linalg.norm(expected))
        difference_norm = float(np.linalg.norm(actual-expected))
        if reference_norm == 0:
            if norm1 != 0 or norm2 != 0 or difference_norm != 0:
                raise ValueError('undefined relative discrete curl audit for cancelling nonzero derivatives')
            relative, bound = 0., 0.
        else:
            relative = difference_norm/reference_norm
            # Reordered derivative products in native H can lose relative
            # accuracy when nearly equal derivative terms cancel. Budget the
            # rounding against their separate norms, with an absolute cap.
            bound = 64*epsilon*(norm1+norm2)/abs(factor)/reference_norm
        threshold = max(5e-12, min(bound, 1e-9))
        component = {'relative_error': relative, 'roundoff_bound': bound,
            'relative_threshold': threshold, 'derivative_term1_norm': norm1,
            'derivative_term2_norm': norm2, 'abs_omega_mu': float(abs(factor)),
            'reference_h_norm': reference_norm, 'machine_epsilon': float(epsilon),
            'roundoff_multiplier': 64}
        if not all(np.isfinite(value) for value in component.values()) or relative > threshold or relative > 1e-9:
            raise ValueError(f'native H does not match independent interior discrete curl: {component}')
        components['xyz'[dimension]] = component
    errors = [components[axis]['relative_error'] for axis in 'xyz']
    return {'component_relative_errors': errors, 'relative_error': max(errors),
            'components': components, 'base_relative_tolerance': 5e-12, 'absolute_relative_error_cap': 1e-9,
            'boundary_faces': 'native outer normal faces are zero; compared interior normal faces',
            'native_formula': '+curl(E)/(i*omega*mu)', 'passed': True}


def sample_fields(emg, model, electric, arrays, supports):
    coords = receivers()
    magnetic = emg.fields.get_magnetic_field(model, electric)
    curl_audit = _curl_check(electric, magnetic, physical_config()['mu_H_per_m'], electric._frequency)
    E, H_native = _native_samples(electric, coords), _native_samples(magnetic, coords)
    interpolation = {}
    for prefix, field, sampled in [('E', electric, E), ('H', magnetic, H_native)]:
        for column, axis in enumerate('xyz'):
            key = prefix+axis
            independent = apply_support(getattr(field, 'f'+axis), supports[key])
            scale = max(float(np.linalg.norm(independent)), np.finfo(float).tiny)
            error = float(np.linalg.norm(independent-sampled[:, column])/scale)
            if not np.isfinite(error) or error > 5e-12:
                raise ValueError(f'native linear receiver differs from audited tensor operator: {key} {error}')
            interpolation[key] = {'relative_error': error, 'passed': True}
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(H_native)):
        raise ValueError('nonfinite receiver response')
    return E, H_native, {'passed': True, 'method': 'linear', 'coordinates_unchanged': True,
                          'receiver_hash': object_hash(coords), 'interpolation': interpolation,
                          'independent_curl': curl_audit}


def solve_frequency(output, index, model, source, base, emg, arrays, supports, *,
                    frequency_hz=None, response_writer=None):
    frequency_started = time.perf_counter()
    output = Path(output)
    frequency = float(FREQUENCIES[index] if frequency_hz is None else frequency_hz)
    if response_writer is None and frequency != float(FREQUENCIES[index]):
        raise ValueError('an explicit nonlegacy frequency requires a matching response_writer')
    status_path = output/f'frequency_{index:02d}_status.json'
    row = {**base, 'index': index, 'frequency_hz': frequency,
           'status': 'running', 'started_utc': _utc(), 'attempts': []}
    write_json(status_path, row)
    for number, options in enumerate(solver_attempts(), 1):
        directory = output/'attempts'/f'frequency_{index:02d}'/f'{time.time_ns()}_{number}'
        directory.mkdir(parents=True)
        attempt = {'number': number, 'solver_options': options, 'started_utc': _utc(),
                   'status': 'running', 'resources_before': _resources()}
        write_json(directory/'attempt.json', attempt)
        tick = time.perf_counter()
        electric, info = None, {}
        try:
            norm_source = float(np.linalg.norm(source.field))
            if not np.isfinite(norm_source) or norm_source <= 0 or not np.all(np.isfinite(source.field)):
                raise ValueError('source normalization requires a finite positive real norm')
            original_hash = _array_hash(source.field)
            normalized = number == 3 and options['sslsolver'] == 'gcrotmk'
            solver_source = source.copy() if normalized else source
            restoration = norm_source if normalized else 1.
            if normalized:
                solver_source.field /= norm_source
            normalization = {'normalization_policy': normalization_policy(), 'normalization_applied': normalized,
                'source_norm': norm_source, 'normalized_norm': float(np.linalg.norm(solver_source.field)),
                'restoration_norm': restoration, 'original_source_sha256': original_hash,
                'solver_source_sha256': _array_hash(solver_source.field)}
            attempt.update(normalization)
            write_json(directory/'attempt.json', attempt)
            with (directory/'native_solver.log').open('w', encoding='utf-8') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    # Every fixed attempt starts with native zero initialization.
                    electric, info = emg.solve(model, solver_source, **options)
                    if normalized:
                        electric = electric.copy()
                        electric.field *= restoration
                    residual = emg.solver.residual(emg.models.VolumeModel(model, source),
                                                   source, electric, norm=False)
            attempt['native_info'] = info
            attempt['native_exit'] = info.get('exit')
            attempt['true_residual'] = float(np.linalg.norm(residual.field)/norm_source)
            attempt['residual_source'] = 'original_physical_source'
            # Persist the original-system measurement before any receiver/H audit.
            write_json(directory/'attempt.json', attempt)
            if _array_hash(source.field) != original_hash:
                raise ValueError('original physical source was mutated')
            measured = accept_solution(electric, source, info, residual)
            E, H_native, audit = sample_fields(emg, model, electric, arrays, supports)
            sidecar = output/f'frequency_{index:02d}_native_H.npz'
            side_meta = {**base, **normalization, 'index': index, 'frequency_hz': frequency,
                         'native_exit': int(info['exit']), 'residual': measured}
            atomic_npz(sidecar, coords=receivers(), H_native=H_native,
                       meta=json.dumps(validation_common._json_safe(side_meta), sort_keys=True))
            attempt.update(status='complete', residual=measured, receiver_audit=audit,
                           elapsed_seconds=time.perf_counter()-tick, resources_after=_resources())
            write_json(directory/'attempt.json', attempt)
            meta = {**base, **normalization, 'status': 'complete', 'native_exit': int(info['exit']),
                    'residual': measured, 'residual_definition': 'norm(residual(VolumeModel(model,sfield),sfield,E,norm=False).field)/norm(sfield.field)',
                    'native_reported_rel_error': info.get('rel_error'), 'accepted_attempt': number,
                    'actual_solver_options': options, 'native_H_sha256': sha256(sidecar),
                    'attempt_directory': str(directory), 'receiver_audit': audit,
                    'resources': _resources(), 'elapsed_seconds': time.perf_counter()-tick,
                    'total_elapsed_seconds': time.perf_counter()-frequency_started}
            if response_writer is None:
                save_frequency(output, index, receivers(), E.astype(np.complex128), contract_h(H_native), meta)
            else:
                response_writer(output, index, receivers(), E.astype(np.complex128), contract_h(H_native),
                                {**meta, 'index': index, 'frequency_hz': frequency})
            row['attempts'].append(str(directory))
            row.update(status='complete', residual=measured, native_exit=0,
                       accepted_attempt=number, completed_utc=_utc(), resources=_resources(),
                       total_elapsed_seconds=time.perf_counter()-frequency_started, **normalization)
            write_json(status_path, row)
            return row
        except Exception as error:
            attempt.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                           elapsed_seconds=time.perf_counter()-tick, resources_after=_resources())
            if info.get('exit') == 0 and electric is not None:
                failed_field = directory/'electric_physical_coefficients.npz'
                try:
                    atomic_npz(failed_field, electric_physical=electric.field, source_original=source.field,
                        frequency_hz=frequency, mesh_hash=base['mesh_hash'],
                        true_residual=attempt.get('true_residual', np.nan))
                    attempt.update(failed_field_path=str(failed_field), failed_field_sha256=sha256(failed_field))
                except Exception as preserve_error:
                    attempt['failed_field_preservation_error'] = str(preserve_error)
            write_json(directory/'attempt.json', attempt)
            row['attempts'].append(str(directory))
            row.update(last_error=str(error), resources=_resources(),
                       total_elapsed_seconds=time.perf_counter()-frequency_started)
            write_json(status_path, row)
    row.update(status='failed', completed_utc=_utc(),
               total_elapsed_seconds=time.perf_counter()-frequency_started)
    write_json(status_path, row)
    return row


def prepare_supports(root, arrays, create=False):
    if create:
        raise ValueError('generate sampling supports with mesh_gen/generate.py')
    directory = Path(root)
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
    if not path.exists() or not metapath.exists():
        if not create or path.exists() or metapath.exists():
            raise ValueError('receiver support audit is missing or partial; run mesh_gen/generate.py block_emg3d')
        atomic_npz(path, **audit_arrays)
        summary['npz_sha256'] = sha256(path)
        write_json(metapath, summary)
    else:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(audit_arrays) or any(not np.array_equal(saved[name], value) for name, value in audit_arrays.items()):
                raise ValueError('receiver support audit changed')
        saved_meta = json.loads(metapath.read_text(encoding='utf-8'))
        if saved_meta.get('components') != summary['components'] or saved_meta.get('npz_sha256') != sha256(path):
            raise ValueError('receiver support audit metadata changed')
        summary = saved_meta
    return supports, summary


def load_native(source=None):
    if source is not None:
        source = Path(source).expanduser().resolve()
        if not (source/'emg3d/__init__.py').is_file():
            raise ValueError('--source must contain the pinned emg3d package directory')
        sys.path.insert(0, str(source))
    try:
        emg = importlib.import_module('emg3d')
    except ModuleNotFoundError as error:
        raise RuntimeError('emg3d v1.9.1 is an external dependency; use its existing environment or --source') from error
    if source is not None and not Path(emg.__file__).resolve().is_relative_to(source):
        raise ValueError('imported emg3d does not originate from --source')
    if emg.__version__ != VERSION:
        raise ValueError(f'expected emg3d {VERSION}, got {emg.__version__}')
    return emg


def runtime_identity(source, emg):
    source = Path(source).resolve()
    def git(*arguments):
        return subprocess.run(['git', '-C', str(source), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()
    commit = git('rev-parse', 'HEAD')
    if commit != SOURCE_COMMIT or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('emg3d source must be the clean, exact approved v1.9.1 commit')
    versions = {'python': platform.python_version(), 'emg3d': emg.__version__}
    for name in ('numpy', 'scipy', 'numba', 'llvmlite', 'empymod', 'discretize', 'scooby', 'threadpoolctl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {str(path.relative_to(source)).replace('\\', '/'): sha256(path)
               for path in sorted((source/'emg3d').rglob('*.py'))}
    environment = single_cpu_environment()
    # Loaded-library enumeration order can vary between processes. Keep the
    # original diagnostics, but hash a stable ordering without dropping fields.
    pools = sorted((dict(pool) for pool in environment['threadpools']), key=lambda pool: (
        str(pool.get('filepath', '')), str(pool.get('user_api', '')),
        str(pool.get('internal_api', '')), str(pool.get('prefix', '')), object_hash(pool)))
    canonical = {'versions': versions, 'source_commit': commit, 'native_source_sha256': sources,
                 'thread_policy': 'one process, one affinity CPU, one native thread',
                 'threadpools': pools, 'platform': platform.platform(),
                 'machine': platform.machine(), 'python_executable': str(Path(sys.executable).resolve())}
    return {'runtime_hash': object_hash(canonical), 'emg3d_version': emg.__version__,
        'source_commit': commit, 'versions': versions, 'native_source_sha256': sources,
        'source_tree_hash': object_hash(sources), 'source_path': str(source),
        'imported_package_path': str(Path(emg.__file__).resolve()),
        'single_cpu_environment': environment, 'runtime_canonical': canonical}


def make_model(emg, arrays):
    from scipy.constants import mu_0
    grid = emg.TensorMesh([arrays['h'+axis] for axis in 'xyz'], origin=[-100000.]*3)
    for axis in 'xyz':
        if not np.array_equal(getattr(grid, 'nodes_'+axis), arrays['nodes_'+axis]):
            raise ValueError('native mesh reconstructed different interface nodes')
    model = emg.Model(grid, property_x=arrays['sigma'], mapping='Conductivity',
                      mu_r=physical_config()['mu_H_per_m']/mu_0, epsilon_r=None)
    return grid, model


def make_source(emg, grid, frequency):
    if not frequency > 0:
        raise ValueError('only positive frequency-domain sources are allowed')
    source = physical_config()['source']
    wire = emg.TxElectricWire([source['start_m'], source['end_m']], strength=source['current_A'])
    return emg.get_source_field(grid, wire, frequency=float(frequency))


def native_smoke(emg=None):
    if emg is None:
        emg = importlib.import_module('emg3d')
    from scipy.constants import mu_0
    # Tiny uniform kernel fixture only; this is never saved as a campaign mesh.
    grid = emg.TensorMesh([np.full(8, 100.)]*3, origin=[-2700., -400., -400.])
    mu = physical_config()['mu_H_per_m']
    model = emg.Model(grid, property_x=.01, mapping='Conductivity', mu_r=mu/mu_0, epsilon_r=None)
    manufactured = emg.Field(grid, frequency=1.)
    manufactured.fz[:] = grid.nodes_x[:, None, None]
    magnetic = emg.fields.get_magnetic_field(model, manufactured)
    audit = _curl_check(manufactured, magnetic, mu, 1.)
    expected_y = -1/(2j*np.pi*mu)
    np.testing.assert_allclose(magnetic.fy[:, 1:-1, :], expected_y, rtol=1e-12)
    np.testing.assert_allclose(contract_h(magnetic.fy[:, 1:-1, :]), -expected_y, rtol=1e-12)
    source = physical_config()['source']
    wire = emg.TxElectricWire([source['start_m'], source['end_m']], strength=1.)
    vector = emg.get_source_field(grid, wire, frequency=None)
    moment = [float(np.sum(getattr(vector, 'f'+axis))) for axis in 'xyz']
    np.testing.assert_allclose(moment, [500., 0., 0.], atol=1e-10, rtol=1e-12)
    sfield = make_source(emg, grid, 1.)
    double = make_source(emg, grid, 2.)
    np.testing.assert_allclose(sfield.field, -2j*np.pi*mu_0*vector.field, rtol=1e-12)
    np.testing.assert_allclose(double.field, 2*sfield.field, rtol=1e-12)
    options = solver_attempts()[0]
    electric, info = emg.solve(model, sfield, **options)
    residual = emg.solver.residual(emg.models.VolumeModel(model, sfield), sfield, electric, norm=False)
    measured = accept_solution(electric, sfield, info, residual)
    coords = np.array([[-2250., 0., -150.], [-2200., 50., -100.]])
    sampled = _native_samples(manufactured, coords)
    np.testing.assert_allclose(sampled[:, :2], 0., atol=1e-12)
    np.testing.assert_allclose(sampled[:, 2], coords[:, 0], rtol=1e-12)
    return {'passed': True, 'scope': 'tiny kernel/API fixture, not grid convergence',
        'fixture_shape': [8, 8, 8], 'source_moment_Am': moment,
        'positive_frequency_scaling_passed': True, 'curl_relative_error': audit['relative_error'],
        'manufactured_E': 'E_z=x; E_x=E_y=0; curl_y=-1', 'H_contract_mapping': H_MAPPING,
        'native_solver_info': info, 'true_residual': measured, 'native_exit': info['exit'],
        'linear_receiver_affine_field_passed': True}


def parse_indices(text):
    maximum = len(FREQUENCIES) - 1
    try:
        indices = [int(part) for part in text.split(',')]
    except ValueError as error:
        raise ValueError(f'--indices must contain comma-separated integers in 0..{maximum}') from error
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index > maximum for index in indices):
        raise ValueError(f'--indices must be unique integers in 0..{maximum}')
    return indices


def _base_metadata(mesh_meta, runtime):
    implementation = {'worker': sha256(__file__), 'validation_common': sha256(validation_common.__file__)}
    parts = {'inputs': identity(), 'mesh_hash': mesh_meta['mesh_hash'],
             'implementation_hash': object_hash(implementation), 'runtime_hash': runtime['runtime_hash'],
             'solver_hash': object_hash(solver_attempts()), 'H_contract_mapping': H_MAPPING,
             'receiver_interpolation': 'linear', 'dtype': 'complex128', 'normalization_policy': normalization_policy()}
    key = execution_key(parts)
    return {**identity(), **parts, 'execution_key': key, 'checkpoint_key': key,
        'implementation_source_sha256': implementation, 'runtime_identity': runtime,
        'method': 'emg3d', 'solver_profile': 'emg3d_fixed_attempts_v1', 'solver_options': solver_attempts(),
        'software_version': VERSION, 'source_commit': SOURCE_COMMIT, 'time_convention': '+iwt',
        'units_E': 'V/m', 'units_H': 'A/m', 'mpi_processes': 1, 'threads': 1,
        'cells': mesh_meta['cells'], 'unknowns': mesh_meta['complex_edges'],
        'real_equivalent_unknowns': mesh_meta['real_equivalent_unknowns'],
        'pec_free_complex_edges': mesh_meta['pec_free_complex_edges'], 'system_scalar_type': 'complex128',
        'material_mapping': 'Conductivity', 'mu_H_per_m': physical_config()['mu_H_per_m'],
        'epsilon_r': None, 'physical_config': physical_config()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='input data root, default checkout/data')
    parser.add_argument('--source', type=Path, help='external pinned emg3d checkout, or use installed checkout')
    parser.add_argument('--stage', default='solve', choices=('smoke', 'solve'))
    parser.add_argument('--indices', default=','.join(map(str, range(len(FREQUENCIES)))))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args(argv)
    inputs = get_case_mesh('block_emg3d', data_root=args.data_root, require=True)
    directory = inputs['m_dir']
    indices = parse_indices(args.indices)
    arrays, mesh_meta = prepare_mesh(directory)
    supports, _ = prepare_supports(directory, arrays)
    if args.check_inputs:
        print(json.dumps(dict(status='passed', inputs={k: str(v) for k, v in inputs.items()},
                              shape_cells=mesh_meta['shape_cells']), indent=2))
        return 0
    from datetime import datetime, timezone
    from uuid import uuid4
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    output = (args.output or ROOT/'outputs/case1_block/cross_validation'/tag/'emg3d').resolve()
    data_directory = (args.data_root or ROOT/'data').resolve()
    if output == data_directory or data_directory in output.parents:
        raise ValueError('--output must be outside the input data directory')
    emg = load_native(args.source)
    runtime = runtime_identity(args.source or Path(emg.__file__).resolve().parent.parent, emg)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'smoke':
        tick = time.perf_counter()
        result = native_smoke(emg)
        write_json(output/'smoke.json', {**result, 'runtime_identity': runtime,
                   'elapsed_seconds': time.perf_counter()-tick, 'resources': _resources()})
        print(json.dumps({'status': 'complete', 'stage': 'smoke', 'true_residual': result['true_residual']}))
        return 0
    grid, model = make_model(emg, arrays)
    base = _base_metadata(mesh_meta, runtime)
    base["input_paths"] = {k: str(v) for k, v in inputs.items()}
    failed = []
    for index in indices:
        with frequency_lock(output, index):
            if completed(output, index, base['checkpoint_key']):
                print(json.dumps({'index': index, 'status': 'skipped_complete', 'execution_key': base['execution_key']}))
                continue
            source = make_source(emg, grid, float(FREQUENCIES[index]))
            row = solve_frequency(output, index, model, source, base, emg, arrays, supports)
            print(json.dumps({'index': index, 'status': row['status'], 'execution_key': base['execution_key']}))
            if row['status'] != 'complete':
                failed.append(index)
    # Slurm array workers never merge globally; the coordinator calls common.merge_responses.
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
