#!/usr/bin/env python3
"""P1 layered-sphere worker using portable data/outputs and release solvers.

Adapted from the frozen 84ba36e frequency-count worker. MPI defaults to the
P1 historical Kg kernel from 86c71e; the frequency-count experiment defaults
to current release core. --kernel current selects the newer MPI variant.
Performance monitoring belongs to the launcher. --reference retains the
original full-field HDF5 check and requires --save-fields reference output;
lightweight receiver-only comparisons belong to common.receiver_metrics.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "core"), str(ROOT)]

from runs.common.mesh_inputs import get_case_mesh, runtime_mesh

ORIGINAL_FREQUENCY_RUNNER_SHA256 = "04597c19699f01dcf86bc6e99e8b5bbaeeeea31ce9bbe17c7d60c1199a4ef985"
ORIGINAL_FREQUENCY_COMMIT = "84ba36e7149b156cc69dd39fce2595c7b42d6da3"
ORIGINAL_MPI_COMMIT = "86c71eafa94204a6fc69a7a6b0bd829e009ac7fa"

FREQUENCIES = [2.**i for i in range(-6, 14)]
TEST2_N_FREQUENCIES = (2, 8, 10, 16, 20, 30, 40)
REPRESENTATIVE = [2.**-6, 16., 8192.]
SUMOR_OPTIONS = dict(poles=[-2274.7560643370834, -2.221441469079183],
                     pole_every=20, pole_schedule='hold_first', max_basis=400,
                     basis_strategy='residual_mr', field_formulation='split_u',
                     inner_rtol=1e-6, tol=1e-6, direct=False,
                     residual_correction=False)
FIELD_METRICS = ('field_E_relative_M', 'field_E_relative_L2',
                 'receiver_E_relative_L2', 'receiver_H_relative_L2')

# MUMPS 5.6.2 Users' Guide, pp. 98-99, from the official release archive.
# Keep raw INFOG values: negative entry counts use units of one million.
MUMPS_562_LABELS = dict(
    icntl={'7': 'sequential_ordering_control', '10': 'maximum_iterative_refinements',
           '28': 'analysis_sequential_or_parallel_control', '29': 'parallel_ordering_control',
           '35': 'BLR_factorization_control'},
    cntl={'1': 'relative_pivoting_threshold'},
    infog={'1': 'last_MUMPS_call_status', '2': 'last_MUMPS_call_additional_status',
           '7': 'actual_ordering_code_interpret_using_INFOG_32',
           '9': 'factor_real_or_complex_workspace_entries_global_sum',
           '10': 'factor_integer_workspace_entries_global_sum',
           '11': 'largest_frontal_matrix_order_after_factorization',
           '16': 'estimated_incore_factor_memory_MB_max_rank',
           '17': 'estimated_incore_factor_memory_MB_sum_ranks',
           '18': 'allocated_factor_memory_MB_max_rank',
           '19': 'allocated_factor_memory_MB_sum_ranks',
           '20': 'estimated_full_rank_factor_entries_global_sum',
           '21': 'effectively_used_factor_memory_MB_max_rank',
           '22': 'effectively_used_factor_memory_MB_sum_ranks',
           '29': 'effective_factor_entries_full_rank_assumption_global_sum',
           '32': 'actual_analysis_type_1_sequential_2_parallel',
           '35': 'effective_factor_entries_with_BLR_global_sum'})


class MumpsSetupObserver:
    """Read direct factors once after native setup; never set numerical options.

    The original class and every solve method run unchanged. Getters run on all
    ranks in the same order. Raw values are retained per rank; derived global
    summaries explicitly use rank 0. Collection overhead remains in full wall.
    """
    def __init__(self, mor):
        initialized = time.perf_counter()
        self.mor, self.original, self.records = mor, mor._LinearSolve, []
        header = Path(os.environ.get('CONDA_PREFIX', ''))/'include'/'dmumps_c.h'
        version, header_sha = None, None
        if header.is_file():
            match = re.search(r'^#define MUMPS_VERSION\s+"([^"]+)"', header.read_text(), re.MULTILINE)
            version = match.group(1) if match else None
            header_sha = sha256(header)
        self.provenance = dict(original_class_source_sha256=hashlib.sha256(
            inspect.getsource(self.original).encode()).hexdigest(),
            original_class=self.original.__module__+'.'+self.original.__qualname__,
            original_class_source_file=inspect.getsourcefile(self.original),
            expected_mumps_version='5.6.2', observed_mumps_header_version=version,
            mumps_header_sha256=header_sha, labels_match_observed_version=version == '5.6.2',
            labels=MUMPS_562_LABELS, labels_guide_pages=[71, 82, 98, 99],
            ordering_code_labels=dict(
                sequential={'0': 'AMD', '1': 'user_PERM_IN', '2': 'AMF', '3': 'SCOTCH',
                            '4': 'PORD', '5': 'Metis', '6': 'QAMD', '7': 'automatic'},
                parallel={'0': 'automatic', '1': 'PT-SCOTCH', '2': 'ParMetis'}),
            labels_archive_url='https://mumps-solver.org/MUMPS_5.6.2.tar.gz',
            labels_archive_sha256='13a2c1aff2bd1aa92fe84b7b35d88f43434019963ca09ef7e8c90821a8f1d59a',
            labels_guide_sha256='32acdd3e09fb69f9fab16c94ae67768d15c61ac9c27abf66eb1e0e6ecd904050',
            memory_MB_definition='1000000 bytes; MUMPS internal memory, not process PSS',
            collection_policy='one getter snapshot per successful native direct setup; raw ranks gathered only to rank 0; no numerical setter or per-RHS viewer',
            timing_policy='observer overhead included in complete_forward_s; separately recorded without subtraction')
        observer = self

        class ObservedLinearSolve(self.original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if self.direct:
                    observer.capture(self)

        mor._LinearSolve = ObservedLinearSolve
        self.initialization_wall_s = time.perf_counter()-initialized

    def capture(self, solver):
        from mpi4py import MPI
        from petsc4py import PETSc
        started = time.perf_counter()
        matrix = solver.matrix
        comm = matrix.comm.tompi4py()
        raw = dict(rank=comm.rank, status='unavailable')
        try:
            # This is the *input* matrix's true sparse nnz. PETSc's MUMPS
            # factor.getInfo(nz_used) is only INFOG(20), an estimate, and its
            # memory field is hard-coded zero in PETSc 3.20.6.
            information = matrix.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
            raw['input_matrix_size'] = list(matrix.getSize())
            raw['input_matrix_local_rows'] = list(matrix.getOwnershipRange())
            raw['input_matrix_global_nnz_used'] = information['nz_used']
            factor = solver.ksp.getPC().getFactorMatrix()
            raw['factor_matrix_size'] = list(factor.getSize())
            raw['icntl'] = {key: int(factor.getMumpsIcntl(int(key))) for key in MUMPS_562_LABELS['icntl']}
            raw['cntl'] = {key: float(factor.getMumpsCntl(int(key))) for key in MUMPS_562_LABELS['cntl']}
            raw['infog'] = {key: int(factor.getMumpsInfog(int(key))) for key in MUMPS_562_LABELS['infog']}
            raw['status'] = 'available'
        except Exception as error:
            raw['error'] = type(error).__name__+': '+str(error)
        by_rank = comm.gather(raw, root=0)
        available = comm.bcast(all(row['status'] == 'available' for row in by_rank) if comm.rank == 0 else None, root=0)
        record = dict(setup_index=len(self.records), name=solver.name,
                      status='available' if available else 'incomplete', global_summary_rank=0)
        if comm.rank == 0:
            record['raw_by_rank'] = by_rank
        if comm.rank == 0 and by_rank[0]['status'] == 'available':
            host = by_rank[0]
            entries = host['infog']['29']
            record['global_summary'] = dict(input_matrix_size=host['input_matrix_size'],
                input_matrix_nnz=host['input_matrix_global_nnz_used'], factor_matrix_size=host['factor_matrix_size'],
                analysis_type_raw=host['infog']['32'], actual_ordering_code_raw=host['infog']['7'],
                factor_entries_raw=entries,
                factor_entries_count_or_million_unit_scale=entries if entries >= 0 else -entries*1000000,
                factor_entries_raw_unit='entries' if entries >= 0 else 'millions_of_entries',
                factor_entries_interpretation='full_rank_assumption; use raw INFOG(35) for BLR storage')
            analysis = {1: 'sequential', 2: 'parallel'}.get(host['infog']['32'])
            record['global_summary']['actual_analysis_type_label'] = analysis
            record['global_summary']['actual_ordering_label'] = (
                self.provenance['ordering_code_labels'].get(analysis, {}).get(str(host['infog']['7']))
                if self.provenance['labels_match_observed_version'] else None)
        duration = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
        record['capture_wall_s_max'] = duration
        self.records.append(record)
        solver.stats['seconds']['mumps_diagnostic_capture'] = solver.stats['seconds'].get('mumps_diagnostic_capture', 0.)+duration
        solver.stats['counts']['mumps_diagnostic_setups'] = solver.stats['counts'].get('mumps_diagnostic_setups', 0)+1

    def restore(self):
        self.mor._LinearSolve = self.original


def require(condition, message):
    if not condition:
        raise ValueError(message)


def test2_frequencies(n_frequency):
    """Frozen Test 2 log-frequency grid, including both band endpoints."""
    require(isinstance(n_frequency, int) and n_frequency in TEST2_N_FREQUENCIES,
            'Test 2 --n-frequency must be one of '+str(TEST2_N_FREQUENCIES))
    return [2.**(-6 + 19*j/(n_frequency-1)) for j in range(n_frequency)]


def resolve_frequencies(args):
    """Validate the selected experiment without importing the FEM runtime."""
    experiment = getattr(args, 'experiment', 'test1')
    n_frequency = getattr(args, 'n_frequency', None)
    require(experiment in ('test1', 'test2'), 'Unknown experiment')
    if experiment == 'test2':
        require(args.freq is None, 'Test 2 cannot use --freq; it is mutually exclusive with --n-frequency')
        require(n_frequency is not None, 'Test 2 requires --n-frequency')
        fs = test2_frequencies(n_frequency)
    else:
        require(n_frequency is None, '--n-frequency requires --experiment test2')
        fs = list(args.freq or FREQUENCIES)
    require(fs and all(math.isfinite(f) and f > 0 for f in fs) and fs == sorted(set(fs)),
            'Frequencies must be unique, finite, positive and ascending')
    if experiment == 'test1':
        require(all(f in FREQUENCIES for f in fs), 'Frequency subset must belong to the frozen twenty-frequency grid')
    if args.method == 'CPC':
        require(fs[0] == FREQUENCIES[0] and fs[-1] == FREQUENCIES[-1],
                'CPC subset must retain both band endpoints to preserve its geometric pole')
    return fs


def conductivity(role):
    require(role in ('sphere', 'no_sphere'), 'Unknown model role')
    return [1e-8, .01, .001, .02, 1. if role == 'sphere' else .001]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(item):
    if hasattr(item, 'tolist'):
        return clean_json(item.tolist())
    if isinstance(item, dict):
        return {str(key): clean_json(value) for key, value in item.items()}
    if isinstance(item, (tuple, list)):
        return [clean_json(value) for value in item]
    if isinstance(item, float) and not math.isfinite(item):
        return None
    return item


def write_json_rank0(path, value, rank):
    if rank == 0:
        Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


def rank0_call(comm, function):
    """Propagate rank-zero I/O errors before entering the next collective."""
    error, value = None, None
    if comm.rank == 0:
        try:
            value = function()
        except Exception as caught:
            error = type(caught).__name__+': '+str(caught)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    return comm.bcast(value, root=0)


def validate_geometry(parameters, rx):
    import numpy as np
    require(parameters.get('n_layers') == 3, 'Expected three ground layers')
    require(parameters.get('grounding') == [True], 'Expected a grounded, open source path')
    tx = np.asarray(parameters.get('tx'), dtype=float)
    require(tx.ndim == 3 and tx.shape[0] == 1 and tx.shape[1] >= 2 and tx.shape[2] == 3,
            'Expected one complete mesh-fitted source polyline')
    source = tx[0]
    require(np.isfinite(source).all() and np.all(np.diff(source[:, 0]) > 0) and
            np.all(np.abs(source[:, 1]) < 1e-7) and
            np.allclose(source[[0, -1], 0], [-2250., -1750.], rtol=0, atol=1e-7),
            'Source must follow the surface from (-2250,0) to (-1750,0), without loop closure')
    rx = np.asarray(rx, dtype=float)
    expected_xy = np.array([[x, y] for x in range(-225, 226, 50) for y in range(-475, 476, 50)])
    require(rx.shape == (200, 3) and np.isfinite(rx).all(), 'Expected exactly 200 finite receivers')
    require(np.allclose(rx[:, :2], expected_xy, rtol=0, atol=1e-7), 'Receiver order must be x-major then y-minor')


def input_metadata(args):
    import numpy as np
    mesh = args.mesh.resolve()
    metadata = json.loads((mesh/'mesh.json').read_text())
    name = metadata.get('mesh_name', 'terrain_sphere')
    h5 = mesh/'meshes'/'_h5'/(name+'.h5')
    params_file = mesh/'meshes'/'para'/(name+'_parameters.json')
    params = json.loads(params_file.read_text())
    rx_file = args.rx.resolve() if args.rx else mesh/'receivers.csv'
    with rx_file.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rx = np.array([[float(row[key]) for key in ('x_m', 'y_m', 'z_m')] for row in rows])
    require([int(row['rx_id']) for row in rows] == list(range(1, 201)), 'Receiver CSV IDs must be 1..200')
    validate_geometry(params, rx)
    require(np.array_equal(np.vstack(params['rx']), rx), 'CSV receivers differ from mesh parameters')
    require(metadata.get('fe_degree') == 1, 'Test 1 requires P1 FEM')
    files = dict(mesh=h5, parameters=params_file, receivers=rx_file, mesh_metadata=mesh/'mesh.json')
    for key, filename in (('source', 'source.csv'), ('source_edge_audit', 'source_edge_audit.json'),
                          ('receiver_audit', 'receiver_audit.json'), ('geometry_audit', 'geometry_audit.json')):
        if (mesh/filename).exists():
            files[key] = mesh/filename
    hashes = {key: sha256(path) for key, path in files.items()}
    require(hashes['mesh'] == metadata['mesh_sha256'], 'Mesh HDF5 SHA mismatch')
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(metadata=metadata, parameters=params, mesh_name=name, rx=rx.tolist(),
                input_sha256=hashes, geometry_fingerprint=fingerprint,
                file_paths={key: str(path) for key, path in files.items()})


def receiver_geometry(ops, comm):
    """Physical cell identity is independent of repartition/global cell numbering."""
    import numpy as np
    local = []
    for index, cell, _, _ in ops['receiver_cells']:
        xyz = np.asarray(cell.get_vertex_coordinates()).reshape(-1, 3)
        xyz = xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]
        key = hashlib.sha256(np.ascontiguousarray(xyz, dtype='<f8').tobytes()).hexdigest()
        local.append(dict(index=int(index), owner=comm.rank, cell_global_id=int(cell.global_index()),
                          physical_cell_sha256=key, cell_vertices=xyz.tolist()))
    all_rows = [row for part in comm.allgather(local) for row in part]
    all_rows.sort(key=lambda row: row['index'])
    require([row['index'] for row in all_rows] == list(range(len(ops['rx']))),
            'Receiver sampling must have exactly one owner per receiver')
    return all_rows


def load_model(args, inputs, fs):
    from custEM.core import MOD
    actual_paths = get_case_mesh('sphere', data_root=args.data_root, require=True)
    from mpi4py import MPI
    view = rank0_call(MPI.COMM_WORLD,
                      lambda: {key: str(value) for key, value in runtime_mesh(actual_paths, args.run).items()})
    model = MOD('terrain_'+args.method+'_'+args.role, inputs['mesh_name'], 'E_t', p=1,
                ned_kind='1st', debug_level=30, overwrite_mesh=False, overwrite_results=False,
                file_format='h5', profiler=False, export_domains=False,
                m_dir=view['m_dir'],
                para_dir=str(Path(view['parameters']).parent),
                r_dir=str(args.run.resolve()/'native_results'))
    model.MP.rx, model.MP.n_rx = [inputs['rx']], 1
    sigma = conductivity(args.role)
    model.MP.update_model_parameters(frequencies=fs, current=1., sigma_air=sigma[0],
                                     sigma_ground=sigma[1:])
    model.FE.build_var_form(s_type='path', tx=model.MP.tx, grounding=model.MP.grounding,
                            quasi_static=True, ip=False, bc=None)
    require(model.FE.grounding == [True], 'Native source lost its grounded status')
    import numpy as np
    require(np.array_equal(model.FE.tx, inputs['parameters']['tx']), 'Native source polyline differs from mesh parameters')
    require(np.asarray(model.MP.sigma).shape == (5, 1) and
            np.array_equal(np.asarray(model.MP.sigma)[:, 0], sigma),
            'Native domain conductivities differ from requested model role')
    return model


def hardware(comm):
    """Descriptive metadata only; resource verification belongs to the launcher."""
    cpuinfo = Path('/proc/cpuinfo')
    cpu_model = next((line.split(':', 1)[1].strip() for line in
                      cpuinfo.read_text().splitlines() if line.startswith('model name')),
                     platform.processor()) if cpuinfo.is_file() else platform.processor()
    return dict(rank=comm.rank, pid=os.getpid(), hostname=platform.node(),
                cpu_model=cpu_model, numerical_threads=None,
                resource_verification='not_performed_by_worker')






def source_filenames(args):
    names = ['mor.py', 'operators.py', 'full_order.py']
    if args.kernel == 'paper_mpi':
        names.append('mor_mpi_legacy.py')
    return names


def kernel_module(args):
    """The retained historical module is specific to P1 SUPERMOR MPI runs."""
    if args.kernel == 'paper_mpi' and args.method == 'SUMOR':
        from supermor import mor_mpi_legacy
        return mor_mpi_legacy
    from supermor import mor
    return mor


def main(args):
    fs = resolve_frequencies(args)
    import numpy as np
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    selected_output = (str(args.run) if args.run is not None else str(
        ROOT / 'outputs' / 'case2_layered_sphere' /
        ('mpi_scaling' if args.experiment == 'test1' else 'frequency_count') /
        (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]))) if comm.rank == 0 else None
    args.run = Path(comm.bcast(selected_output, root=0)).expanduser().resolve()
    data_directory = (args.data_root or ROOT/'data').expanduser().resolve()
    if args.run == data_directory or data_directory in args.run.parents:
        raise ValueError('--output must be outside the input data directory')
    rank0_call(comm, lambda: args.run.mkdir(parents=True, exist_ok=False))
    comm.Barrier()
    started = time.perf_counter()  # Before metadata, mesh read, initialization and assembly.
    stats = dict(status='failed', method=args.method, model_role=args.role, fe_degree=1,
                 experiment=getattr(args, 'experiment', 'test1'), n_frequency=len(fs),
                 frequency_grid_sha256=hashlib.sha256(json.dumps(fs, separators=(',', ':')).encode()).hexdigest(),
                 acceptance_semantics='v2_equation_1e-6_reference_1e-10_separate',
                 production_commit=args.production_commit, mpi_ranks=comm.size,
                 threads_per_rank=None, resource_verification="external_launcher_responsibility", emitted_indices=[], emitted_residuals=[], failures=[])
    receivers, ops, model, mumps_observer = {}, None, None, None
    try:
        import custEM
        import dolfin as df
        from petsc4py import PETSc
        mor = kernel_module(args)
        from supermor.operators import assemble_mor, sample_receivers, store_field
        from supermor.full_order import full_order
        require(df.MPI.size(df.MPI.comm_world) == comm.size, 'DOLFIN MPI communicator mismatch')
        for key in ('PMI_SIZE', 'OMPI_COMM_WORLD_SIZE'):
            if key in os.environ:
                require(int(os.environ[key]) == comm.size, 'Launcher/MPI communicator size mismatch: '+key)
        require(not np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating), 'Require real PETSc build')
        mumps_observer = MumpsSetupObserver(mor)
        stats['mumps_observer'] = mumps_observer.provenance
        stats['mumps_observer_initialization_s_max'] = comm.allreduce(mumps_observer.initialization_wall_s, op=MPI.MAX)
        stats['mumps_diagnostics'] = mumps_observer.records
        df.set_log_level(40)
        rank_hardware = comm.allgather(hardware(comm))
        inputs = rank0_call(comm, lambda: input_metadata(args))
        stats.update(frequencies_hz=fs, mesh_sha256=inputs['input_sha256']['mesh'],
                     input_sha256=inputs['input_sha256'], geometry_fingerprint=inputs['geometry_fingerprint'],
                     conductivity_S_per_m=conductivity(args.role), rank_hardware=rank_hardware,
                     runner_sha256=sha256(__file__), receiver_domain_marker=args.receiver_domain,
                     production_source_sha256=rank0_call(comm, lambda: {
                         str(ROOT/'core'/'supermor'/name): sha256(ROOT/'core'/'supermor'/name)
                         for name in source_filenames(args)}),
                     kernel_selection=args.kernel,
                     historical_core_equivalence=(
                         'P1 historical MPI Kg definitions retained; shared P1 helpers audited'
                         if args.kernel == 'paper_mpi' else
                         'frequency_no_Kg_core_lineage; migrated release sources hashed separately'),
                     historical_production_commit=(ORIGINAL_FREQUENCY_COMMIT
                         if args.experiment == 'test2' else ORIGINAL_MPI_COMMIT),
                     original_frequency_runner_sha256=ORIGINAL_FREQUENCY_RUNNER_SHA256,
                     software=dict(python=sys.version, numpy=np.__version__, dolfin=df.__version__,
                                   PETSc=PETSc.Sys.getVersion(), custEM_file=custEM.__file__),
                     algorithm_protocol=(SUMOR_OPTIONS if args.method == 'SUMOR' else
                         dict(steps=80, max_cycles=10, tol=1e-6, pole=-71.08612701053386) if args.method == 'CPC' else
                         dict(outer='fgmres', restart=100, max_it=2000, rtol=1e-6,
                              preconditioner='PRESB', inner='CG/AMS', inner_rtol=1e-3, inner_max_it=300,
                              zero_initial_guess_each_frequency=True,
                              inner_true_residual_policy='diagnostic',
                              final_maxwell_true_residual_target=1e-6) if args.method == 'IFOM' else
                         dict(solver='MUMPS', true_residual_target=1e-10, maximum_same_LU_corrections=3,
                              retain_best_finite_residual=True)),
                     full_fields_saved=bool(args.save_fields), timing_policy='barrier; before mesh read through receiver/stat output; maximum rank wall time')
        rank0_call(comm, lambda: (args.run/'native_results'/'E_t'/inputs['mesh_name']).mkdir(parents=True))
        model = load_model(args, inputs, fs)
        ops = assemble_mor(model, receiver_domain=args.receiver_domain)
        actual = dict(cells=int(model.FS.mesh.num_entities_global(3)),
                      vertices=int(model.FS.mesh.num_entities_global(0)), field_dofs=model.FS.V.dim(),
                      mixed_dofs=model.FS.M.dim())
        for key, value in actual.items():
            require(value == int(inputs['metadata'][key]), 'Actual '+key+' differs from mesh.json')
        stats.update(**actual, assembly_seconds=ops['assembly_seconds'],
                     receiver_geometry=receiver_geometry(ops, comm),
                     ownership=comm.allgather(dict(rank=comm.rank, K_rows=ops['K'].getOwnershipRange(),
                         q=ops['q'].getOwnershipRange(), local_q_norm=float(np.linalg.norm(ops['q'].getArray(readonly=True))))),
                     global_source_norm=float(ops['q'].norm()),
                     initialization_and_assembly_s=comm.allreduce(time.perf_counter()-started, op=MPI.MAX))
        reference_rows = None
        if args.mpi_check:
            require(args.reference is not None, '--mpi-check requires --reference')
        if args.reference is not None:
            def read_reference():
                saved = json.loads((args.reference/'stats.json').read_text())
                require(saved['status'] == 'converged', 'Reference did not converge')
                for key in ('production_commit', 'mesh_sha256', 'geometry_fingerprint', 'model_role',
                            'frequencies_hz', 'conductivity_S_per_m', 'field_dofs', 'receiver_domain_marker'):
                    require(saved[key] == stats[key], 'Reference '+key+' mismatch')
                expected_cells = [row['physical_cell_sha256'] for row in saved['receiver_geometry']]
                actual_cells = [row['physical_cell_sha256'] for row in stats['receiver_geometry']]
                require(expected_cells == actual_cells, 'Receiver physical cells changed across partitions')
                require(saved['full_fields_saved'], 'Reference has no full-field checkpoints')
                if args.mpi_check:
                    require(saved['method'] == args.method and saved['mpi_ranks'] == 1,
                            'MPI check requires same-method one-rank reference')
                with np.load(args.reference/'receivers.npz', allow_pickle=False) as archive:
                    require(np.array_equal(archive['frequency_indices'], np.arange(len(fs))),
                            'Reference receiver frequencies incomplete')
                    data = {key: archive[key].copy() for key in ('E', 'H', 'rx_xyz')}
                    per_frequency = saved.get('solver_stats', {}).get('per_frequency', [])
                    data['reference_accepted'] = {row['index']: row.get('reference_accepted')
                                                  for row in per_frequency}
                    return data
            reference_rows = rank0_call(comm, read_reference)
            stats.update(reference=str(args.reference.resolve()), reference_kind='same_method_MPI' if args.mpi_check else 'same_mesh_accuracy',
                         accuracy=[])
            if args.mpi_check:
                stats['mpi_acceptance_policy'] = dict(receiver_E_relative_L2=1e-4, receiver_H_relative_L2=1e-4,
                    physical_receiver_cells='identical', full_field_M_L2='diagnostic_only')
        actions = [mor._Action(ops[key]) for key in ('K', 'M')]
        q_local = ops['q'].getArray(readonly=True).copy()
        normq = float(ops['q'].norm())
        require(math.isfinite(normq) and normq > 0, 'Assembled source must have positive finite global norm')

        def emit(index, real, imag):
            finite = bool(np.isfinite(real.getArray(readonly=True)).all() and np.isfinite(imag.getArray(readonly=True)).all())
            require(comm.allreduce(finite, op=MPI.LAND), 'Nonfinite emitted field')
            field = real.getArray(readonly=True)+1j*imag.getArray(readonly=True)
            _, rho, _ = mor._residual(comm, *actions, q_local, field, 2*np.pi*fs[index], normq)
            data = sample_receivers(ops, real, imag, 2*np.pi*fs[index])  # collective, unique owner internally
            require(np.array_equal(data['coordinates'], inputs['rx']), 'Native receiver coordinates differ')
            stats['emitted_indices'].append(int(index))
            stats['emitted_residuals'].append(dict(index=int(index), frequency_hz=fs[index], residual=rho))
            if comm.rank == 0:
                receivers[index] = data
                print('%s %s frequency %d/%d %.8g Hz residual %.6e' %
                      (args.method, args.role, index+1, len(fs), fs[index], rho), flush=True)
            if args.save_fields or reference_rows is not None:
                store_field(model, real, imag)
            if reference_rows is not None:
                reference_field = df.Function(model.FS.M)
                with df.HDF5File(df.MPI.comm_world, str(args.reference/('field_%02d.h5' % index)), 'r') as stream:
                    stream.read(reference_field, '/E')
                rr, ri = reference_field.split(deepcopy=True)
                er, ei = model.FS.U[0].split(deepcopy=True)
                row = dict(index=index, frequency_hz=fs[index],
                           reference_accepted=reference_rows['reference_accepted'].get(index),
                           interpretation='same_method_MPI_consistency' if args.mpi_check else
                               ('qualified_reference_difference' if reference_rows['reference_accepted'].get(index) is True
                                else 'response_difference_against_unverified_reference'))
                for label, weight in (('M', model.MP.sigma_func), ('L2', df.Constant(1.))):
                    difference = df.assemble((df.inner(weight*(er-rr), er-rr)+df.inner(weight*(ei-ri), ei-ri))*df.dx)
                    denominator = df.assemble((df.inner(weight*rr, rr)+df.inner(weight*ri, ri))*df.dx)
                    row['field_E_relative_'+label] = float(np.sqrt(max(0., difference)/denominator))
                for key in ('E', 'H'):
                    ref = reference_rows[key][index]
                    denominator = np.linalg.norm(ref)
                    row['receiver_'+key+'_relative_L2'] = float(np.linalg.norm(data[key]-ref)/denominator)
                stats['accuracy'].append(row)
            if args.save_fields:
                with df.HDF5File(df.MPI.comm_world, str(args.run/('field_%02d.h5' % index)), 'w') as stream:
                    stream.write(model.FS.U[0], '/E')

        tick = time.perf_counter()
        try:
            if args.method == 'SUMOR':
                solver_stats = mor.solve_split_u(ops, fs, emit=emit, **SUMOR_OPTIONS)
            elif args.method == 'CPC':
                solver_stats = mor.solve_cpc(ops, fs, emit=emit, steps=80, max_cycles=10, tol=1e-6)
            else:
                solver_stats = full_order(ops, fs, emit, direct=args.method == 'DFOM')
        finally:
            stats['solver_wall_s'] = comm.allreduce(time.perf_counter()-tick, op=MPI.MAX)
            for action in actions:
                action.destroy()
        stats['solver_stats'] = solver_stats
        expected_direct = (sum(row['linear_solver_stats']['counts'].get('DFOM_factorizations', 0)
                           for row in solver_stats['per_frequency']) if args.method == 'DFOM' else
                           solver_stats['counts'].get('shift_factorizations', 0)+
                           solver_stats['counts'].get('mass_factorizations', 0) if args.method == 'CPC' else 0)
        stats['mumps_diagnostic_summary'] = dict(expected_direct_setups=expected_direct,
            recorded_direct_setups=len(mumps_observer.records),
            status='available' if len(mumps_observer.records) == expected_direct and
                mumps_observer.provenance['labels_match_observed_version'] and
                all(row['status'] == 'available' for row in mumps_observer.records) else 'incomplete')
        if args.method == 'DFOM':
            for key in ('reference_status', 'reference_target', 'reference_accepted_indices', 'unverified_reference_indices'):
                stats[key] = solver_stats[key]
        require(solver_stats.get('status') == 'converged', 'Native solver status: '+str(solver_stats.get('failure')))
        require(sorted(stats['emitted_indices']) == list(range(len(fs))), 'Incomplete emitted frequencies')
        threshold = 1e-6
        require(all(row['residual'] <= threshold for row in stats['emitted_residuals']), 'Emitted original-matrix residual exceeds method tolerance')
        if args.mpi_check:
            require(all(math.isfinite(row[key]) and row[key] <= 1e-4 for row in stats['accuracy']
                        for key in FIELD_METRICS[2:]),
                    'Same-method MPI receiver E/H discrepancy exceeds 1e-4')
        unchanged = rank0_call(comm, lambda: all(sha256(path) == inputs['input_sha256'][key]
                                                 for key, path in inputs['file_paths'].items()) and
            all(sha256(name) == digest for name, digest in stats['production_source_sha256'].items()))
        require(unchanged, 'Production source or geometry input changed during solve')
        stats['frozen_inputs_unchanged'] = True
        stats['status'] = 'converged'
    except Exception as caught:
        stats['failures'].append(type(caught).__name__+': '+str(caught))
        stats['traceback'] = traceback.format_exc()
        if hasattr(caught, 'stats'):
            stats['solver_stats'] = caught.stats
    finally:
        def save():
            indices = sorted(receivers)
            shape = (0, 200, 3)
            if inputs := stats.get('input_sha256'):
                np.savez_compressed(args.run/'receivers.npz',
                    freq_hz=np.array(stats['frequencies_hz'])[indices],
                    frequencies_hz=np.array(stats['frequencies_hz'])[indices],
                    indices=indices, frequency_indices=indices, rx_xyz=np.array(model.MP.rx[0]) if model is not None else np.empty((0, 3)),
                    E=np.array([receivers[i]['E'] for i in indices]) if indices else np.empty(shape, complex),
                    H=np.array([receivers[i]['H'] for i in indices]) if indices else np.empty(shape, complex),
                    mesh_sha256=inputs['mesh'], production_commit=args.production_commit,
                    model_role=args.role, conductivity_S_per_m=conductivity(args.role),
                    experiment=stats['experiment'], n_frequency=stats['n_frequency'],
                    frequency_grid_sha256=stats['frequency_grid_sha256'],
                    receiver_domain_marker=args.receiver_domain, units_E='V/m', units_H='A/m')
            write_json_rank0(args.run/'stats.json', stats, comm.rank)
        rank0_call(comm, save)
        comm.Barrier()
        stats['complete_forward_s'] = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
        rank0_call(comm, lambda: write_json_rank0(args.run/'stats.json', stats, comm.rank))
        if ops is not None:
            ops['destroy']()
        if mumps_observer is not None:
            mumps_observer.restore()
    return 0 if stats['status'] == 'converged' else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-root', type=Path, help='Input data root (default: checkout/data)')
    result.add_argument('--kernel', choices=['paper_mpi', 'current'],
                        help='Default: paper_mpi for test1, current for test2; paper_mpi is the retained P1 Kg kernel')
    result.add_argument('--check-inputs', action='store_true', help='Validate existing inputs without FEM assembly')
    result.add_argument('--rx', type=Path)
    result.add_argument('--output', '--run', dest='run', type=Path)
    result.add_argument('--method', required=True, choices=['SUPERMOR', 'RK', 'SUMOR', 'CPC', 'DFOM', 'IFOM'])
    result.add_argument('--role', choices=['sphere', 'no_sphere'], default='sphere')
    result.add_argument('--experiment', choices=['test1', 'test2'], default='test1')
    frequency = result.add_mutually_exclusive_group()
    frequency.add_argument('--freq', type=float, nargs='+')
    frequency.add_argument('--n-frequency', type=int, choices=TEST2_N_FREQUENCIES)
    result.add_argument('--receiver-domain', type=int, choices=[1], default=1)
    result.add_argument('--save-fields', action='store_true')
    result.add_argument('--reference', type=Path)
    result.add_argument('--mpi-check', action='store_true')
    return result


def parse_args(argv=None):
    result = parser()
    args = result.parse_args(argv)
    args.method = {'SUPERMOR': 'SUMOR', 'RK': 'CPC'}.get(args.method, args.method)
    if args.kernel is None:
        args.kernel = 'paper_mpi' if args.experiment == 'test1' else 'current'
    if args.experiment == 'test2' and args.kernel != 'current':
        result.error('The paper frequency-count experiment requires --kernel current')
    code_hashes = {name: sha256(ROOT/'core'/'supermor'/name)
                   for name in source_filenames(args)}
    args.production_commit = 'release-' + hashlib.sha256(
        json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()
    paths = get_case_mesh('sphere', data_root=args.data_root, require=True)
    args.mesh = paths['metadata'].parent
    if args.rx is None:
        args.rx = paths['receivers']
    try:
        resolve_frequencies(args)
    except ValueError as error:
        result.error(str(error))
    return args


if __name__ == '__main__':
    selected = parse_args()
    if selected.check_inputs:
        checked = input_metadata(selected)
        print(json.dumps(clean_json(dict(status='inputs_valid', mesh=checked['file_paths'],
            input_sha256=checked['input_sha256'], frequencies_hz=resolve_frequencies(selected),
            kernel_selection=selected.kernel, numerical_solve_performed=False)), indent=2))
    else:
        sys.exit(main(selected))
