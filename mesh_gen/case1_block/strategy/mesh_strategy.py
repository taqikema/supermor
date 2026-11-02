#!/usr/bin/env python3
"""Generate and audit the shared B1 P1 mesh without changing custEM kernels."""
import argparse
from copy import deepcopy
from hashlib import sha256 as _sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from strategy_contract import FREQUENCIES, physical_config, receivers, refinement_anchors, sha256, write_json, source_directory


MESH_NAME = 'csem_mor'
TARGET_CELLS = 309638
TARGET_REAL_DOFS = 720744
MAX_ATTEMPTS = 1
TARGET_RELATIVE_TOLERANCE = .35
OLD_BACKGROUND_SCALE = 3.866185156447366


def mesh_plan(volume_scale=1.):
    volume_scale = float(volume_scale)
    if not np.isfinite(volume_scale) or volume_scale <= 0.:
        raise ValueError('volume_scale must be positive and finite')
    return dict(core_bounds_m=[[-7000., 7000.], [-5000., 5000.], [-3250., 3250.]],
        outer_bounds_m=physical_config()['domain_bounds_m'],
        inner_surface_halfwidth_m=[5000., 2500.], inner_surface_center_m=[500., 200.],
        refinement_envelope=dict(start=[-1500., -1800., -1500.], stop=[4000., 1800., -100.], marker=1),
        refinement=dict(local_volume=330000. * volume_scale, brick_volume=44000. * volume_scale,
                        surface_area=10000. * volume_scale ** (2. / 3.)),
        volume_scale=volume_scale, background_reference_volume_scale=OLD_BACKGROUND_SCALE,
        source_segments=20, receiver_count=3321, surface_anchors_m=refinement_anchors().tolist(),
        surface_anchor_radius_m=35., surface_anchor_segments=3,
        shells=[dict(halfwidth_m=r, max_volume_m3=(r / 8.) ** 3 * OLD_BACKGROUND_SCALE)
                for r in (20000., 100000.)],
        target_cells=TARGET_CELLS, target_cells_relative_tolerance=TARGET_RELATIVE_TOLERANCE,
        target_real_dofs=TARGET_REAL_DOFS, target_real_dofs_relative_tolerance=TARGET_RELATIVE_TOLERANCE)


def next_volume_scale(volume_scale, cells):
    if not np.isfinite(cells) or cells <= 0:
        raise ValueError('candidate tetrahedron count must be positive and finite')
    return mesh_plan(volume_scale)['volume_scale'] * float(cells) / TARGET_CELLS


def within_targets(cells, real_dofs):
    lower, upper = 1. - TARGET_RELATIVE_TOLERANCE, 1. + TARGET_RELATIVE_TOLERANCE
    return (lower * TARGET_CELLS <= cells <= upper * TARGET_CELLS
            and lower * TARGET_REAL_DOFS <= real_dofs <= upper * TARGET_REAL_DOFS)


def dimension_metadata(vertices, cells, field_dofs):
    dimensions = dict(vertices=int(vertices), cells=int(cells), field_dofs=int(field_dofs),
                      real_primary_dofs=2 * int(field_dofs), mixed_dofs=2 * int(field_dofs))
    return dict(cells=dimensions['cells'], edge_dofs=dimensions['field_dofs'],
                mixed_dofs=dimensions['mixed_dofs'], dimensions=dimensions)


def validate_geometry(bounds, markers):
    if not np.allclose(bounds, [[-100000., 100000.]] * 3, rtol=0., atol=1e-6):
        raise ValueError('actual HDF5 bounding box must be [-100000,100000] on every axis')
    if set(markers) != {0, 1, 2}:
        raise ValueError('actual HDF5 material markers must be exactly air=0/background=1/block=2')


def expected_material_marker(point):
    inside = all(low < coordinate < high for coordinate, (low, high)
                 in zip(point, physical_config()['block_bounds_m']))
    return 0 if point[2] > 0. else (2 if inside else 1)


def audit_parameters(parameters):
    if not np.array_equal(np.asarray(parameters.get('rx')), receivers()[None, :, :]):
        raise ValueError('mesh parameters do not register the full ordered receiver grid')
    tx = np.column_stack((np.linspace(-2500., -2000., 21), np.zeros(21), np.zeros(21)))
    actual_tx = np.asarray(parameters.get('tx'))
    if actual_tx.shape != (1, 21, 3) or not np.allclose(actual_tx, tx[None, :, :], rtol=0., atol=1e-8):
        raise ValueError('mesh parameters source must be the oriented 20-segment wire')
    if parameters.get('grounding') != [True]:
        raise ValueError('mesh source must be grounded')
    if set(parameters.get('domains', [])) != {0, 1, 2} or parameters.get('n_domains') != 3:
        raise ValueError('mesh parameters material markers differ')
    identity = parameters.get('strategy_contract')
    if identity is not None and (identity.get('physical_config') != physical_config()
            or identity.get('frequencies_hz') != FREQUENCIES.tolist()):
        raise ValueError('mesh parameter physical/frequency identity differs')
    return dict(source_segments=20, current_A=physical_config()['source']['current_A'],
                receiver_count=3321, source_grounded=True)


def source_identity(source):
    """Fingerprint the actual generator and installed mesh libraries.

    Historical unused examples are not a runtime dependency of this entry.
    """
    source = Path(source).resolve()
    names = ('custEM/meshgen/meshgen_tools.py',
             'custEM/meshgen/meshgen_utils.py', 'custEM/meshgen/vtk_convert.py')
    hashes = {name: sha256(source / name) for name in names}
    generator_files = (Path(__file__).resolve(), Path(__file__).with_name('strategy_contract.py').resolve())
    return dict(source_directory=str(source), source_sha256=hashes,
                mesh_generator_sha256={path.name: sha256(path) for path in generator_files},
                physical_config=physical_config(), frequencies_hz=FREQUENCIES.tolist())


def candidate_metadata(plan, identity):
    physics = physical_config()
    return dict(name=MESH_NAME, case='bricks', status='generated', plan=plan,
        outer_km=100., source_start=physics['source']['start_m'],
        source_stop=physics['source']['end_m'], current_A=1., receiver_count=3321,
        receiver_xyz=receivers().tolist(), internal_mesh_frozen=False,
        marker_conductivity_S_per_m={'0': 1e-8, '1': .01, '2': 1.}, **identity)


def generate_candidate(directory, source, plan):
    """The original B1 BlankWorld recipe, with the requested geometry fixes."""
    frozen_plan = plan
    # BlankWorld extends inner_area_shift in place; all native arguments are disposable.
    plan = deepcopy(frozen_plan)
    sys.path.insert(0, str(source))
    from custEM.meshgen import meshgen_utils as mu
    from custEM.meshgen.meshgen_tools import BlankWorld

    directory.mkdir(parents=True, exist_ok=True)
    mdir = directory / 'meshes'
    spec = plan['refinement']
    tx = mu.line_x(start=-2500., stop=-2000., n_segs=20, y=0., z=0.)
    world = BlankWorld(name=MESH_NAME,
        x_dim=plan['core_bounds_m'][0], y_dim=plan['core_bounds_m'][1], z_dim=plan['core_bounds_m'][2],
        inner_area='rectangle', inner_area_size=plan['inner_surface_halfwidth_m'],
        inner_area_shift=plan['inner_surface_center_m'], inner_area_cell_size=spec['surface_area'],
        outer_area_cell_size=250000. * OLD_BACKGROUND_SCALE ** (2. / 3.),
        airspace_cell_size=120000000. * OLD_BACKGROUND_SCALE,
        subsurface_cell_size=80000000. * OLD_BACKGROUND_SCALE, preserve_edges=True,
        m_dir=str(mdir), r_dir=str(directory / 'generation_results'), p_dir=str(mdir / 'para'),
        s_dir=str(mdir / '_mesh'), w_dir=str(mdir / 'mesh_create'))
    world.build_surface(insert_line_tx=[tx], insert_paths=mu.refine_rx(
        np.asarray(plan['surface_anchors_m']), r=35., n_segs=3))
    # add_rx only records interpolation positions; no full-grid PLC insertion.
    world.add_rx(receivers())
    world.build_halfspace_mesh()
    world.add_brick(**plan['refinement_envelope'], cell_size=spec['local_volume'])
    bounds = physical_config()['block_bounds_m']
    world.add_brick(start=[pair[0] for pair in bounds], stop=[pair[1] for pair in bounds],
                    marker=2, cell_size=spec['brick_volume'])
    half = np.asarray([7000., 5000., 3250.])
    for index, shell in enumerate(plan['shells']):
        factors = shell['halfwidth_m'] / half
        world.extend_world(x_fact=float(factors[0]), y_fact=float(factors[1]), z_fact=float(factors[2]),
            marker=[0, 1], cell_sizes=[shell['max_volume_m3']] * 2, initial=index == 0)
        half[:] = shell['halfwidth_m']
    world.call_tetgen(tet_param='-pq1.4aA', print_infos=True)
    parameters_path = mdir / 'para' / (MESH_NAME + '_parameters.json')
    parameters = json.loads(parameters_path.read_text(encoding='utf-8'))
    parameters['strategy_contract'] = source_identity(source)
    audit_parameters(parameters)
    write_json(parameters_path, parameters)
    write_json(directory / 'mesh.json', candidate_metadata(frozen_plan, source_identity(source)))


def input_hashes(directory):
    return dict(h5=sha256(directory / 'meshes' / '_h5' / (MESH_NAME + '.h5')),
                parameters=sha256(directory / 'meshes' / 'para' / (MESH_NAME + '_parameters.json')))


def mesh_hash(hashes):
    return _sha256(json.dumps(hashes, sort_keys=True).encode('ascii')).hexdigest()


def convert_and_audit(directory, source):
    sys.path.insert(0, str(source))
    import dolfin as df
    from custEM.meshgen import vtk2xml2h5

    mdir = directory / 'meshes'
    vtk2xml2h5(MESH_NAME, str(mdir), None, None)
    mesh = df.Mesh(df.MPI.comm_self)
    with df.HDF5File(df.MPI.comm_self, str(mdir / '_h5' / (MESH_NAME + '.h5')), 'r') as stream:
        stream.read(mesh, '/mesh', False)
        markers = df.MeshFunction('size_t', mesh, 3)
        stream.read(markers, '/domains')
    xyz = mesh.coordinates()
    bounds = np.column_stack((xyz.min(axis=0), xyz.max(axis=0))).tolist()
    validate_geometry(bounds, markers.array())
    counts = {str(marker): 0 for marker in (0, 1, 2)}
    volumes = []
    for cell in df.cells(mesh):
        midpoint = cell.midpoint()
        expected = expected_material_marker((midpoint.x(), midpoint.y(), midpoint.z()))
        if markers[cell] != expected:
            raise ValueError('cell %d has material %d, expected %d' % (cell.index(), markers[cell], expected))
        counts[str(expected)] += 1
        volumes.append(cell.volume())
    if not volumes or not np.isfinite(volumes).all() or min(volumes) <= 0.:
        raise ValueError('tetrahedron volumes must be finite and positive')
    tree = mesh.bounding_box_tree()
    receiver_markers = []
    for point in receivers():
        cell_id = int(tree.compute_first_entity_collision(df.Point(*point)))
        if cell_id >= mesh.num_cells():
            raise ValueError('receiver outside the actual mesh: ' + repr(point.tolist()))
        receiver_markers.append(int(markers.array()[cell_id]))
    if set(receiver_markers) != {1}:
        raise ValueError('all receivers at z=-1 must belong to background marker 1')
    field_dofs = int(df.FunctionSpace(mesh, 'N1curl', 1).dim())
    parameters = json.loads((mdir / 'para' / (MESH_NAME + '_parameters.json')).read_text(encoding='utf-8'))
    parameter_audit = audit_parameters(parameters)
    hashes = input_hashes(directory)
    metadata_path = directory / 'mesh.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata.update(status='audited', actual_bounds_m=bounds,
        **dimension_metadata(mesh.num_vertices(), mesh.num_cells(), field_dofs),
        marker_cells=counts, min_cell_volume_m3=min(volumes), max_cell_volume_m3=max(volumes),
        receiver_material_marker=1, parameter_audit=parameter_audit, input_sha256=hashes,
        mesh_hash=mesh_hash(hashes), within_targets=within_targets(mesh.num_cells(), 2 * field_dofs))
    write_json(metadata_path, metadata)


def verify_frozen(directory, source, plan=None):
    metadata = json.loads((directory / 'mesh.json').read_text(encoding='utf-8'))
    current = source_identity(source)
    for key in ('physical_config', 'frequencies_hz'):
        if metadata.get(key) != current[key]:
            raise ValueError('mesh checkpoint changed identity: ' + key)
    # A copied frozen artifact retains its historical source path and the
    # unused example fingerprint. Compare the libraries actually used here.
    recorded = metadata.get('source_sha256', {})
    if any(recorded.get(name) != digest for name, digest in current['source_sha256'].items()):
        raise ValueError('mesh checkpoint changed mesh-library identity')
    if ('mesh_generator_sha256' in metadata and
            metadata['mesh_generator_sha256'] != current['mesh_generator_sha256']):
        raise ValueError('mesh checkpoint changed generator identity')
    if plan is None:
        plan = mesh_plan(metadata.get('plan', {}).get('volume_scale', 1.))
    if metadata.get('plan') != plan:
        raise ValueError('mesh checkpoint uses a different refinement plan')
    if metadata.get('status') == 'audited':
        hashes = input_hashes(directory)
        if metadata.get('input_sha256') != hashes or metadata.get('mesh_hash') != mesh_hash(hashes):
            raise ValueError('frozen mesh HDF5/parameters changed')
    return metadata


def run_stage(stage, directory, source, scale):
    python = os.environ.get('CUSTEM_MESH_PYTHON' if stage == 'generate' else 'CUSTEM_FENICS_PYTHON', sys.executable)
    subprocess.run([python, str(Path(__file__).resolve()), '--root', str(directory), '--source', str(source),
                    '--mode', stage, '--volume-scale', str(scale)], check=True)


def build(root, source):
    destination = root / 'mesh'
    if destination.exists():
        metadata = verify_frozen(destination, source)
        if metadata.get('status') != 'audited' or not metadata.get('within_targets'):
            raise ValueError('existing shared mesh is not an accepted audited candidate')
        return metadata
    attempts, scale = [], 1.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        directory = root / 'mesh_candidates' / ('attempt_%02d' % attempt)
        plan = mesh_plan(scale)
        checkpoint = directory / 'mesh.json'
        if checkpoint.exists():
            metadata = verify_frozen(directory, source, plan)
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError('preserving incomplete mesh candidate without metadata: ' + str(directory))
            run_stage('generate', directory, source, scale)
            metadata = verify_frozen(directory, source, plan)
        if metadata.get('status') == 'generated':
            run_stage('convert', directory, source, scale)
            metadata = verify_frozen(directory, source, plan)
        if metadata.get('status') != 'audited':
            raise ValueError('candidate audit did not complete')
        dimensions = metadata['dimensions']
        accepted = within_targets(dimensions['cells'], dimensions['real_primary_dofs'])
        attempts.append(dict(attempt=attempt, volume_scale=scale, dimensions=dimensions,
                             accepted=accepted, mesh_hash=metadata['mesh_hash']))
        write_json(root / 'mesh_search.json', dict(max_attempts=MAX_ATTEMPTS, attempts=attempts))
        if accepted:
            metadata.update(accepted_attempt=attempt, attempts=attempts, shared_mesh_frozen=True)
            write_json(checkpoint, metadata)
            directory.rename(destination)
            verify_frozen(destination, source, plan)
            return metadata
    raise RuntimeError('mesh outside tetrahedron/real-primary-DOF targets after 1 attempt; candidate preserved without retuning')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / 'data/mesh_data/case1_block/strategy')
    parser.add_argument('--source', type=Path, help='Optional custEM source root; otherwise use the installed package')
    parser.add_argument('--mode', choices=('build', 'generate', 'convert'), default='build')
    parser.add_argument('--volume-scale', type=float, default=1.)
    parser.add_argument("--plan", action="store_true", help="Print the formal mesh plan without generating files")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(output_root=str(args.root.resolve()), plan=mesh_plan(args.volume_scale)), indent=2))
        return
    if int(os.environ.get('OMPI_COMM_WORLD_SIZE', os.environ.get('PMI_SIZE', '1'))) != 1:
        raise RuntimeError('mesh generation/conversion must run in one rank')
    root = args.root.resolve()
    source = args.source.resolve() if args.source is not None else source_directory()
    plan = mesh_plan(args.volume_scale)
    source_identity(source)
    if args.mode == 'build':
        build(root, source)
    elif args.mode == 'generate':
        generate_candidate(root, source, plan)
    else:
        convert_and_audit(root, source)


if __name__ == '__main__':
    main()
