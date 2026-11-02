#!/usr/bin/env python3
"""Refine an audited custem_L2 mesh near the fixed receiver plane, in serial."""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import shutil

import numpy as np

from validation_common import identity, sha256, write_json


NAME = 'custem_L2'
ROI_BOUNDS_M = [[900., 5100.], [-1100., 1100.], [-20., 0.]]


def receiver_markers(coordinates, cells, materials):
    """Mark background tetrahedra whose bounding box touches the receiver ROI."""
    roi = np.asarray(ROI_BOUNDS_M)
    marked = np.zeros(len(cells), dtype=bool)
    # Bound temporary vertex/bounding-box arrays on the production mesh.
    for start in range(0, len(cells), 100000):
        stop = min(start + 100000, len(cells))
        points = coordinates[cells[start:stop]]
        intersects = np.all((points.max(axis=1) >= roi[:, 0]) &
                            (points.min(axis=1) <= roi[:, 1]), axis=1)
        marked[start:stop] = intersects & (materials[start:stop] == 1)
    return marked


def inherit_materials(materials, parent_cells, child_count):
    """Use DOLFIN's cell ancestry for every child, including conformity closure."""
    parent_cells = np.asarray(parent_cells)
    if (parent_cells.ndim != 1 or len(parent_cells) != child_count or
            not np.issubdtype(parent_cells.dtype, np.integer)):
        raise ValueError('parent_cell must be a one-dimensional integer map for every child')
    if child_count == 0 or parent_cells.min() < 0 or parent_cells.max() >= len(materials):
        raise ValueError('parent_cell contains an out-of-range parent index')
    return materials[parent_cells]


def validate_parent(parent):
    parent = Path(parent)
    metadata = json.loads((parent / 'mesh.json').read_text(encoding='utf-8'))
    if metadata.get('name') != NAME or metadata.get('status') != 'audited':
        raise ValueError('parent must be an audited ' + NAME + ' mesh')
    for key, expected in identity().items():
        if metadata.get(key) != expected:
            raise ValueError('parent has incompatible ' + key)
    hashes = dict(h5=sha256(parent / '_h5' / (NAME + '.h5')),
                  parameters=sha256(parent / 'para' / (NAME + '_parameters.json')))
    if hashes != metadata.get('input_sha256'):
        raise ValueError('parent HDF5 or parameters SHA-256 differs from its audit')
    digest = hash_bytes(json.dumps(hashes, sort_keys=True).encode('ascii')).hexdigest()
    if metadata.get('mesh_hash') != digest:
        raise ValueError('parent mesh_hash differs from its audited input SHA-256')
    return metadata


def validate_paths(parent, root):
    parent = Path(parent).resolve()
    if parent.name != NAME or parent.parent.name != 'meshes':
        raise ValueError('parent path must end with meshes/' + NAME)
    parent_root = parent.parent.parent
    root = Path(root).resolve()
    if root == parent_root or root in parent_root.parents or parent_root in root.parents:
        raise ValueError('refinement root must not overlap the parent experiment root')
    target = (root / 'meshes' / NAME).resolve()
    if target == parent_root or parent_root in target.parents or target in parent_root.parents:
        raise ValueError('refinement target must not overlap the parent experiment root')
    if target.exists():
        raise FileExistsError('preserving existing refinement target: ' + str(target))
    return parent, target


def refine(parent, root, passes=1):
    if isinstance(passes, bool) or passes not in (1, 2):
        raise ValueError('passes must be 1 or 2')
    parent, target = validate_paths(parent, root)
    original = validate_parent(parent)
    parent_metadata_sha256 = sha256(parent / 'mesh.json')

    import dolfin as df
    from mesh_custem import audit

    if df.MPI.size(df.MPI.comm_world) != 1:
        raise RuntimeError('receiver mesh refinement must run in serial')
    mesh = df.Mesh(df.MPI.comm_self)
    with df.HDF5File(df.MPI.comm_self, str(parent / '_h5' / (NAME + '.h5')), 'r') as stream:
        stream.read(mesh, '/mesh', False)
        if mesh.topology().dim() != 3 or mesh.geometry().dim() != 3:
            raise ValueError('parent must be a three-dimensional tetrahedral mesh')
        materials = df.MeshFunction('size_t', mesh, 3)
        stream.read(materials, '/domains')
    target.mkdir(parents=True, exist_ok=False)
    statistics = []
    previous_algorithm = df.parameters['refinement_algorithm']
    df.parameters['refinement_algorithm'] = 'plaza'
    try:
        for index in range(passes):
            parent_count = mesh.num_cells()
            marked = receiver_markers(mesh.coordinates(), mesh.cells(), materials.array())
            if not marked.any():
                raise ValueError('receiver ROI contains no background cells to refine')
            markers = df.MeshFunction('bool', mesh, 3, False)
            markers.array()[:] = marked
            child = df.refine(mesh, markers, redistribute=False)
            parent_cells = child.data().array('parent_cell', 3)
            inherited = inherit_materials(materials.array(), parent_cells, child.num_cells())
            descendants = np.bincount(np.asarray(parent_cells, dtype=np.intp), minlength=parent_count)
            if np.any(descendants == 0) or np.any(descendants[marked] <= 1):
                raise ValueError('parent_cell map omits parents or marked cells were not refined')
            child_materials = df.MeshFunction('size_t', child, 3)
            child_materials.array()[:] = inherited
            statistics.append(dict(pass_number=index + 1, parent_cells=parent_count,
                marked_cells=int(marked.sum()), child_cells=child.num_cells(),
                refined_parent_cells=int((descendants > 1).sum()),
                closure_parent_cells=int(((descendants > 1) & ~marked).sum()),
                unchanged_parent_cells=int((descendants == 1).sum()),
                child_material_cells={str(marker): int((inherited == marker).sum()) for marker in (0, 1, 2)}))
            print(json.dumps(statistics[-1], sort_keys=True), flush=True)
            mesh, materials = child, child_materials
    finally:
        df.parameters['refinement_algorithm'] = previous_algorithm

    (target / '_h5').mkdir()
    with df.HDF5File(df.MPI.comm_self, str(target / '_h5' / (NAME + '.h5')), 'w') as stream:
        stream.write(mesh, '/mesh')
        stream.write(materials, '/domains')
    (target / 'para').mkdir()
    parameters = target / 'para' / (NAME + '_parameters.json')
    shutil.copyfile(parent / 'para' / parameters.name, parameters)
    if sha256(parameters) != original['input_sha256']['parameters']:
        raise ValueError('copied parameters SHA-256 changed; refusing mesh audit')
    plan = dict(original.get('plan', {}))
    plan.pop('target_real_dofs', None)
    metadata = dict(name=NAME, level=2, status='refined', plan=plan, **identity(),
        parent_directory=str(parent), parent_mesh_hash=original['mesh_hash'],
        parent_input_sha256=original['input_sha256'], parent_metadata_sha256=parent_metadata_sha256,
        receiver_refinement=dict(algorithm='plaza', redistribute=False, passes=passes,
            roi_bounds_m=ROI_BOUNDS_M, selection='inclusive tetrahedron bounding-box intersection',
            selected_material_marker=1, maximum_real_dofs=None, pass_statistics=statistics))
    for key in ('source_directory', 'receiver_count', 'marker_conductivity_S_per_m'):
        if key in original:
            metadata[key] = original[key]
    write_json(target / 'mesh.json', metadata)
    # The audit opens its own mesh and function spaces; release refinement meshes first.
    del mesh, materials, child, child_materials, markers, parent_cells, inherited, descendants, marked
    gc.collect()
    audit(target, NAME)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path(__file__).resolve().parents[3] / 'data/mesh_data/case1_block/cross_validation/custem/parent/meshes/custem_L2')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3] / 'data/mesh_data/case1_block/cross_validation/custem/final')
    parser.add_argument('--passes', type=int, choices=(1, 2), default=1)
    parser.add_argument("--plan", action="store_true", help="Print refinement inputs and output without generating files")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(parent=str(args.parent.resolve()), output_root=str(args.root.resolve()), passes=args.passes, roi_bounds_m=ROI_BOUNDS_M), indent=2))
        return
    target = refine(args.parent, args.root, args.passes)
    print(json.dumps(dict(status='audited', directory=str(target)), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
