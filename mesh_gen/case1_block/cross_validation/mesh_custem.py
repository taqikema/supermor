#!/usr/bin/env python3
"""Serial pyGIMLi meshing and separate FEniCS HDF5 conversion/audit."""
from __future__ import annotations

import argparse
from hashlib import sha256 as hash_bytes
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from validation_common import identity, mesh_spec, physical_config, receivers, sha256, write_json


def mesh_plan(level, volume_scale=1.):
    volume_scale = float(volume_scale)
    if not np.isfinite(volume_scale) or volume_scale <= 0.:
        raise ValueError('volume_scale must be positive and finite')
    spec = dict(mesh_spec("custem", level))
    for key in ('local_volume', 'brick_volume', 'shallow_volume'):
        spec[key] *= volume_scale
    spec['surface_area'] *= volume_scale ** (2. / 3.)
    spec['source_segments'] *= 2
    plane_anchors = [[float(x), float(y), 0.] for y in (-1000., -500., 0., 500., 1000.)
                     for x in np.arange(1000., 5001., 200.)]
    corridor_anchors = [[float(x), float(y), 0.] for y in (-500., 0., 500.)
                        for x in np.r_[np.arange(-2500., 1300., 250.), 1300.]]
    source_anchors = [[float(x), float(y), 0.] for y in (-250., 0., 250.)
                      for x in (-2750., -2500., -2250., -2000., -1750.)]
    return dict(
        core_bounds_m=[[-7500., 7500.], [-5000., 5000.], [-4000., 4000.]],
        outer_bounds_m=[[-100000., 100000.]] * 3,
        shallow_bounds_m=[[-3000., 5500.], [-1500., 1500.], [-100., 0.]],
        envelope_bounds_m=[[-3000., 5500.], [-1500., 1500.], [-1200., -100.]],
        inner_area_halfwidth_m=[5000., 2500.], inner_area_shift_m=[1000., 0.],
        refinement=spec, volume_scale=volume_scale,
        volume_scale_scope='local refinement volumes and inner surface area',
        target_real_dofs=[7200000, 8800000],
        surface_anchor_radius_m=25., surface_anchor_segments=3,
        surface_anchors_m=np.unique(plane_anchors + corridor_anchors + source_anchors, axis=0).tolist())


def recovery_action(metadata, plan, vtk_exists, actual_hashes):
    """Choose a frozen checkpoint action without changing any mesh files."""
    for key, expected in identity().items():
        if metadata.get(key) != expected:
            raise ValueError("mesh recovery rejected: changed " + key)
    if metadata.get("plan") != plan:
        raise ValueError("mesh recovery rejected: mesh plan differs")
    if metadata.get("status") == "generated":
        if not vtk_exists:
            raise ValueError("mesh recovery rejected: generated checkpoint has no original VTK")
        return "convert"
    if metadata.get("status") == "audited":
        if actual_hashes is None or actual_hashes != metadata.get("input_sha256"):
            raise ValueError("mesh recovery rejected: audited HDF5 or parameters missing/changed")
        digest = hash_bytes(json.dumps(actual_hashes, sort_keys=True).encode("ascii")).hexdigest()
        if metadata.get("mesh_hash") != digest:
            raise ValueError("mesh recovery rejected: audited mesh_hash differs")
        return "complete"
    raise ValueError("mesh recovery rejected: unsupported checkpoint status")


def checkpoint_action(directory, name, level, volume_scale=1.):
    metadata = json.loads((directory / "mesh.json").read_text(encoding="utf-8"))
    hashes = None
    if metadata.get("status") == "audited":
        h5 = directory / "_h5" / (name + ".h5")
        parameters = directory / "para" / (name + "_parameters.json")
        if h5.is_file() and parameters.is_file():
            hashes = dict(h5=sha256(h5), parameters=sha256(parameters))
    return recovery_action(metadata, mesh_plan(level, volume_scale),
                           (directory / "_vtk" / (name + ".vtk")).is_file(), hashes)


def run_conversion(root, level, volume_scale=1.):
    command = [os.environ.get("CUSTEM_FENICS_PYTHON", sys.executable), str(Path(__file__).resolve()),
               "--root", str(root), "--level", str(level), "--mode", "convert",
               "--volume-scale", str(volume_scale)]
    subprocess.run(command, check=True)


def partition_rectangles(rectangles):
    """Disjoint planar pieces covering a union of axis-aligned rectangles."""
    xs = sorted({x for rectangle in rectangles for x in rectangle[:2]})
    ys = sorted({y for rectangle in rectangles for y in rectangle[2:]})
    return [(x0, x1, y0, y1) for x0, x1 in zip(xs[:-1], xs[1:])
            for y0, y1 in zip(ys[:-1], ys[1:])
            if any(a <= (x0+x1)/2 <= b and c <= (y0+y1)/2 <= d
                   for a, b, c, d in rectangles)]


def split_facet_edges(polygons, coordinates):
    """Use identical segments where the z=-100 partition meets side facets."""
    candidates = np.flatnonzero(np.isclose(coordinates[:, 2], -100., rtol=0., atol=1e-8))
    points = coordinates[candidates]
    result = []
    for polygon in polygons:
        nodes = []
        for start, stop in zip(polygon, polygon[1:] + polygon[:1]):
            nodes.append(start)
            direction = coordinates[stop] - coordinates[start]
            squared = float(direction @ direction)
            if squared == 0.:
                continue
            fractions = ((points - coordinates[start]) @ direction) / squared
            distance = np.linalg.norm(points - coordinates[start] - fractions[:, None]*direction, axis=1)
            interior = np.flatnonzero((fractions > 1e-10) & (fractions < 1.-1e-10) & (distance < 1e-7))
            nodes.extend(candidates[interior[np.argsort(fractions[interior])]].tolist())
        result.append(nodes)
    return result


def validate_geometry(bounds, markers):
    if not np.allclose(bounds, [[-100000., 100000.]]*3, rtol=0., atol=1e-6):
        raise ValueError("actual HDF5 bounding box must be [-100000,100000] on every axis")
    if set(markers) != {0, 1, 2}:
        raise ValueError("actual HDF5 material markers must be exactly air=0/background=1/block=2")


def expected_material_marker(point, physics):
    bounds = physics['block_bounds_m']
    in_block = all(lower < coordinate < upper for coordinate, (lower, upper) in zip(point, bounds))
    return 0 if point[2] > 0. else (2 if in_block else 1)


def audit(directory, name):
    import dolfin as df
    h5 = directory / "_h5" / (name + ".h5")
    mesh = df.Mesh(df.MPI.comm_self)
    with df.HDF5File(df.MPI.comm_self, str(h5), "r") as stream:
        stream.read(mesh, "/mesh", False)
        markers = df.MeshFunction("size_t", mesh, 3)
        stream.read(markers, "/domains")
    coordinates = mesh.coordinates()
    bounds = np.column_stack((coordinates.min(axis=0), coordinates.max(axis=0))).tolist()
    validate_geometry(bounds, set(markers.array().tolist()))
    physics = physical_config()
    volumes = []
    marker_cells = {str(marker): 0 for marker in (0, 1, 2)}
    for cell in df.cells(mesh):
        midpoint = cell.midpoint()
        x, y, z = midpoint.x(), midpoint.y(), midpoint.z()
        expected = expected_material_marker((x, y, z), physics)
        if markers[cell] != expected:
            raise ValueError("material mismatch at cell %d; expected %d, actual %d" % (cell.index(), expected, markers[cell]))
        marker_cells[str(expected)] += 1
        volumes.append(cell.volume())
    if not np.isfinite(volumes).all() or min(volumes) <= 0.:
        raise ValueError("invalid tetrahedron volume")
    V = df.FunctionSpace(mesh, "N1curl", 1)
    S = df.FunctionSpace(mesh, "CG", 1)
    element = df.FiniteElement("N1curl", mesh.ufl_cell(), 1)
    mixed = df.FunctionSpace(mesh, df.MixedElement([element, element]))
    dimensions = dict(vertices=mesh.num_vertices(), cells=mesh.num_cells(),
                      field_dofs=V.dim(), mixed_dofs=mixed.dim(), scalar_dofs=S.dim())
    parameters = directory / "para" / (name + "_parameters.json")
    input_hashes = dict(h5=sha256(h5), parameters=sha256(parameters))
    mesh_hash = hash_bytes(json.dumps(input_hashes, sort_keys=True).encode("ascii")).hexdigest()
    metadata_path = directory / "mesh.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(status="audited", actual_bounds_m=bounds, dimensions=dimensions,
                    marker_cells=marker_cells, min_cell_volume_m3=min(volumes),
                    max_cell_volume_m3=max(volumes), input_sha256=input_hashes, mesh_hash=mesh_hash)
    target = metadata.get('plan', {}).get('target_real_dofs')
    if target is not None:
        metadata['within_target_real_dofs'] = target[0] <= mixed.dim() <= target[1]
    write_json(metadata_path, metadata)


def convert(root, level, volume_scale=1.):
    name = "custem_L%d" % level
    directory = root / "meshes" / name
    if checkpoint_action(directory, name, level, volume_scale) == "complete":
        return
    from validation_common import source_directory
    sys.path.insert(0, str(source_directory(root)))
    from custEM.meshgen import vtk2xml2h5
    vtk2xml2h5(name, str(directory), None, None)
    audit(directory, name)


def smoke_mesh(root):
    """Small conforming physical model for collective MPI 1/2/16 checks."""
    import dolfin as df
    name = "custem_smoke"
    directory = root / "meshes" / name
    if (directory / "mesh.json").exists():
        raise FileExistsError("preserving existing smoke mesh: " + str(directory))
    axes = [np.asarray([-100000., -7500., -3000., -2500., -2250., -2000., 1000., 1300., 3500., 5000., 5500., 7500., 100000.]),
            np.asarray([-100000., -5000., -1500., -1000., -500., 0., 400., 500., 1000., 1500., 5000., 100000.]),
            np.asarray([-100000., -4000., -1500., -700., -300., -100., -1., -.1, 0., 4000., 100000.])]
    n = [len(axis)-1 for axis in axes]
    mesh = df.UnitCubeMesh(df.MPI.comm_self, *n)
    for axis, points in enumerate(axes):
        index = np.rint(mesh.coordinates()[:, axis]*n[axis]).astype(int)
        mesh.coordinates()[:, axis] = points[index]
    markers = df.MeshFunction("size_t", mesh, 3, 0)
    physics = physical_config()
    for cell in df.cells(mesh):
        p = cell.midpoint()
        markers[cell] = expected_material_marker((p.x(), p.y(), p.z()), physics)
    h5 = directory / "_h5" / (name + ".h5")
    h5.parent.mkdir(parents=True, exist_ok=True)
    with df.HDF5File(df.MPI.comm_self, str(h5), "w") as stream:
        stream.write(mesh, "/mesh")
        stream.write(markers, "/domains")
    parameters = dict(topo=None, centering=False, easting_shift=None, northing_shift=None,
        rotation=None, n_layers=1, layer_depths=[], zeroing=False, bathymetry=False,
        rx=[receivers(True).tolist()], tx=[[[-2500., 0., 0.], [-2250., 0., 0.], [-2000., 0., 0.]]],
        grounding=[True], domains=[0, 1, 2], n_domains=3, post_rotate=None, overwrite_markers=None)
    write_json(directory / "para" / (name + "_parameters.json"), parameters)
    write_json(directory / "mesh.json", dict(name=name, smoke=True, status="generated", **identity()))
    audit(directory, name)


def replace_shared_refinement_faces(world, rectangles):
    # The shallow box bottom and envelope top share z=-100. Replace their
    # overlapping faces with a planar partition before handing the PLC to TetGen.
    positions = world.Omega.positions().array()
    remove = [i for i, polygon in enumerate(world.Omega.n_polys)
              if np.allclose(positions[polygon, 2], -100., rtol=0., atol=1e-8)]
    if len(remove) != 2:
        raise RuntimeError("expected shallow bottom and envelope top at z=-100; found %d" % len(remove))
    for index in reversed(remove):
        del world.Omega.n_polys[index]
        del world.poly_marker[index]
    pieces = partition_rectangles(rectangles)
    for x0, x1, y0, y1 in pieces:
        points = [(x0, y0, -100.), (x1, y0, -100.), (x1, y1, -100.), (x0, y1, -100.)]
        world.Omega.n_polys.append([world.Omega.createNodeWithCheck(point).id() for point in points])
        world.poly_marker.append(23)
    world.Omega.n_polys = split_facet_edges(world.Omega.n_polys, world.Omega.positions().array())


def build(root, level, volume_scale=1.):
    name = "custem_L%d" % level
    directory = root / "meshes" / name
    if (directory / "mesh.json").exists():
        if checkpoint_action(directory, name, level, volume_scale) == "convert":
            run_conversion(root, level, volume_scale)
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("mesh recovery rejected: preserving nonempty directory without mesh.json: " + str(directory))
    from validation_common import source_directory
    source = source_directory(root)
    sys.path.insert(0, str(source))
    from custEM.meshgen import meshgen_utils as mu
    from custEM.meshgen.meshgen_tools import BlankWorld

    directory.mkdir(parents=True, exist_ok=True)
    plan = mesh_plan(level, volume_scale)
    spec = plan["refinement"]
    physics = physical_config()
    source_spec = physics['source']
    old_scale = 3.866185156447366
    tx = mu.line_x(start=source_spec['start_m'][0], stop=source_spec['end_m'][0],
                   n_segs=spec["source_segments"], y=source_spec['start_m'][1], z=source_spec['start_m'][2])
    world = BlankWorld(name=name,
        x_dim=plan["core_bounds_m"][0], y_dim=plan["core_bounds_m"][1], z_dim=plan["core_bounds_m"][2],
        inner_area="rectangle", inner_area_size=plan['inner_area_halfwidth_m'],
        inner_area_shift=list(plan['inner_area_shift_m']),
        inner_area_cell_size=spec["surface_area"],
        outer_area_cell_size=250000.*old_scale**(2./3.),
        airspace_cell_size=120000000.*old_scale,
        subsurface_cell_size=80000000.*old_scale, preserve_edges=True,
        m_dir=str(directory), r_dir=str(directory / "generation_results"), p_dir=str(directory / "para"),
        s_dir=str(directory / "_mesh"), w_dir=str(directory / "mesh_create"))
    world.build_surface(insert_line_tx=[tx], insert_paths=mu.refine_rx(
        np.asarray(plan["surface_anchors_m"]), r=plan['surface_anchor_radius_m'],
        n_segs=plan['surface_anchor_segments']))
    # A surface-intersecting refinement region must enter through build_surface,
    # so its top shares the existing air/earth interface instead of duplicating it.
    sb, eb = plan['shallow_bounds_m'], plan['envelope_bounds_m']
    shallow = np.asarray([[x, y, sb[2][1]] for x, y in
                           [(sb[0][0], sb[1][0]), (sb[0][1], sb[1][0]),
                            (sb[0][1], sb[1][1]), (sb[0][0], sb[1][1])]])
    world.add_surface_anomaly(insert_paths=[shallow], depths=[sb[2][0]],
        cell_sizes=[spec["shallow_volume"]], marker=[1], marker_positions=[[0., 0., -50.]])
    rx = receivers(extra_depth=True)
    world.add_rx(rx)
    world.build_halfspace_mesh()
    # Both refinement volumes carry background marker 1; only the brick is 2.
    world.add_brick(start=[bounds[0] for bounds in eb], stop=[bounds[1] for bounds in eb],
                    cell_size=spec["local_volume"], marker=1)
    replace_shared_refinement_faces(world, [tuple(sb[0] + sb[1]), tuple(eb[0] + eb[1])])
    block = physics['block_bounds_m']
    world.add_brick(start=[bounds[0] for bounds in block], stop=[bounds[1] for bounds in block],
                    cell_size=spec["brick_volume"], marker=2)
    half = np.array([7500., 5000., 4000.])
    shells = []
    for index, radius in enumerate([20000., 100000.]):
        factors = radius / half
        volume = (radius/8.)**3 * old_scale
        world.extend_world(x_fact=float(factors[0]), y_fact=float(factors[1]),
            z_fact=float(factors[2]), marker=[0, 1], cell_sizes=[volume, volume], initial=index == 0)
        half[:] = radius
        shells.append(dict(halfwidth_m=radius, max_volume_m3=volume))
    world.call_tetgen(tet_param="-pq1.4aA", print_infos=True)
    metadata = dict(name=name, level=level, status="generated", plan=plan, shells=shells,
                    **identity(), source_directory=str(source), receiver_count=len(rx),
                    marker_conductivity_S_per_m={"0": 1e-8, "1": .01, "2": 1.})
    write_json(directory / "mesh.json", metadata)
    run_conversion(root, level, volume_scale)


def main():
    from validation_common import nonnegative_level
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3] / "data/mesh_data/case1_block/cross_validation/custem/parent")
    parser.add_argument("--level", type=nonnegative_level, default=2)
    parser.add_argument('--volume-scale', type=float, default=1.)
    parser.add_argument("--mode", choices=("build", "convert", "smoke"), default="build")
    parser.add_argument("--plan", action="store_true", help="Print the formal mesh plan without generating files")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(output_root=str(args.root.resolve()), level=args.level, plan=mesh_plan(args.level, args.volume_scale)), indent=2))
        return
    ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
    if ranks != 1:
        raise RuntimeError("mesh generation/conversion must be serial on a compute node")
    root = args.root.resolve()
    mesh_plan(args.level, args.volume_scale)
    if args.mode == "build":
        build(root, args.level, args.volume_scale)
    elif args.mode == "convert":
        convert(root, args.level, args.volume_scale)
    else:
        smoke_mesh(root)


if __name__ == "__main__":
    main()
