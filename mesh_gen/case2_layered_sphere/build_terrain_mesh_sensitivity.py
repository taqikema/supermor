#!/usr/bin/env python3
"""Generate/audit meshes for local-refinement and outer-box sensitivity.

Run serially in cust_env. Every attempt is retained. Formal eligibility is
only a mesh-size/geometric result, not sensitivity/MPI/physics acceptance.
"""
import argparse
import csv
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "mesh_data" / "case2_layered_sphere"
TEMPLATE_PATH = DATA_ROOT / "inputs" / "sphere_surface_template.npz"

NAME = "terrain_sphere"
CENTER = np.array([0., 0., -650.])
RADIUS = 250.
SIGMA = [1e-8, .01, .001, .02, 1.]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def terrain(x, y):
    """Original central relief; zero value/first derivative at its box edge."""
    x, y = np.asarray(x), np.asarray(y)
    inside = (np.abs(x) <= 3000.) & (np.abs(y) <= 2500.)
    return np.where(inside, np.cos(np.pi*x/6000.)**2*np.cos(np.pi*y/5000.)**2 *
                    (.7*np.sin(2*np.pi*x/3200.)+.3*np.cos(2*np.pi*y/2800.)), 0.)


def ground(x, y):
    return 250.*terrain(x, y)


def second_top(x, y):
    return -300.+90.*terrain(x, y)


def third_top(x, y):
    return -1000.+40.*terrain(x, y)


def sphere_surface():
    """Exact original display-mesh boundary; no runtime VTK dependency."""
    with np.load(TEMPLATE_PATH) as template:
        points, faces = template["points"], template["faces"]
    require(points.shape == (642, 3) and faces.shape == (1280, 3), "Invalid sphere template")
    require(np.max(np.abs(np.linalg.norm(points-CENTER, axis=1)-RADIUS)) < .002,
            "Sphere template differs from the original 250 m sphere")
    return points, faces


def receiver_xy():
    return np.array([[x,y] for x in -225.+50.*np.arange(10) for y in -475.+50.*np.arange(20)])


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def generate(folder, outer_m, scale, mode, quality, local_volume_factor=1.):
    from custEM.meshgen.meshgen_tools import BlankWorld
    from custEM.meshgen import meshgen_utils as mu
    # A single halfspace extension would terminate the deeper layer interfaces.
    # Instead all three interfaces span the full box, with compact central relief.
    class LayerWorld(BlankWorld):
        def add_marker(self, label, pos, marker=None, cell_size=0.):
            # Stock seeds use box corners + a_tol. Keep them well inside each
            # region, independently of coordinate export precision.
            far = -.5*outer_m
            positions = {"airspace ": [far, far, .5*outer_m],
                         "layer #0 ": [far, far, -150.],
                         "layer #1 ": [far, far, -650.],
                         "subsurface ": [far, far, -.5*(outer_m+1000.)]}
            return super().add_marker(label, positions.get(label, pos), marker, cell_size)

        def add_interface(self, frame=None, topo=None, id_offset=None, *args, **kwargs):
            if frame is None:
                return super().add_interface(frame, topo, id_offset, *args, **kwargs)
            # BlankWorld's normal inner-area logic targets self.surface only;
            # add the same local area explicitly to each subsurface PLC frame.
            corners = [(-3300.,-2800.,0.),(3300.,-2800.,0.),(3300.,2800.,0.),(-3300.,2800.,0.)]
            nodes = [frame.createNodeWithCheck(p) for p in corners]
            for a,b in zip(nodes, nodes[1:]+nodes[:1]):
                frame.createEdge(a,b)
            self.add_area_marker(frame, [0.,0.,0.], self.local_interface_area, marker=556)
            previous = self.inner_area
            self.inner_area = None
            try:
                return super().add_interface(frame, topo, id_offset, *args, **kwargs)
            finally:
                self.inner_area = previous

    require(shutil.which("tetgen"), "TetGen executable not found in PATH")
    folder.mkdir(parents=True, exist_ok=False)
    meshes = folder/"meshes"
    fine_area = ((20000. if mode == "pilot" else 2000.)*scale**(2./3.) *
                 local_volume_factor**(2./3.))
    settings = dict(outer_halfwidth_m=outer_m, volume_scale=scale, quality=quality,
                    outer_fullwidth_m=2*outer_m, local_volume_factor=local_volume_factor,
                    surface_local_area_m2=fine_area, interface_local_area_m2=2*fine_area,
                    far_surface_area_m2=2.5e7*scale**(2./3.),
                    sphere_max_volume_m3=1.5e5*scale*local_volume_factor,
                    far_region_max_volume_m3=[1e12*scale,1e10*scale,1e10*scale,1e12*scale],
                    sphere_surface_template_sha256=sha(TEMPLATE_PATH),
                    sphere_polyhedron_volume_m3=64890089.549438335,
                    poly_coordinate_format=".17g")
    write(folder/"generation_settings.json", settings)
    world = LayerWorld(name=NAME, x_dim=[-outer_m,outer_m], y_dim=[-outer_m,outer_m],
        z_dim=[-outer_m,outer_m], inner_area="rectangle", inner_area_size=[3300.,2800.],
        inner_area_cell_size=fine_area, outer_area_cell_size=settings["far_surface_area_m2"],
        interface_cell_sizes=[settings["far_surface_area_m2"]]*2,
        layer_cell_sizes=settings["far_region_max_volume_m3"][1:3],
        airspace_cell_size=settings["far_region_max_volume_m3"][0],
        subsurface_cell_size=settings["far_region_max_volume_m3"][3],
        preserve_edges=True, topo=ground, subsurface_topos=[second_top,third_top],
        a_tol=1e-8, backup_script=False, export_omega=False,
        m_dir=str(meshes), r_dir=str(folder/"unused_results"), p_dir=str(meshes/"para"),
        s_dir=str(meshes/"_mesh"), w_dir=str(folder/"mesh_create"))
    world.local_interface_area = settings["interface_local_area_m2"]
    xy = receiver_xy()
    rx_seed = np.column_stack((xy, np.zeros(len(xy))))
    tx_seed = mu.line_x(start=-2250., stop=-1750., n_segs=20, y=0., z=0.)
    world.build_surface(insert_line_tx=[tx_seed], insert_points=rx_seed.tolist(),
                        insert_paths=mu.refine_rx(rx_seed, r=15., n_segs=3))
    world.build_layered_earth_mesh(n_layers=3, layer_depths=[-300.,-1000.])
    points, faces = sphere_surface()
    nodes = [world.Omega.createNodeWithCheck(p.tolist()) for p in points]
    for face in faces:
        world.Omega.createTriangleFace(*(nodes[int(i)] for i in face), marker=23)
    world.add_marker("sphere", CENTER, cell_size=settings["sphere_max_volume_m3"])
    # Export through the established wrapper, but run TetGen with checked status.
    # The wrapper's os.system status is not reliable; absence/nonzero is fatal.
    world.eval_unique_markers()
    world.write_mesh_parameters(ignore_tx_z=True)
    poly = folder/"mesh_create"/(NAME+".poly")
    mu.export_tetgen_poly_file(world.Omega, str(poly), poly_marker=world.poly_marker,
                              float_format=".17g")
    mu.export_tetgen_edge_file(world.preserve_edges, world.edge_marker,
                              str(poly.with_suffix(".edge")))
    command = ["tetgen", "-pq%gaAkV" % quality, str(poly)]
    with (folder/"tetgen.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0, "TetGen failed; inspect retained tetgen.log")
    generated = poly.with_name(NAME+".1.vtk")
    require(generated.exists() and generated.stat().st_size > 0, "TetGen did not produce a new VTK mesh")
    shutil.copy2(generated, meshes/"_vtk"/(NAME+".vtk"))
    return settings


def run_stage(stage, folder, outer_m):
    """FEniCS must not share a process with preceding PLC/Triangle operations."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--output", str(folder.resolve()), "--outer-km", str(outer_m/1000.),
               "--internal-stage", stage]
    log = folder/(stage+".log")
    print(json.dumps(dict(stage=stage, status="starting", log=str(log))), flush=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    with log.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    require(completed.returncode == 0,
            "%s stage failed (returncode %s); inspect %s" % (stage, completed.returncode, log))
    print(json.dumps(dict(stage=stage, status="completed", log=str(log))), flush=True)


def internal_stage(stage, folder, outer_m):
    faulthandler.enable(all_threads=True)
    print(json.dumps(dict(stage=stage, status="started", pid=os.getpid())), flush=True)
    if stage == "convert":
        from custEM.meshgen import vtk2xml2h5
        print("Converter imported; reading retained VTK", flush=True)
        vtk2xml2h5(NAME, str(folder/"meshes"), None, None)
        print("VTK/XML/HDF5 conversion completed", flush=True)
    elif stage == "audit":
        result = audit(folder, outer_m)
        print(json.dumps(dict(cells=result["cells"], edges=result["edges"],
                              field_dofs=result["field_dofs"])), flush=True)
    print(json.dumps(dict(stage=stage, status="completed", pid=os.getpid())), flush=True)


def audit(folder, outer_m=None):
    import dolfin as df
    from scipy.spatial import cKDTree
    require(df.MPI.size(df.MPI.comm_world) == 1, "Mesh audit is serial")
    mesh = df.Mesh()
    h5path = folder/"meshes/_h5"/(NAME+".h5")
    with df.HDF5File(df.MPI.comm_world, str(h5path), "r") as h5:
        h5.read(mesh, "/mesh", False)
        domains = df.MeshFunction("size_t", mesh, 3)
        h5.read(domains, "/domains")
    mesh.order()
    mesh.init(1); mesh.init(2,3); mesh.init(2,0); mesh.init(0,3)
    xyz, cells, tags = mesh.coordinates(), mesh.cells(), domains.array()
    require(set(np.unique(tags)) == set(range(5)), "Expected air, three layers and sphere tags 0..4")
    volumes, scaled, signed = [], [], []
    region_volumes = np.zeros(5)
    max_sphere_excess = -np.inf
    for start in range(0,len(cells),50000):
        stop = min(start+50000,len(cells)); points = xyz[cells[start:stop]]
        determinant = np.linalg.det(points[:,1:]-points[:,:1])
        require(np.isfinite(determinant).all() and np.all(determinant != 0), "Degenerate tetrahedron")
        volume = np.abs(determinant)/6.
        longest = np.maximum.reduce([np.linalg.norm(points[:,a]-points[:,b],axis=1)
                                      for a in range(4) for b in range(a)])
        volumes.append(volume); scaled.append(6*volume/longest**3);signed.append(determinant)
        for marker in range(5):region_volumes[marker] += volume[tags[start:stop]==marker].sum()
        sphere = points[tags[start:stop]==4]
        if len(sphere):max_sphere_excess=max(max_sphere_excess,float(np.linalg.norm(sphere-CENTER,axis=2).max()-RADIUS))
    volume, shape = np.concatenate(volumes), np.concatenate(scaled)
    lower, upper = xyz.min(0), xyz.max(0)
    if outer_m is None:outer_m=float(np.max(np.abs(np.r_[lower,upper])))
    require(np.allclose(lower,-outer_m,atol=1e-5,rtol=0) and np.allclose(upper,outer_m,atol=1e-5,rtol=0), "Unexpected full-domain bounds")
    require(abs(volume.sum()/(2*outer_m)**3-1.) < 1e-8, "Volume does not fill the outer box")
    require(max_sphere_excess <= .002, "Sphere extends outside its prescribed radius")
    ground_faces, ground_earth, outer_faces, unexpected = [], [], 0, []
    allowed = {(0,1),(1,2),(2,3),(2,4)}
    for facet in df.facets(mesh):
        adjacent = facet.entities(3)
        require(len(adjacent) in (1,2), "Nonmanifold facet")
        ids = facet.entities(0)
        if len(adjacent)==1:
            points=xyz[ids]
            require(any(np.all(np.abs(points[:,axis]-bound)<1e-5) for axis in range(3) for bound in (lower[axis],upper[axis])), "Outer facet is inside domain")
            outer_faces += 1
        else:
            pair=tuple(sorted((int(tags[adjacent[0]]),int(tags[adjacent[1]]))))
            if pair[0]!=pair[1]:
                if pair not in allowed:unexpected.append(pair)
                if pair==(0,1):
                    ground_faces.append(ids.copy());ground_earth.append(int(next(c for c in adjacent if tags[c]==1)))
    require(not unexpected, "Unexpected physical-region interface pairs: "+str(set(unexpected)))
    ground_faces=np.asarray(ground_faces,dtype=np.int64)
    require(len(ground_faces)>0, "No ground interface")
    ground_vertices=np.unique(ground_faces)
    # All requested receiver XY points were constrained surface vertices.
    tree=cKDTree(xyz[ground_vertices,:2])
    distances, locations=tree.query(receiver_xy())
    require(np.all(distances<1e-7), "A receiver XY seed was not retained on the ground")
    rx_ids=ground_vertices[locations]; rx=xyz[rx_ids].copy();earth_cells=[]
    for vertex in rx_ids:
        adjacent=df.Vertex(mesh,int(vertex)).entities(3)
        candidates=[int(c) for c in adjacent if tags[c]==1]
        require(candidates, "Receiver has no first-layer cell")
        earth_cells.append(min(candidates))
    # Recover the exact FE wire after TetGen subdivision/ASCII conversion.
    source_edges=set()
    for face in ground_faces:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            points=xyz[[a,b]]
            if np.all(np.abs(points[:,1])<1e-7) and np.all(points[:,0]>=-2250.-1e-7) and np.all(points[:,0]<=-1750.+1e-7):
                source_edges.add(tuple(sorted((int(a),int(b)))))
    source_ids=sorted({node for edge in source_edges for node in edge}, key=lambda node:xyz[node,0])
    require(len(source_ids)>=21, "Source PLC constraints were lost")
    wire=xyz[source_ids].copy()
    require(np.allclose(wire[[0,-1],:2],[[-2250.,0.],[-1750.,0.]],atol=1e-7,rtol=0), "Wire endpoints changed")
    expected={tuple(sorted((a,b))) for a,b in zip(source_ids,source_ids[1:])}
    require(expected==source_edges and np.all(np.diff(wire[:,0])>0), "Wire is disconnected, branches, or contains a gap")
    path_length=float(np.linalg.norm(np.diff(wire,axis=0),axis=1).sum())
    V=df.FunctionSpace(mesh,"N1curl",1)
    edges=mesh.num_entities(1)
    require(V.dim()==edges, "P1 natural-boundary DOF count differs from global edges")
    params_path=folder/"meshes/para"/(NAME+"_parameters.json")
    params=json.loads(params_path.read_text())
    params.update(tx=[wire.tolist()],rx=[rx.tolist()],grounding=[True])
    write(params_path,params)
    write_csv(folder/"receivers.csv",["rx_id","line_id","station_id","x_m","y_m","z_m","ground_vertex_id","earth_cell_global_id"],
              [[i+1,i//20+1,i%20+1,*rx[i],int(rx_ids[i]),earth_cells[i]] for i in range(200)])
    write_csv(folder/"source.csv",["vertex_id","x_m","y_m","z_m","mesh_vertex_id"],
              [[i+1,*wire[i],int(node)] for i,node in enumerate(source_ids)])
    source_audit=dict(passed=True,source_type="grounded finite wire; FE line functional, not point dipole",current_A=1.,
        horizontal_length_m=500.,terrain_polyline_length_m=path_length,segments=len(source_edges),
        vertices=wire.tolist(),mesh_vertex_ids=source_ids,mesh_edges=[list(edge) for edge in sorted(source_edges)],
        source_on_ground_interface=True,complete_mesh_edges=True)
    write(folder/"source_edge_audit.json",source_audit)
    write(folder/"receiver_audit.json",dict(passed=True,count=200,grid_shape=[10,20],xy_seed_max_error_m=float(distances.max()),
        coordinates=rx.tolist(),ground_vertex_ids=rx_ids.tolist(),earth_cell_global_ids=earth_cells,
        sampling_policy="first-layer side; minimum global incident material-1 cell, pointwise E/curl-H"))
    audit_record=dict(passed=True,cells=int(mesh.num_cells()),vertices=int(mesh.num_vertices()),edges=int(edges),
        field_dofs=int(V.dim()),mixed_dofs=int(2*V.dim()),outer_boundary_facets=outer_faces,
        minimum_cell_volume_m3=float(volume.min()),total_volume_m3=float(volume.sum()),
        normalized_volume_quality_quantiles=np.quantile(shape,[0,.01,.5,1]).tolist(),
        signed_volume_positive=int(sum(np.count_nonzero(a>0) for a in signed)),
        signed_volume_negative=int(sum(np.count_nonzero(a<0) for a in signed)),
        sphere_radius_excess_m=float(max_sphere_excess),sphere_volume_m3=float(region_volumes[4]),
        region_volumes_m3=region_volumes.tolist(),bounds_min_m=lower.tolist(),bounds_max_m=upper.tolist(),
        ground_interface_triangles=len(ground_faces),source_edge_audit_passed=True,receiver_audit_passed=True)
    write(folder/"geometry_audit.json",audit_record)
    metadata=dict(mesh_name=NAME,case="terrain_layered_sphere",fe_degree=1,cells=int(mesh.num_cells()),
        vertices=int(mesh.num_vertices()),edges=int(edges),field_dofs=int(V.dim()),mixed_dofs=int(2*V.dim()),
        outer_km=outer_m/1000.,outer_halfwidth_m=outer_m,outer_fullwidth_m=2*outer_m,
        mesh_sha256=sha(h5path),parameters_sha256=sha(params_path),
        physical=dict(conductivity_S_per_m=SIGMA,mu_H_per_m=4*np.pi*1e-7,sphere_center_m=CENTER.tolist(),
                      sphere_radius_m=RADIUS,layer_top_reference_z_m=[0.,-300.,-1000.],relief_coefficients_m=[250.,90.,40.]),
        source=source_audit,receiver_csv="receivers.csv",receiver_xyz=rx.tolist(),sampling_material=1,
        geometry_audit="geometry_audit.json",source_csv="source.csv",source_edge_audit="source_edge_audit.json",
        receiver_audit="receiver_audit.json",ready_for_formal_run=False,
        required_before_freeze=["local-refinement/outer-boundary sensitivity","small-mesh MPI acceptance","formal-size representative frequencies"])
    settings_path=folder/"generation_settings.json"
    if settings_path.exists():
        settings=json.loads(settings_path.read_text())
        metadata.update(local_volume_factor=settings.get("local_volume_factor",1.),
                        generation_settings="generation_settings.json",
                        generation_settings_sha256=sha(settings_path))
    write(folder/"mesh.json",metadata)
    return metadata


def main(args):
    require(args.outer_km>=10, "Outer box must contain the central terrain patch")
    if args.internal_stage:
        internal_stage(args.internal_stage, args.output.resolve(), args.outer_km*1000.)
        return
    if args.audit_only:
        run_stage("audit", args.output.resolve(), args.outer_km*1000.)
        result=json.loads((args.output/"mesh.json").read_text())
        print(json.dumps(result,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    scale=args.scale;records=[]
    for attempt in range(1,args.max_attempts+1):
        folder=args.output/("attempt_%02d" % attempt);started=time.perf_counter()
        settings=generate(folder,args.outer_km*1000.,scale,args.mode,args.quality,args.local_volume_factor)
        run_stage("convert", folder, args.outer_km*1000.)
        run_stage("audit", folder, args.outer_km*1000.)
        metadata=json.loads((folder/"mesh.json").read_text())
        eligible=(args.minimum_edges<=metadata["edges"]<=args.maximum_edges)
        record=dict(attempt=attempt,directory=str(folder.resolve()),actual_edges=metadata["edges"],
                    settings=settings,elapsed_s=time.perf_counter()-started,size_in_requested_range=eligible)
        records.append(record);write(args.output/"attempts.json",records)
        write(args.output/"selection.json",dict(**record,formal_mesh_size_eligible=bool(eligible and args.mode=="formal" and args.outer_km>=80),
              ready_for_formal_run=False))
        print(json.dumps(record),flush=True)
        if eligible:break
        scale*=float(np.clip((metadata["edges"]/args.target_edges)**.9,.3,3.))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DATA_ROOT/"candidate_q17_local025")
    parser.add_argument("--template",type=Path,default=TEMPLATE_PATH,
                        help="Sphere geometry template; defaults to the package data directory")
    parser.add_argument("--mode",choices=("pilot","formal"),default="pilot")
    parser.add_argument("--outer-km",type=float,default=20.,
                        help="Box half-width along each axis in km; 80 means 160 km full width")
    parser.add_argument("--scale",type=float,default=1.)
    parser.add_argument("--local-volume-factor",type=float,default=1.,
                        help="Central surface/interface area factor^(2/3) and sphere volume factor; far constraints unchanged")
    parser.add_argument("--quality",type=float,default=1.4)
    parser.add_argument("--target-edges",type=int,default=2500000)
    parser.add_argument("--minimum-edges",type=int,default=2000000)
    parser.add_argument("--maximum-edges",type=int,default=3000000)
    parser.add_argument("--max-attempts",type=int,default=1)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--internal-stage",choices=("convert","audit"),help=argparse.SUPPRESS)
    args=parser.parse_args()
    TEMPLATE_PATH=args.template.expanduser().resolve()
    require(np.isfinite(args.local_volume_factor) and args.local_volume_factor>0,
            "Local volume factor must be finite and positive")
    require(args.scale>0 and args.max_attempts>0 and 0<args.minimum_edges<=args.target_edges<=args.maximum_edges,"Invalid scale/target range")
    main(args)
