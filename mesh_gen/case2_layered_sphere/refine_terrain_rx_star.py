#!/usr/bin/env python3
"""One receiver-star TetGen refinement; preserve the input and sampling policy.

No EM solve or acceptance claim is made. TetGen, conversion, audit and geometry
measurements are separate processes. Every attempted output is retained.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "mesh_data" / "case2_layered_sphere"

NAME="terrain_sphere"
POINT_TOL_M=1e-8
SOURCE_RX_TOL_M=1e-7
REGION_VOLUME_RTOL=1e-10
REGION_VOLUME_ATOL_M3=1e-3
MIN_EDGES=2000000
MAX_EDGES=3000000


def require(condition,message):
    if not condition:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024**2),b""):digest.update(block)
    return digest.hexdigest()


def companion(prefix,suffix):return Path(str(prefix)+"."+suffix)


def table(path):
    with Path(path).open() as stream:
        for line in stream:
            line=line.split("#",1)[0].strip()
            if line:
                header=[int(word) for word in line.split()];break
        else:raise RuntimeError("Empty TetGen file: "+str(path))
        values=np.atleast_2d(np.loadtxt(stream,comments="#"))
    require(len(values)==header[0] and np.isfinite(values).all(),"Invalid TetGen table: "+str(path))
    ids=values[:,0].astype(np.int64)
    require(ids[0] in (0,1) and np.array_equal(values[:,0],np.arange(ids[0],ids[0]+len(ids))),
            "TetGen IDs must be consecutive zero/one based: "+str(path))
    return header,values


def mesh_tables(prefix):
    nh,nodes=table(companion(prefix,"node"));eh,elements=table(companion(prefix,"ele"))
    require(nh[1]==3 and nodes.shape[1]==4+nh[2]+nh[3],"Invalid node dimension/columns")
    require(eh[1:]==[4,1] and elements.shape[1]==6,"Require P1 tetrahedra with one material attribute")
    require(nodes[0,0]==elements[0,0],"Node/element numbering bases differ")
    connectivity=elements[:,1:5].astype(np.int64)
    require(np.array_equal(connectivity,elements[:,1:5]) and connectivity.min()>=nodes[0,0] and
            connectivity.max()<nodes[0,0]+len(nodes),"Invalid tetrahedron node IDs")
    tags=elements[:,5].astype(np.int64)
    require(np.array_equal(tags,elements[:,5]),"Material attributes must be integers")
    return nodes,elements[:,0].astype(np.int64),connectivity,tags


def tet_geometry(points):
    volume=np.abs(np.einsum("ij,ij->i",points[:,1]-points[:,0],np.cross(points[:,2]-points[:,0],points[:,3]-points[:,0])))/6.
    longest=np.maximum.reduce([np.linalg.norm(points[:,a]-points[:,b],axis=1) for a in range(4) for b in range(a)])
    area=np.maximum.reduce([.5*np.linalg.norm(np.cross(points[:,b]-points[:,a],points[:,c]-points[:,a]),axis=1)
                           for a,b,c in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    require(np.isfinite(volume).all() and np.all(volume>0) and np.all(area>0),"Degenerate/nonfinite tetrahedron")
    return volume,longest,3*volume/area,6*volume/longest**3


def map_points(points,coordinates,tolerance):
    from scipy.spatial import cKDTree
    distance,indices=cKDTree(points).query(coordinates)
    require(np.isfinite(distance).all() and np.all(distance<=tolerance) and len(np.unique(indices))==len(indices),
            "Requested old/source/receiver points missing, moved, or merged")
    return indices,distance


def receiver_constraints(nodes,element_ids,connectivity,tags,rx,factor):
    require(rx.shape==(200,3),"Require all 200 original receiver coordinates")
    indices,distances=map_points(nodes[:,1:4],rx,SOURCE_RX_TOL_M)
    receiver_node_ids=nodes[indices,0].astype(np.int64)
    selected=np.isin(connectivity,receiver_node_ids).any(axis=1)
    require(selected.any() and set(tags[selected])<={0,1},"Receiver stars must contain air/material-1 cells only")
    star_connectivity,star_tags=connectivity[selected],tags[selected]
    for node in receiver_node_ids:
        require(set(star_tags[np.any(star_connectivity==node,axis=1)])=={0,1},"A receiver is not on the air/first-layer interface")
    volume=np.empty(len(connectivity));origin=int(nodes[0,0])
    for start in range(0,len(connectivity),50000):
        stop=min(start+50000,len(connectivity))
        volume[start:stop]=tet_geometry(nodes[connectivity[start:stop]-origin,1:4])[0]
    caps=np.full(len(connectivity),-1.,dtype=float)
    caps[selected]=volume[selected]*factor
    return caps,selected,volume,receiver_node_ids,float(distances.max())


def measure_receivers(mesh_folder,output_prefix):
    """Read-only fresh DOLFIN process; same material-1/min-global-cell policy."""
    import faulthandler
    faulthandler.enable(all_threads=True)
    import dolfin as df
    require(df.MPI.size(df.MPI.comm_world)==1,"Geometry measurements are serial")
    mesh=df.Mesh();metadata=read(mesh_folder/"mesh.json");rx=np.asarray(metadata["receiver_xyz"],dtype=float)
    require(rx.shape==(200,3) and np.isfinite(rx).all(),"Require all 200 finite original Rx coordinates")
    with df.HDF5File(df.MPI.comm_world,str(mesh_folder/"meshes/_h5"/(NAME+".h5")),"r") as stream:
        stream.read(mesh,"/mesh",False);domains=df.MeshFunction("size_t",mesh,3);stream.read(domains,"/domains")
    mesh.order();mesh.init(0,3);mesh.init(3,0)
    coordinates,cells,tags=mesh.coordinates(),mesh.cells(),domains.array()
    indices,distance=map_points(coordinates,rx,SOURCE_RX_TOL_M)
    rows=[];detail=[];all_cells=set()
    for index,vertex in enumerate(indices):
        incident=np.asarray(df.Vertex(mesh,int(vertex)).entities(3),dtype=np.int64)
        require(set(tags[incident])=={0,1},"Receiver does not have both air and earth incident cells")
        earth=incident[tags[incident]==1];all_cells.update(map(int,incident))
        row=dict(rx_id=index+1,x_m=float(rx[index,0]),y_m=float(rx[index,1]),z_m=float(rx[index,2]),
                 air_incident_count=int(np.count_nonzero(tags[incident]==0)))
        for label,ids in (("all",incident),("earth",earth)):
            volume,longest,altitude,quality=tet_geometry(coordinates[cells[ids]])
            row.update({label+"_incident_count":len(ids),label+"_volume_min_m3":float(volume.min()),
                label+"_volume_median_m3":float(np.median(volume)),label+"_volume_max_m3":float(volume.max()),
                label+"_hmax_m":float(longest.max()),label+"_minimum_altitude_m":float(altitude.min()),
                label+"_normalized_volume_quality_min":float(quality.min())})
        selected=min(map(int,earth),key=lambda cell:df.Cell(mesh,cell).global_index())
        point=coordinates[cells[selected]];v,h,a,q=tet_geometry(point[None,:,:])
        canonical=point[np.lexsort((point[:,2],point[:,1],point[:,0]))]
        row.update(selected_cell_global_id=int(df.Cell(mesh,selected).global_index()),selected_volume_m3=float(v[0]),
            selected_hmax_m=float(h[0]),selected_minimum_altitude_m=float(a[0]),
            selected_normalized_volume_quality=float(q[0]),
            selected_physical_cell_sha256=hashlib.sha256(np.ascontiguousarray(canonical,dtype="<f8").tobytes()).hexdigest())
        rows.append(row);detail.append(dict(rx_id=index+1,selected_cell_vertices_m=canonical.tolist(),
                                           incident_global_ids=[int(df.Cell(mesh,int(c)).global_index()) for c in incident]))
    with Path(str(output_prefix)+".csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(kind="geometry_only_no_EM",count=200,unique_incident_cells=len(all_cells),
        sampling_policy="material 1; minimum global incident cell; unchanged pointwise E/curl-H",
        receiver_xyz=rx.tolist(),maximum_vertex_match_distance_m=float(distance.max()),
        maximum_earth_hmax_m=max(row["earth_hmax_m"] for row in rows),
        maximum_selected_hmax_m=max(row["selected_hmax_m"] for row in rows),per_receiver=rows,cell_details=detail)
    write(Path(str(output_prefix)+".json"),summary)
    print(json.dumps({key:summary[key] for key in ("count","unique_incident_cells","maximum_earth_hmax_m","maximum_selected_hmax_m")}),flush=True)


def run_metrics(mesh_folder,output,phase):
    command=[sys.executable,"-u",str(Path(__file__).resolve()),"--input",str(mesh_folder),"--output",str(output),
             "--internal-metrics",phase]
    with (output/("metrics_"+phase+".log")).open("w") as stream:
        process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONFAULTHANDLER="1"))
    require(process.returncode==0,"Fresh receiver metrics failed; inspect metrics_"+phase+".log")


def check_conservation(before,after,old_geometry,new_geometry):
    require(before["physical"]==after["physical"],"Physical model changed")
    require(before["outer_km"]==after["outer_km"]==80.,"Outer box must remain +/-80 km")
    require(np.allclose(before["receiver_xyz"],after["receiver_xyz"],atol=SOURCE_RX_TOL_M,rtol=0),"Original receiver coordinates changed")
    old,new=before["source"],after["source"]
    require(old["current_A"]==new["current_A"]==1. and old["horizontal_length_m"]==new["horizontal_length_m"]==500.,"Source current/length changed")
    a,b=np.asarray(old["vertices"]),np.asarray(new["vertices"])
    require(np.allclose(a[[0,-1]],b[[0,-1]],atol=SOURCE_RX_TOL_M,rtol=0),"Source endpoints changed")
    x=np.unique(np.r_[a[:,0],b[:,0]])
    require(np.allclose(np.interp(x,a[:,0],a[:,2]),np.interp(x,b[:,0],b[:,2]),atol=SOURCE_RX_TOL_M,rtol=0) and
            np.all(np.abs(a[:,1])<=SOURCE_RX_TOL_M) and np.all(np.abs(b[:,1])<=SOURCE_RX_TOL_M),"Source polyline changed")
    require(abs(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()-np.linalg.norm(np.diff(b,axis=0),axis=1).sum())<=SOURCE_RX_TOL_M,
            "Terrain-following wire length changed")
    require(new["complete_mesh_edges"] and new["source_on_ground_interface"],"Refined source is not complete FE edges")
    old_volume=np.asarray(old_geometry["region_volumes_m3"]);new_volume=np.asarray(new_geometry["region_volumes_m3"])
    require(np.allclose(old_volume,new_volume,rtol=REGION_VOLUME_RTOL,atol=REGION_VOLUME_ATOL_M3),"Physical-region volumes changed")
    for key in ("bounds_min_m","bounds_max_m"):
        require(np.allclose(old_geometry[key],new_geometry[key],atol=POINT_TOL_M,rtol=0),"Outer boundary changed")
    return dict(passed=True,region_volume_difference_m3=(new_volume-old_volume).tolist(),
                point_tolerance_m=POINT_TOL_M,source_receiver_tolerance_m=SOURCE_RX_TOL_M,
                region_volume_rtol=REGION_VOLUME_RTOL,region_volume_atol_m3=REGION_VOLUME_ATOL_M3)


def refine(args):
    source=args.input.resolve();output=args.output.resolve()
    require(math.isfinite(args.volume_factor) and 0<args.volume_factor<1,"Volume factor must be finite and between zero and one")
    require(math.isfinite(args.quality) and args.quality>=1,"Quality radius-edge bound must be finite and >=1")
    require(shutil.which("tetgen"),"TetGen executable missing from PATH")
    before=read(source/"mesh.json");old_geometry=read(source/"geometry_audit.json")
    require(before["mesh_name"]==NAME and before["fe_degree"]==1 and before["outer_km"]==80. and
            old_geometry.get("passed") is True,"Require an audited P1 +/-80 km input mesh")
    require(before.get("source",{}).get("passed") is True and before["source"].get("complete_mesh_edges") is True,
            "Original source edge audit did not pass")
    previous=read(source/"refinement.json") if (source/"refinement.json").exists() else {}
    original=source/previous.get("tetgen_output_prefix","mesh_create/"+NAME+".1")
    inputs=[companion(original,suffix) for suffix in ("node","ele","face","edge")]
    inputs += [source/name for name in ("mesh.json","geometry_audit.json","source_edge_audit.json","receiver_audit.json","receivers.csv","source.csv","generation_settings.json")]
    parameters=source/"meshes/para"/(NAME+"_parameters.json");h5=source/"meshes/_h5"/(NAME+".h5")
    inputs += [parameters,h5,Path(__file__).resolve(),Path(__file__).with_name("build_terrain_mesh_sensitivity.py")]
    if (source/"refinement.json").exists():inputs.append(source/"refinement.json")
    require(all(path.is_file() for path in inputs),"Missing original TetGen companions/audit/FEM input")
    hashes={str(path):sha(path) for path in inputs}
    require(hashes[str(h5)]==before["mesh_sha256"] and hashes[str(parameters)]==before["parameters_sha256"],"Input FEM/parameters SHA mismatch")
    output.mkdir(parents=True,exist_ok=False)
    record=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),parent_directory=str(source),
        input_sha256=hashes,volume_factor=args.volume_factor,quality=args.quality,
        size_budget_complex_dofs=[MIN_EDGES,MAX_EDGES],ready_for_formal_run=False,spatial_sensitivity_passed=None)
    write(output/"refinement.json",record)
    try:
        for name in ("mesh_create","meshes/_vtk","meshes/_xml","meshes/_h5","meshes/para"):(output/name).mkdir(parents=True)
        copied=output/"mesh_create"/(NAME+".1")
        for suffix in ("node","ele","face","edge"):shutil.copy2(companion(original,suffix),companion(copied,suffix))
        shutil.copy2(parameters,output/"meshes/para"/parameters.name)
        settings=read(source/"generation_settings.json")
        settings.update(generation_route="TetGen -r per-cell receiver-star volume refinement",quality=args.quality,
            receiver_star_volume_factor=args.volume_factor,global_region_volume_limits_reapplied=False)
        write(output/"generation_settings.json",settings)
        run_metrics(source,output,"before")
        nodes,element_ids,connectivity,tags=mesh_tables(copied)
        require(set(tags)=={0,1,2,3,4},"Input material tags differ from 0..4")
        rx=np.asarray(before["receiver_xyz"],dtype=float)
        caps,selected,volumes,rx_ids,distance=receiver_constraints(nodes,element_ids,connectivity,tags,rx,args.volume_factor)
        volume_file=companion(copied,"vol")
        with volume_file.open("w") as stream:
            stream.write(str(len(element_ids))+"\n")
            np.savetxt(stream,np.column_stack((element_ids,caps)),fmt=["%d","%.17g"])
        np.savez_compressed(output/"selected_tetrahedra.npz",tetgen_element_ids=element_ids[selected],
            material=tags[selected],original_volume_m3=volumes[selected],maximum_volume_m3=caps[selected],receiver_tetgen_node_ids=rx_ids)
        command=["tetgen","-r","-a","-q%g"%args.quality,"-O0","-J","-k","-C","-V",str(copied)]
        record.update(command=command,volume_file_sha256=sha(volume_file),selected_tetrahedra=int(selected.sum()),
            selected_by_material={str(tag):int(np.count_nonzero(selected&(tags==tag))) for tag in (0,1)},
            receiver_node_match_max_distance_m=distance)
        write(output/"refinement.json",record)
        with (output/"tetgen.log").open("w") as stream:process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        require(process.returncode==0,"TetGen refinement failed; inspect retained tetgen.log")
        generated=output/"mesh_create"/(NAME+".2")
        new_nodes,new_ids,new_cells,new_tags=mesh_tables(generated)
        matched,displacement=map_points(new_nodes[:,1:4],nodes[:,1:4],POINT_TOL_M)
        require(set(new_tags)==set(tags),"Physical material tags were lost or added")
        require(len(np.unique(new_cells))==len(new_nodes),"Refinement left unused vertices")
        require(len(new_nodes)>=len(nodes),"Original vertices were removed")
        np.savez_compressed(output/"old_node_mapping.npz",old_node_ids=nodes[:,0].astype(np.int64),
            new_node_ids=new_nodes[matched,0].astype(np.int64),coordinate_displacement_m=displacement)
        record.update(old_node_conservation=dict(passed=True,count=len(nodes),maximum_displacement_m=float(displacement.max()),
            bit_exact_coordinates=bool(np.array_equal(nodes[:,1:4],new_nodes[matched,1:4]))),tetgen_output_prefix="mesh_create/"+NAME+".2")
        vtk=companion(generated,"vtk");require(vtk.is_file(),"TetGen did not produce VTK")
        shutil.copy2(vtk,output/"meshes/_vtk"/(NAME+".vtk"))
        import build_terrain_mesh_sensitivity as builder
        builder.run_stage("convert",output,80000.)
        builder.run_stage("audit",output,80000.)
        after=read(output/"mesh.json");new_geometry=read(output/"geometry_audit.json")
        require(after["cells"]==len(new_ids) and after["vertices"]==len(new_nodes),"FEM/TetGen counts differ")
        conservation=check_conservation(before,after,old_geometry,new_geometry)
        run_metrics(output,output,"after")
        before_metrics=read(output/"rx_star_before.json");after_metrics=read(output/"rx_star_after.json")
        eligible=MIN_EDGES<=after["field_dofs"]<=MAX_EDGES
        record.update(status="completed",ended_utc=datetime.now(timezone.utc).isoformat(),
            conservation=conservation,cells=after["cells"],vertices=after["vertices"],field_dofs=after["field_dofs"],
            dof_increment=after["field_dofs"]-before["field_dofs"],geometry_and_size_eligible=eligible,
            acceptance_status="eligible_for_validation" if eligible else "unaccepted_size_budget",
            parent_field_dofs=before["field_dofs"],receiver_earth_hmax_before_m=before_metrics["maximum_earth_hmax_m"],
            receiver_earth_hmax_after_m=after_metrics["maximum_earth_hmax_m"],
            output_sha256={str(path.relative_to(output)):sha(path) for path in
                [companion(generated,suffix) for suffix in ("node","ele","face","edge","vtk")]+[output/"mesh.json",output/"rx_star_before.csv",output/"rx_star_after.csv"]})
        require(all(sha(path)==digest for path,digest in hashes.items()),"An original input or script changed during refinement")
        record["original_inputs_unchanged"]=True
        after["receiver_star_refinement"]=dict(volume_factor=args.volume_factor,parent_mesh_sha256=before["mesh_sha256"],
            field_dof_increment=record["dof_increment"],geometry_and_size_eligible=eligible,refinement_record="refinement.json")
        write(output/"mesh.json",after)
        record["output_sha256"]["mesh.json"]=sha(output/"mesh.json")
    except BaseException as error:
        record.update(status="failed",failure=type(error).__name__+": "+str(error),
                      ended_utc=datetime.now(timezone.utc).isoformat(),geometry_and_size_eligible=False,acceptance_status="unaccepted_failure")
        write(output/"refinement.json",record)
        raise
    write(output/"refinement.json",record)
    print(json.dumps(record,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DATA_ROOT/"candidate_q17_local025/attempt_01")
    parser.add_argument("--output",type=Path,default=DATA_ROOT/"tetgen_rx_star_refinement")
    parser.add_argument("--volume-factor",type=float,default=.25)
    parser.add_argument("--quality",type=float,default=1.7)
    parser.add_argument("--internal-metrics",choices=("before","after"),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.internal_metrics:measure_receivers(args.input.resolve(),args.output.resolve()/("rx_star_"+args.internal_metrics))
    else:refine(args)


if __name__=="__main__":main()
