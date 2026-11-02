#!/usr/bin/env python3
"""One serial DOLFIN Plaza earth-side Rx-star refinement; new mesh strategy.

Only original Rx-incident material-1 tetrahedra are marked. Air cells can be
subdivided by conformity closure. Old vertices, parent-cell volumes, material
tags, the full FE wire and receiver coordinates must pass conservation checks.
Every attempt is retained; size eligibility is not EM/sensitivity acceptance.
The input outer half-width may be >=80 km and must remain unchanged. An output
outside the original 2--3M DOF budget remains unaccepted_size_budget, including
larger meshes prepared only as spatial-sensitivity references.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil

import numpy as np
import refine_terrain_rx_star as base


def cell_volumes(xyz,cells):
    result=np.empty(len(cells))
    for start in range(0,len(cells),50000):
        stop=min(start+50000,len(cells));points=xyz[cells[start:stop]]
        result[start:stop]=np.abs(np.einsum('ij,ij->i',points[:,1]-points[:,0],
            np.cross(points[:,2]-points[:,0],points[:,3]-points[:,0])))/6.
    base.require(np.isfinite(result).all() and np.all(result>0),'Degenerate/nonfinite tetrahedron')
    return result


def refine_once(mesh,domains,rx):
    import dolfin as df
    base.require(df.MPI.size(mesh.mpi_comm())==1,'Plaza refinement must run serially')
    rx=np.asarray(rx,dtype=float)
    base.require(rx.ndim==2 and rx.shape[1]==3 and len(rx)>0 and np.isfinite(rx).all(),'Invalid Rx coordinates')
    old_xyz=mesh.coordinates().copy();old_cells=mesh.cells().copy();old_tags=domains.array().copy()
    mesh.init(0,3)
    vertices,distances=base.map_points(old_xyz,rx,base.SOURCE_RX_TOL_M)
    selected=np.zeros(mesh.num_cells(),dtype=bool)
    for vertex in vertices:
        incident=df.Vertex(mesh,int(vertex)).entities(3)
        base.require(set(old_tags[incident])=={0,1},'Rx star must contain both air/material-1 and no other regions')
        selected[incident[old_tags[incident]==1]]=True
    markers=df.MeshFunction('bool',mesh,3,False);markers.array()[:]=selected
    original_algorithm=df.parameters['refinement_algorithm']
    try:
        df.parameters['refinement_algorithm']='plaza'
        refined=df.refine(mesh,markers,redistribute=False)
    finally:
        df.parameters['refinement_algorithm']=original_algorithm
    parent=np.asarray(refined.data().array('parent_cell',3),dtype=np.int64).copy()
    base.require(len(parent)==refined.num_cells() and parent.min()>=0 and parent.max()<len(old_cells),
                 'Invalid native parent_cell map')
    refined_domains=df.MeshFunction('size_t',refined,3)
    refined_domains.array()[:]=old_tags[parent]
    xyz,cells=refined.coordinates(),refined.cells()
    mapping,displacement=base.map_points(xyz,old_xyz,base.POINT_TOL_M)
    base.require(np.array_equal(old_xyz,xyz[mapping]),'An original vertex moved or was lost; require bit-exact preservation')
    old_volume=cell_volumes(old_xyz,old_cells);new_volume=cell_volumes(xyz,cells)
    sum_by_parent=np.bincount(parent,weights=new_volume,minlength=len(old_cells))
    base.require(np.allclose(sum_by_parent,old_volume,rtol=1e-10,atol=1e-8),'Child volumes do not conserve each parent tetrahedron')
    base.require(np.array_equal(old_xyz.min(0),xyz.min(0)) and np.array_equal(old_xyz.max(0),xyz.max(0)),
                 'Outer bounds changed')
    rx_new,rx_distance=base.map_points(xyz,rx,base.SOURCE_RX_TOL_M)
    refined.init(0,3)
    for vertex in rx_new:
        base.require(set(refined_domains.array()[df.Vertex(refined,int(vertex)).entities(3)])=={0,1},
                     'Refined Rx lost the air/material-1 interface')
    mesh.init(1);refined.init(1)
    child_count=np.bincount(parent,minlength=len(old_cells))
    record=dict(algorithm='DOLFIN Plaza',redistribute=False,rounds=1,
        marking_strategy='all Rx-incident material-1 cells only; air through conformity closure',
        explicitly_marked_materials=[1],
        old_cells=len(old_cells),new_cells=len(cells),old_vertices=len(old_xyz),new_vertices=len(xyz),
        old_field_dofs=int(mesh.num_entities(1)),new_field_dofs=int(refined.num_entities(1)),
        marked_cells=int(selected.sum()),marked_by_material={str(tag):int(np.count_nonzero(selected&(old_tags==tag))) for tag in (0,1)},
        conformity_refined_unmarked_parents=int(np.count_nonzero((child_count>1)&~selected)),
        conformity_refined_unmarked_by_material={str(tag):int(np.count_nonzero((child_count>1)&~selected&(old_tags==tag))) for tag in range(5)},
        unchanged_parent_cells=int(np.count_nonzero(child_count==1)),
        old_vertices_bit_exact=True,maximum_old_vertex_displacement_m=float(displacement.max()),
        parent_volume_conserved=True,maximum_parent_relative_volume_error=float(np.max(np.abs(sum_by_parent-old_volume)/old_volume)),
        old_region_volumes=np.bincount(old_tags.astype(np.int64),weights=old_volume,minlength=5).tolist(),
        new_region_volumes=np.bincount(refined_domains.array().astype(np.int64),weights=new_volume,minlength=5).tolist(),
        original_receiver_match_max_distance_m=float(distances.max()),refined_receiver_match_max_distance_m=float(rx_distance.max()))
    arrays=dict(parent_cell=parent,marked_parent_cells=np.flatnonzero(selected),old_vertex_ids=np.arange(len(old_xyz)),
        new_vertex_ids=mapping,coordinate_displacement_m=displacement,receiver_new_vertex_ids=rx_new)
    return refined,refined_domains,record,arrays


def write_h5(path,mesh,domains):
    import dolfin as df
    with df.HDF5File(df.MPI.comm_world,str(path),'w') as stream:
        stream.write(mesh,'/mesh');stream.write(domains,'/domains')


def check_conservation(before,after,old_geometry,new_geometry):
    """Same strict checks as the original base; permit a fixed wider box."""
    base.require(before["physical"]==after["physical"],"Physical model changed")
    base.require(np.isfinite(before["outer_km"]) and before["outer_km"]==after["outer_km"] and
                 before["outer_km"]>=80.,"Outer half-width must remain unchanged and be >=80 km")
    base.require(np.allclose(before["receiver_xyz"],after["receiver_xyz"],atol=base.SOURCE_RX_TOL_M,rtol=0),"Original receiver coordinates changed")
    old,new=before["source"],after["source"]
    base.require(old["current_A"]==new["current_A"]==1. and old["horizontal_length_m"]==new["horizontal_length_m"]==500.,"Source current/length changed")
    a,b=np.asarray(old["vertices"]),np.asarray(new["vertices"])
    base.require(np.allclose(a[[0,-1]],b[[0,-1]],atol=base.SOURCE_RX_TOL_M,rtol=0),"Source endpoints changed")
    x=np.unique(np.r_[a[:,0],b[:,0]])
    base.require(np.allclose(np.interp(x,a[:,0],a[:,2]),np.interp(x,b[:,0],b[:,2]),atol=base.SOURCE_RX_TOL_M,rtol=0) and
            np.all(np.abs(a[:,1])<=base.SOURCE_RX_TOL_M) and np.all(np.abs(b[:,1])<=base.SOURCE_RX_TOL_M),"Source polyline changed")
    base.require(abs(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()-np.linalg.norm(np.diff(b,axis=0),axis=1).sum())<=base.SOURCE_RX_TOL_M,
            "Terrain-following wire length changed")
    base.require(new["complete_mesh_edges"] and new["source_on_ground_interface"],"Refined source is not complete FE edges")
    old_volume=np.asarray(old_geometry["region_volumes_m3"]);new_volume=np.asarray(new_geometry["region_volumes_m3"])
    base.require(np.allclose(old_volume,new_volume,rtol=base.REGION_VOLUME_RTOL,atol=base.REGION_VOLUME_ATOL_M3),"Physical-region volumes changed")
    for key in ("bounds_min_m","bounds_max_m"):
        base.require(np.allclose(old_geometry[key],new_geometry[key],atol=base.POINT_TOL_M,rtol=0),"Outer boundary changed")
    return dict(passed=True,region_volume_difference_m3=(new_volume-old_volume).tolist(),
                point_tolerance_m=base.POINT_TOL_M,source_receiver_tolerance_m=base.SOURCE_RX_TOL_M,
                region_volume_rtol=base.REGION_VOLUME_RTOL,region_volume_atol_m3=base.REGION_VOLUME_ATOL_M3)


def main(args):
    import dolfin as df
    import build_terrain_mesh_sensitivity as builder
    base.require(df.MPI.size(df.MPI.comm_world)==1,'Refinement entry point requires exactly one MPI rank')
    source,output=args.input.resolve(),args.output.resolve()
    base.require(args.max_added_dofs>0,'max-added-dofs must be positive')
    before=base.read(source/'mesh.json');old_geometry=base.read(source/'geometry_audit.json')
    base.require(before['mesh_name']==base.NAME and before['fe_degree']==1 and
        np.isfinite(before['outer_km']) and before['outer_km']>=80. and
        old_geometry.get('passed') is True and before['source'].get('complete_mesh_edges') is True,
        'Require audited P1 input with outer half-width >=80 km and complete FE wire')
    rx=np.asarray(before['receiver_xyz'],dtype=float)
    base.require(rx.shape==(200,3),'Require the original 200 Rx coordinates')
    h5=source/'meshes/_h5'/(base.NAME+'.h5');parameters=source/'meshes/para'/(base.NAME+'_parameters.json')
    files=[source/name for name in ('mesh.json','geometry_audit.json','source_edge_audit.json','receiver_audit.json','receivers.csv','source.csv')]
    files += [h5,parameters,Path(__file__).resolve(),Path(base.__file__).resolve(),Path(builder.__file__).resolve()]
    hashes={str(path):base.sha(path) for path in files}
    base.require(hashes[str(h5)]==before['mesh_sha256'] and hashes[str(parameters)]==before['parameters_sha256'],
                 'Input HDF5/parameters SHA mismatch')
    output.mkdir(parents=True,exist_ok=False)
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),parent_directory=str(source),
        input_sha256=hashes,route='serial DOLFIN Plaza; earth-only Rx stars; one round; direct HDF5; no TetGen',
        outer_half_width_km=before['outer_km'],
        size_budget_complex_dofs=[base.MIN_EDGES,base.MAX_EDGES],maximum_added_dofs=args.max_added_dofs,
        ready_for_formal_run=False,spatial_sensitivity_passed=None)
    base.write(output/'refinement.json',record)
    try:
        for name in ('meshes/_h5','meshes/para'):(output/name).mkdir(parents=True)
        shutil.copy2(parameters,output/'meshes/para'/parameters.name)
        settings=base.read(source/'generation_settings.json') if (source/'generation_settings.json').exists() else {}
        settings.update(generation_route=record['route'],receiver_star_refinement_algorithm='plaza',
            receiver_star_explicitly_marked_materials=[1],refinement_rounds_this_call=1)
        base.write(output/'generation_settings.json',settings)
        base.run_metrics(source,output,'before')
        mesh=df.Mesh()
        with df.HDF5File(df.MPI.comm_world,str(h5),'r') as stream:
            stream.read(mesh,'/mesh',False);domains=df.MeshFunction('size_t',mesh,3);stream.read(domains,'/domains')
        base.require(set(domains.array())=={0,1,2,3,4},'Material tags must be exactly 0..4')
        refined,new_domains,native,arrays=refine_once(mesh,domains,rx)
        base.require(native['old_field_dofs']==before['field_dofs'],'Input edge DOF identity changed')
        record['native_refinement']=native
        np.savez_compressed(output/'parent_cell_and_old_node_mapping.npz',**arrays)
        candidate=output/'meshes/_h5'/h5.name
        write_h5(candidate,refined,new_domains)
        # Fresh native reader audits the new HDF5 directly; no VTK conversion.
        builder.run_stage('audit',output,before['outer_km']*1000.)
        after=base.read(output/'mesh.json');new_geometry=base.read(output/'geometry_audit.json')
        base.require(after['cells']==native['new_cells'] and after['vertices']==native['new_vertices'] and
            after['field_dofs']==native['new_field_dofs'],'Fresh HDF5 audit differs from native refinement counts')
        conservation=check_conservation(before,after,old_geometry,new_geometry)
        base.run_metrics(output,output,'after')
        increment=after['field_dofs']-before['field_dofs']
        eligible=base.MIN_EDGES<=after['field_dofs']<=base.MAX_EDGES and 0<increment<=args.max_added_dofs
        record.update(status='completed',ended_utc=datetime.now(timezone.utc).isoformat(),conservation=conservation,
            cells=after['cells'],vertices=after['vertices'],field_dofs=after['field_dofs'],dof_increment=increment,
            geometry_and_size_eligible=bool(eligible),
            acceptance_status='eligible_for_validation' if eligible else 'unaccepted_size_budget',
            receiver_metrics_before='rx_star_before.json',receiver_metrics_after='rx_star_after.json')
        base.require(all(base.sha(path)==digest for path,digest in hashes.items()),'Original input or helper changed')
        record['original_inputs_unchanged']=True
        after['receiver_star_refinement']=dict(algorithm='DOLFIN Plaza',rounds_this_call=1,
            explicitly_marked_materials=[1],air_refinement_policy='conformity closure only',
            parent_mesh_sha256=before['mesh_sha256'],field_dof_increment=increment,
            geometry_and_size_eligible=bool(eligible),refinement_record='refinement.json')
        base.write(output/'mesh.json',after)
        record['output_sha256']={name:base.sha(output/name) for name in (
            'mesh.json','geometry_audit.json','parent_cell_and_old_node_mapping.npz',
            'source_edge_audit.json','receiver_audit.json','rx_star_before.csv','rx_star_after.csv',
            'meshes/_h5/'+h5.name,'meshes/para/'+parameters.name)}
    except BaseException as error:
        record.update(status='failed',failure=type(error).__name__+': '+str(error),
            ended_utc=datetime.now(timezone.utc).isoformat(),geometry_and_size_eligible=False,acceptance_status='unaccepted_failure')
        base.write(output/'refinement.json',record)
        raise
    base.write(output/'refinement.json',record)
    print(base.read(output/'refinement.json'),flush=True)
    return 0 if record['geometry_and_size_eligible'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=base.DATA_ROOT/'q17_rx_plaza_round_01')
    parser.add_argument('--output',type=Path,default=base.DATA_ROOT/'q17_rx_plaza_earth_round_02')
    parser.add_argument('--max-added-dofs',type=int,default=100000)
    raise SystemExit(main(parser.parse_args()))
