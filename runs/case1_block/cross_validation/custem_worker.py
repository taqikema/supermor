#!/usr/bin/env python3
"""P1 native mixed-UFL DFOM and native split-u SUPERMOR worker."""
from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256 as hash_bytes
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Executable from any working directory; all dependencies stay external.
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "core"), str(ROOT)]

import numpy as np

from runs.common.mesh_inputs import get_case_mesh, runtime_mesh

if __package__:
    from .validation_common import (FREQUENCIES, identity, merge_responses,
                                    physical_config, receivers, save_frequency,
                                    sha256, write_json)
else:
    from validation_common import (FREQUENCIES, identity, merge_responses,
                                   physical_config, receivers, save_frequency,
                                   sha256, write_json)


THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
POLES = [-2274.7560643370834, -2.221441469079183]
MOR_OPTIONS = dict(basis_strategy="residual_mr", residual_correction=False,
                   direct=False, field_formulation="split_u", inner_rtol=1e-6,
                   tol=1e-6, max_basis=400, poles=POLES, pole_every=20,
                   pole_schedule="hold_first")
TOLERANCE = 1e-6
SOURCE_GEOMETRY_TOLERANCE_M = 1e-8


def configure_dfom_options(options):
    """Retain the formal DFOM diagnostics, refinement and in-core LU policy."""
    options["hed_dfom_sym2_mat_mumps_icntl_4"] = 2
    options["hed_dfom_sym2_mat_mumps_icntl_10"] = 0
    options["hed_dfom_sym2_mat_mumps_icntl_22"] = 0


def nonnegative_level(value):
    level = int(value)
    if level < 0:
        raise ValueError("level must be a nonnegative integer")
    return level


def parse_indices(value):
    result = list(range(len(FREQUENCIES))) if value is None or value.strip() == "" else [int(x) for x in value.split(",")]
    if not result or len(set(result)) != len(result) or any(x < 0 or x >= len(FREQUENCIES) for x in result):
        raise ValueError("indices must be distinct integers from 0 through 3")
    return result


def configure_physics(model, frequencies):
    """Fix the finite-wire path and its geometric edge-selection tolerance."""
    model.MP.update_model_parameters(frequencies=list(frequencies), current=1.,
        sigma_air=1e-8, sigma_ground=[.01, 1.], sigma_0=[.01, .01], mu=physical_config()["mu_H_per_m"])
    model.FE.build_var_form(s_type="line", start=[-2500., 0., 0.], stop=[-2000., 0., 0.],
                            quasi_static=True, ip=False, bc=None, a_tol=SOURCE_GEOMETRY_TOLERANCE_M)


def validate_runtime(stage, ranks, environment):
    accepted = {"dfom": (16,), "sumor": (32,)}[stage]
    if ranks not in accepted:
        raise ValueError("%s requires MPI rank count in %s; actual=%d" % (stage, accepted, ranks))
    if any(environment.get(key) != "1" for key in THREAD_VARIABLES):
        raise ValueError("set all of %s to 1" % ", ".join(THREAD_VARIABLES))


def new_attempt(output):
    directory = Path(output) / "attempts"
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        attempt = directory / ("attempt_%04d" % number)
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            number += 1


def root_action(comm, action):
    """Report root-only I/O failures on every rank before the next collective."""
    packet = None
    if comm.rank == 0:
        try:
            packet = dict(result=action(), error=None)
        except Exception as error:
            packet = dict(result=None, error=type(error).__name__ + ": " + str(error))
    packet = comm.bcast(packet, root=0)
    if packet["error"] is not None:
        raise RuntimeError(packet["error"])
    return packet["result"]


def pending_indices(output, requested, mesh_hash, implementation_hash=None, stage=None):
    result = []
    for index in requested:
        path = Path(output) / ("frequency_%02d.npz" % index)
        if not path.exists():
            result.append(index)
            continue
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("mesh_hash") != mesh_hash or meta.get("physical_hash") != identity()["physical_hash"]:
                raise ValueError("resume rejected: actual mesh or physical inputs changed")
            if int(data["index"]) != index or float(data["frequency"]) != FREQUENCIES[index]:
                raise ValueError("resume rejected: shard frequency/index mismatch")
            if not np.array_equal(data["coords"], receivers(True)):
                raise ValueError("resume rejected: receiver contract changed")
            if meta.get("status") == "complete":
                if implementation_hash is not None and meta.get("implementation_hash") != implementation_hash:
                    raise ValueError("resume rejected: completed shard uses a different implementation")
                if stage is not None and meta.get("stage") != stage:
                    raise ValueError("resume rejected: completed shard uses a different numerical method")
            valid = (meta.get("status") == "complete" and
                     isinstance(meta.get("residual"), (int, float)) and
                     0 <= meta["residual"] <= TOLERANCE and
                     all(data[key].shape == (3402, 3) and np.isfinite(data[key]).all()
                         for key in ("E", "H")))
            if not valid:
                result.append(index)
    return result


def independent_residual(ops, real, imag, omega):
    """Original K E + i omega M E = i omega q, with distributed PETSc norms."""
    residual, work = real.duplicate(), real.duplicate()
    try:
        ops["K"].mult(real, residual)
        ops["M"].mult(imag, work)
        residual.axpy(-omega, work)
        squared = residual.norm()**2
        ops["K"].mult(imag, residual)
        ops["M"].mult(real, work)
        residual.axpy(omega, work)
        residual.axpy(-omega, ops["q"])
        squared += residual.norm()**2
        denominator = omega * ops["q"].norm()
        return float(squared**.5 / denominator) if denominator else float(squared**.5)
    finally:
        residual.destroy()
        work.destroy()


def run_solver(args):
    from mpi4py import MPI
    import dolfin as df
    from petsc4py import PETSc
    comm = MPI.COMM_WORLD
    validate_runtime(args.stage, comm.size, os.environ)
    if df.MPI.size(df.MPI.comm_world) != comm.size:
        raise RuntimeError("DOLFIN and mpi4py communicators differ")
    import custEM
    from custEM.core import MOD
    from supermor.mor import solve_split_u
    from supermor.operators import assemble_mor, sample_receivers
    source = Path(custEM.__file__).resolve().parent.parent
    blas = ctypes.CDLL(str(Path(sys.prefix) / "lib" / "libopenblas.so.0"),
                       mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    blas.openblas_set_num_threads.argtypes = [ctypes.c_int]
    blas.openblas_set_num_threads(1)
    blas.openblas_get_num_threads.restype = ctypes.c_int
    if blas.openblas_get_num_threads() != 1:
        raise RuntimeError("OpenBLAS did not retain one compute thread")
    df.set_log_level(40)
    started = time.perf_counter()
    inputs = get_case_mesh("block_cross", data_root=args.data_root, require=True)
    meshes = inputs["m_dir"]
    name = "custem_L2"
    metadata = json.loads((meshes / "mesh.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "audited":
        raise ValueError("mesh must finish serial HDF5 geometry/material/P1 audit before solving")
    h5, parameters = inputs["mesh"], inputs["parameters"]
    actual_hashes = root_action(comm, lambda: dict(h5=sha256(h5), parameters=sha256(parameters)))
    if actual_hashes != metadata["input_sha256"]:
        raise ValueError("audited mesh inputs changed")
    mesh_hash = metadata["mesh_hash"]
    def default_output():
        from datetime import datetime, timezone
        from uuid import uuid4
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        return str(ROOT / "outputs" / "case1_block" / "cross_validation" / tag / args.stage)
    output = (args.output.resolve() if args.output else
              Path(root_action(comm, default_output)))
    data_directory = (args.data_root or ROOT / "data").resolve()
    if output == data_directory or data_directory in output.parents:
        raise ValueError("--output must be outside the input data directory")
    requested = parse_indices(args.indices)
    kernel_path = Path(__file__).with_name("dfom_sym2_kernel.py")
    def provenance():
        manifest = {str(path.relative_to(source)): sha256(path)
                    for path in sorted((source / "custEM").rglob("*.py"))}
        for path in sorted((ROOT / "core" / "supermor").glob("*.py")):
            manifest["supermor/" + path.name] = sha256(path)
        for path in (Path(__file__).resolve(), Path(__file__).with_name("validation_common.py"), kernel_path):
            manifest["runs/" + path.name] = sha256(path)
        digest = hash_bytes(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        return manifest, digest
    source_hashes, implementation_hash = root_action(comm, provenance)
    def pending(indices):
        return pending_indices(output, indices, mesh_hash, implementation_hash, args.stage)
    selected = root_action(comm, lambda: pending(requested))
    if not selected:
        def finish_resumed():
            merged = merge_responses(output, requested)
            write_json(output / "runstatus.json", dict(status="complete" if merged["required_complete"] else "partial",
                       resumed=True, mpi_ranks=comm.size, threads_per_rank=1, mesh_hash=mesh_hash, **identity()))
        root_action(comm, finish_resumed)
        return 0
    attempt = Path(root_action(comm, lambda: str(new_attempt(output))))
    stats = dict(status="running", stage=args.stage, mpi_ranks=comm.size, threads_per_rank=1,
                 p=1, indices=selected, mesh_hash=mesh_hash, attempt=str(attempt),
                 residual_definition="||K E+i omega M E-i omega q||_2/(omega ||q||_2)",
                 receiver_policy="background marker 1; containing non-ghost cell with smallest global index; rank breaks ties",
                 input_paths={key: str(value) for key, value in inputs.items()},
                 input_sha256=actual_hashes, source_sha256=source_hashes,
                 implementation_hash=implementation_hash, openblas_threads=1, **identity())

    def write_stats():
        def write():
            write_json(attempt / "runstatus.json", stats)
            write_json(output / "runstatus.json", stats)
        root_action(comm, write)

    write_stats()
    ops = None
    try:
        # custEM creates auxiliary mesh directories during model loading. Keep
        # those writes beside this attempt, while actual input hashes stay
        # bound to the immutable data files above. Only rank0 creates links.
        runtime = root_action(comm, lambda: {
            key: str(value) for key, value in runtime_mesh(inputs, attempt).items()})
        model = MOD("block_%s_%s" % (args.stage, attempt.name), name, "E_t", p=1,
                    ned_kind="1st", serial_ordering=True, debug_level=30, overwrite_results=False,
                    overwrite_mesh=False, file_format="h5", profiler=False, export_domains=False,
                    fs_type="ned", m_dir=runtime["m_dir"],
                    para_dir=str(Path(runtime["parameters"]).parent),
                    r_dir=str(attempt / "native_results"))
        rx = receivers(extra_depth=True)
        model.MP.rx, model.MP.n_rx = [rx.tolist()], 1
        configure_physics(model, FREQUENCIES[selected])
        stats["source_geometry_tolerance_m"] = float(model.FE.a_tol)
        actual = dict(vertices=model.FS.mesh.num_entities_global(0), cells=model.FS.mesh.num_entities_global(3),
                      field_dofs=model.FS.V.dim(), mixed_dofs=model.FS.M.dim(), scalar_dofs=model.FS.S.dim())
        if actual != metadata["dimensions"]:
            raise ValueError("distributed P1 dimensions differ from serial HDF5 audit")
        stats.update(dimensions=actual, software=dict(python=sys.version, dolfin=df.__version__,
                     petsc=PETSc.Sys.getVersion(), custEM=custEM.__version__))
        assembly_start = time.perf_counter()
        ops = assemble_mor(model, receiver_domain=1)
        stats["assembly_s"] = comm.allreduce(time.perf_counter()-assembly_start, op=MPI.MAX)
        stats["receiver_cell_global_indices"] = ops["receiver_cell_global_indices"]
        stats["frequencies"] = []
        write_stats()

        def emit(local_index, real, imag):
            index = selected[local_index]
            omega = 2*np.pi*FREQUENCIES[index]
            residual = independent_residual(ops, real, imag, omega)
            data = sample_receivers(ops, real, imag, omega)
            if not np.array_equal(data["coordinates"], rx):
                raise ValueError("native sampler changed coordinates")
            status = "complete" if np.isfinite(residual) and residual <= TOLERANCE else "failed"
            row = dict(index=index, frequency_hz=float(FREQUENCIES[index]), residual=residual, status=status)
            stats["frequencies"].append(row)
            def save():
                save_frequency(output, index, rx, data["E"], data["H"],
                    dict(**identity(), mesh_hash=mesh_hash, status=status, residual=residual,
                         mpi_ranks=comm.size, threads_per_rank=1, p=1, stage=args.stage,
                         attempt=str(attempt), units_E="V/m", units_H="A/m",
                         source_sha256=stats["source_sha256"], physical_config=physical_config(),
                         implementation_hash=implementation_hash,
                         source_geometry_tolerance_m=stats["source_geometry_tolerance_m"],
                         mor_options=stats.get("mor_options"), native_kernel_sha256=stats.get("native_kernel_sha256")))
            root_action(comm, save)
            write_stats()

        if args.stage == "sumor":
            options = dict(MOR_OPTIONS)
            while True:
                stats["mor_options"] = options
                write_stats()
                native = solve_split_u(ops, FREQUENCIES[selected], emit=emit, **options)
                root_action(comm, lambda: write_json(attempt / "solver_stats.json", native))
                if "exhausted %d basis columns" % options["max_basis"] not in str(native.get("failure", "")):
                    break
                options = {**options, "max_basis": options["max_basis"] * 2}
                attempt = Path(root_action(comm, lambda: str(new_attempt(output))))
                stats.update(attempt=str(attempt), frequencies=[])
                # Completed shards retain their accepted values; callbacks on
                # the enlarged basis only checkpoint still-unaccepted indices.
                selected = root_action(comm, lambda: pending(selected))
                if not selected:
                    break
        else:
            spec = importlib.util.spec_from_file_location("block_dfom_sym2", kernel_path)
            kernel = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernel)
            kernel.COMM, kernel.MPI = comm, MPI
            kernel.write = lambda path, data: root_action(comm, lambda: write_json(path, data))
            kernel.OPTIONS.update(tol=TOLERANCE, maximum_lu_corrections=3, mumps_icntl_14=50)
            stats["native_kernel_sha256"] = sha256(kernel_path)
            stats["native_route"] = "native_symmetric_mixed_ufl_mumps_sym2"
            native = kernel.solve_dfom(model, ops, FREQUENCIES[selected], emit, attempt, df, PETSc, np)
            root_action(comm, lambda: write_json(attempt / "solver_stats.json", native))
        stats["solver_status"] = native.get("status")
        remaining = root_action(comm, lambda: pending(requested))
        stats.update(status="complete" if not remaining else "partial", remaining_indices=remaining)
    except Exception as error:
        stats.update(status="failed", failure=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    finally:
        if ops is not None and ops.get("destroy") is not None:
            ops["destroy"]()
        import resource
        local = dict(rank=comm.rank, hostname=platform.node(), pid=os.getpid(),
                     affinity=sorted(os.sched_getaffinity(0)),
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     threads={key: os.environ.get(key) for key in THREAD_VARIABLES})
        resources = comm.gather(local, root=0)
        stats["wall_s"] = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
        stats["resources"] = resources
        root_action(comm, lambda: write_json(attempt / "resources.json", resources))
        stats["merge"] = root_action(comm, lambda: merge_responses(output, requested))
        write_stats()
    return 0 if stats["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="input data root (default: checkout/data)")
    parser.add_argument("--stage", default="sumor", choices=("dfom", "sumor"))
    parser.add_argument("--indices", help="four-frequency indices, default 0,1,2,3")
    parser.add_argument("--output", type=Path,
                        help="result directory, supports identity-checked resume")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    parse_indices(args.indices)
    if args.check_inputs:
        inputs = get_case_mesh("block_cross", data_root=args.data_root, require=True)
        metadata = json.loads(inputs["metadata"].read_text())
        actual = {"h5": sha256(inputs["mesh"]), "parameters": sha256(inputs["parameters"])}
        if actual != metadata["input_sha256"]:
            raise ValueError("audited mesh inputs changed")
        print(json.dumps(dict(status="passed", inputs={k: str(v) for k, v in inputs.items()},
                              dimensions=metadata["dimensions"], input_sha256=actual), indent=2))
        return 0
    try:
        if args.stage == "dfom":
            from petsc4py import PETSc
            configure_dfom_options(PETSc.Options())
        return run_solver(args)
    except Exception:
        if args.stage != "dfom":
            raise
        # The former DFOM entry aborted all ranks on an uncaught startup or
        # worker error. Preserve that behavior after merging its entry here.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
