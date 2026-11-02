#!/usr/bin/env python3
"""One MR3D method/mesh/repeat using release core and data-relative inputs.

The paper uses eight MPI ranks and one numerical thread per rank. Results,
including native custEM products, remain beneath the selected output folder.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "core", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from runs.common.mesh_inputs import get_case_mesh, runtime_mesh

FREQUENCIES = [i / 16 for i in range(2, 21)]
SUMOR_OPTIONS = dict(field_formulation="split_u", basis_strategy="residual_mr",
    tol=1e-6, inner_rtol=1e-6, max_basis=400, pole_every=20,
    pole_schedule="hold_first", direct=False, residual_correction=False,
    poles=[-4.646478196159213, -1.3603495231756633])
METHOD_OPTIONS = dict(SUMOR=SUMOR_OPTIONS,
    IFOM=dict(outer="fgmres", restart=100, max_it=2000, rtol=1e-6,
        preconditioner="PRESB", inner="CG/AMS", inner_rtol=1e-3, inner_max_it=300,
        zero_initial_guess_each_frequency=True, inner_true_residual_policy="diagnostic",
        final_maxwell_true_residual_target=1e-6),
    CPC=dict(steps=80, max_cycles=10, tol=1e-6, pole=-2.4836470664490253),
    DFOM=dict(solver="MUMPS", true_residual_target=1e-10,
        maximum_same_LU_corrections=3, retain_best_finite_residual=True))
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ.setdefault(_name, "1")
EXPECTED_CELLS = dict(zip([f"M{i}" for i in range(1, 6)],
    [385245, 2168870, 4780659, 6409790, 7877059]))
EXPECTED_REAL_DOF = dict(zip(EXPECTED_CELLS,
    [897990, 5042000, 11109832, 14894240, 18302584]))


def load_project_inputs(data_root, level, material_dir=None):
    """Resolve generated geometry and prepared material through separate roots."""
    import numpy as np
    from runs.case3_mr3d.mr3d_native import require_prepared

    data_root = Path(data_root).expanduser().resolve() if data_root else ROOT / "data"
    inputs = get_case_mesh("mr3d_" + level, data_root=data_root, require=True)
    prepared = (Path(material_dir).expanduser().resolve() if material_dir else
                data_root / "model_data" / "case3_mr3d")
    required = ("manifest.json", "axes.npz", "log_rho_h.npy", "log_rho_v.npy")
    missing = [str(prepared / name) for name in required if not (prepared / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing MR3D material inputs; run prepare_materials.py "
                                "or pass --material-dir: " + ", ".join(missing))
    material = require_prepared(prepared)
    if material.get("model") != "original":
        raise ValueError("The paper requires the original MR3D model, not the extended model")
    baseline = json.loads((inputs["prepared"] / "manifest.json").read_text())
    for key in ("model", "shape", "cell_widths_m", "reference_cube_transform", "translation_m"):
        if material.get(key) != baseline.get(key):
            raise ValueError("MR3D material metadata differs from the paper input: " + key)
    for axis in "xyz":
        actual_bounds = material.get("axis_bounds_m", {}).get(axis)
        expected_bounds = baseline.get("axis_bounds_m", {}).get(axis)
        if (actual_bounds is None or expected_bounds is None or
                np.shape(actual_bounds) != (2,) or
                not np.allclose(actual_bounds, expected_bounds, rtol=0., atol=1e-8)):
            raise ValueError("MR3D material coordinate bounds differ from the paper: " + axis)
    manifest = json.loads(inputs["metadata"].read_text())
    manifest["m_dir"] = str(inputs["m_dir"])
    manifest["mesh_name"] = inputs["mesh"].stem
    with np.load(inputs["survey"], allow_pickle=False) as survey:
        inline, broadside = np.array(survey["inline"]), np.array(survey["broadside"])
        endpoints = np.array(survey["source_endpoints"])
    return manifest, prepared, inline, broadside, endpoints


def input_identity(args):
    """Fingerprint the actual release inputs before the measured section."""
    inputs = get_case_mesh("mr3d_" + args.level, data_root=args.data_root, require=True)
    manifest, prepared, inline, broadside, endpoints = load_project_inputs(
        args.data_root, args.level, args.material_dir)
    files = {key: path for key, path in inputs.items() if path.is_file()}
    files.update({"material_" + name: prepared / name for name in
                  ("manifest.json", "axes.npz", "log_rho_h.npy", "log_rho_v.npy")})
    return {key: dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
            for key, path in files.items()}


def release_source_identity():
    import custEM
    import supermor
    from supermor import mor, operators, full_order, materials
    modules = [Path(__file__).resolve(), Path(__file__).with_name("mr3d_native.py"), *(
                   Path(module.__file__).resolve() for module in
                   (supermor, mor, operators, full_order, materials))]
    return dict(custem_file=str(Path(custEM.__file__).resolve()),
                supermor_file=str(Path(supermor.__file__).resolve()),
                files_sha256={str(path): sha256(path) for path in modules})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def on_root(comm, action):
    value, error = None, None
    if comm.rank == 0:
        try:
            value = action()
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    value, error = comm.bcast((value, error), root=0)
    if error is not None:
        raise RuntimeError(error)
    return value


def save_checkpoint(output, received, frequencies, inline, broadside, residuals):
    """Atomically replace receiver-only output, retaining the completed row indices."""
    import numpy as np
    rows = sorted(received)
    count = len(inline)
    arrays = {}
    for field in ("E", "H"):
        values = np.asarray([received[index][field] for index in rows], dtype=np.complex128)
        if not rows:
            values = np.empty((0, count+len(broadside), 3), dtype=np.complex128)
        arrays[field+"_inline"] = values[:, :count]
        arrays[field+"_broadside"] = values[:, count:]
    temporary = Path(output)/"receivers.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, inline=inline, broadside=broadside,
            frequencies_hz=np.asarray([frequencies[index] for index in rows]),
            completed_indices=np.asarray(rows, dtype=np.int64),
            independent_residuals=np.asarray([residuals[index] for index in rows]),
            units_E="V/m", units_H="A/m", coordinates_units="m",
            time_convention="exp(+i omega t)")
    temporary.replace(Path(output)/"receivers.npz")
    atomic_json(Path(output)/"progress.json", dict(status="running", total_frequencies=len(frequencies),
        completed_indices=rows, completed_frequencies=[frequencies[index] for index in rows],
        independent_residuals=[residuals[index] for index in rows]))


def independent_residual(ops, real, imag, omega):
    """Audit the original distributed K/M/q, independently of reduced QR residuals."""
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
        if denominator <= 0:
            raise ValueError("MR3D source vector is zero")
        return float(math.sqrt(squared) / denominator)
    finally:
        residual.destroy()
        work.destroy()


def native_route(ops, frequencies, method, emit):
    """Dispatch only to the frozen production routes and their fixed tolerances."""
    from supermor import mor
    from supermor.full_order import full_order
    if method == "SUMOR":
        return mor.solve_split_u(ops, frequencies, emit=emit, **SUMOR_OPTIONS)
    if method == "CPC":
        pole = -2*math.pi*math.sqrt(min(frequencies)*max(frequencies))
        if not math.isclose(pole, METHOD_OPTIONS["CPC"]["pole"], rel_tol=1e-15):
            raise ValueError("CPC requires the complete frozen frequency band")
        return mor.solve_cpc(ops, frequencies, emit=emit, steps=80, max_cycles=10, tol=1e-6)
    return full_order(ops, frequencies, emit, direct=method == "DFOM")


def validate_native_stats(method, native):
    """Bind native route and preserve finite per-RHS split-u Poisson evidence."""
    expected = dict(SUMOR="split_u", CPC="cpc", IFOM="IFOM", DFOM="DFOM")
    if native.get("route") != expected[method] or native.get("status") != "converged":
        raise RuntimeError(method+" native route/status acceptance failed")
    if method != "SUMOR":
        return
    rows = [row for row in native.get("linear_solves", []) if row.get("name") == "poisson"]
    # The release core records every Poisson RHS but does not emit the old
    # driver's aggregate field. Derive it from the actual per-RHS audits.
    maximum = max((row.get("relative_residual", math.inf) for row in rows), default=math.inf)
    native["poisson_maximum_true_residual"] = maximum
    if (native.get("residual_verified") is not True or native.get("test_direct_inverse") is not False
            or not math.isfinite(maximum) or not 0 <= maximum <= 1e-10
            or not rows or len(rows) != native.get("counts", {}).get("poisson_rhs")):
        raise RuntimeError("SUMOR split-u/Poisson final acceptance failed")
    for row in rows:
        residual = row.get("relative_residual", math.inf)
        if (not math.isfinite(residual) or not 0 <= residual <= 1e-10
                or row.get("rtol") != 1e-10 or row.get("max_it") != 300
                or row.get("direct") is not False or row.get("require_true_residual") is not True
                or row.get("reason", 0) <= 0):
            raise RuntimeError("SUMOR Poisson per-RHS evidence failed")


def solve_native_batch(model, frequencies, method, inline, broadside, output, stats, comm):
    import numpy as np
    from mpi4py import MPI
    from supermor import mor
    from supermor.operators import assemble_mor, sample_receivers
    from runs.case2_layered_sphere.run_methods_mpi import MumpsSetupObserver

    phases = stats["phase_time_s"]
    received, residuals = {}, {}
    stats.update(completed_indices=[], completed_frequencies=[], independent_residuals=[],
                 max_independent_residual=None, finite_receivers=False,
                 assembly_calls=0, solve_calls=0)
    tick = time.perf_counter()
    ops = assemble_mor(model)
    stats["assembly_calls"] = 1
    phases["assembly"] = time.perf_counter()-tick
    phases.update(independent_residual=0., receiver_sampling=0., checkpoint=0.)
    observer = MumpsSetupObserver(mor) if method in ("CPC", "DFOM") else None
    if observer is not None:
        stats["mumps_observer"] = observer.provenance
        stats["mumps_diagnostics"] = observer.records
    try:
        def emit_frequency(index, real, imag):
            index = int(index)
            if index in received:
                raise RuntimeError(f"Duplicate emitted frequency {index}")
            omega = 2*np.pi*frequencies[index]
            tick = time.perf_counter()
            relative = independent_residual(ops, real, imag, omega)
            phases["independent_residual"] += time.perf_counter()-tick
            if not np.isfinite(relative) or relative > 1e-6:
                raise RuntimeError(f"Independent residual {relative:.9g} exceeds 1e-6 at {frequencies[index]} Hz")
            tick = time.perf_counter()
            data = sample_receivers(ops, real, imag, omega)
            phases["receiver_sampling"] += time.perf_counter()-tick
            expected = np.vstack((inline, broadside))
            valid = (np.array_equal(data["coordinates"], expected)
                and all(data[field].shape == (len(expected), 3)
                        and np.isfinite(data[field]).all() for field in ("E", "H")))
            if not comm.allreduce(bool(valid), op=MPI.LAND):
                raise RuntimeError("Native receiver coordinates/shape/finiteness check failed")
            residuals[index] = relative
            received[index] = {field: data[field] for field in ("E", "H")}
            rows = sorted(received)
            stats.update(completed_indices=rows,
                completed_frequencies=[frequencies[row] for row in rows],
                independent_residuals=[residuals[row] for row in rows],
                max_independent_residual=max(residuals.values()), finite_receivers=True)
            tick = time.perf_counter()
            on_root(comm, lambda: save_checkpoint(output, received, frequencies,
                                                  inline, broadside, residuals))
            phases["checkpoint"] += time.perf_counter()-tick
            if comm.rank == 0:
                print("MR3D_FREQUENCY " + json.dumps(dict(method=method, index=index,
                    frequency_hz=frequencies[index], independent_residual=relative,
                    completed=len(received))), flush=True)

        def emit(index, real, imag):
            try:
                emit_frequency(index, real, imag)
            except Exception:
                print(f"MR3D_CALLBACK_FAILED rank={comm.rank}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                comm.Abort(1)
                raise

        tick = time.perf_counter()
        stats["solve_calls"] = 1
        native = native_route(ops, frequencies, method, emit)
        phases["solver"] = time.perf_counter()-tick-sum(phases[key] for key in
            ("independent_residual", "receiver_sampling", "checkpoint"))
        stats["mor_stats"] = stats["solver_stats"] = native
        if method == "DFOM":
            for key in ("reference_status", "reference_target", "reference_accepted_indices",
                        "unverified_reference_indices"):
                stats[key] = native[key]
        if (native.get("status") != "converged"
                or sorted(received) != list(range(len(frequencies)))):
            raise RuntimeError("Native final acceptance failed: " + str(native.get("failure")))
        validate_native_stats(method, native)

        stats["dynamic_basis_count"] = native.get("max_basis", 0)
    finally:
        tick = time.perf_counter()
        if observer is not None:
            observer.restore()
        ops["destroy"]()
        phases["cleanup"] = time.perf_counter()-tick


def process_identity(comm):
    """Use Linux creation time with the same clock origin as psutil."""
    pid = os.getpid()
    process_fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    start_ticks = int(process_fields[19])
    boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime "))
    affinity = sorted(os.sched_getaffinity(0))
    topology = Path(f"/sys/devices/system/cpu/cpu{affinity[0]}/topology")
    return dict(rank=comm.rank, pid=pid,
        create_time=boot + start_ticks / os.sysconf("SC_CLK_TCK"),
        affinity=affinity, hostname=platform.node(),
        socket=int((topology/"physical_package_id").read_text()),
        physical_core=int((topology/"core_id").read_text()))


def numerical_runtimes():
    """Set/query loaded numerical libraries; launcher thread settings are also required."""
    paths = {Path(line.split()[-1]).resolve()
             for line in Path("/proc/self/maps").read_text().splitlines()
             if "/" in line and any(name in line for name in
                ("libopenblas", "libmkl_rt", "libgomp", "libomp.so"))}
    records = []
    for path in sorted(paths):
        library = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
        if "libopenblas" in path.name:
            setter, getter = library.openblas_set_num_threads, library.openblas_get_num_threads
        elif "libmkl_rt" in path.name:
            setter, getter = library.MKL_Set_Num_Threads, library.MKL_Get_Max_Threads
        else:
            setter, getter = library.omp_set_num_threads, library.omp_get_max_threads
        setter.argtypes, getter.restype = [ctypes.c_int], ctypes.c_int
        setter(1)
        count = int(getter())
        if count != 1:
            raise ValueError(f"Numerical runtime {path} has {count} threads")
        records.append(dict(library=str(path), num_threads=count))
    if not records:
        raise ValueError("No supported loaded numerical runtime could be validated")
    return records


def create_native_model(manifest, output, method, level, repeat, comm):
    from custEM.core import MOD
    native_results = Path(output)/"native_results"
    on_root(comm, lambda: native_results.mkdir(exist_ok=False))
    return MOD(f"{method.lower()}_{level}_repeat_{repeat:02d}", manifest["mesh_name"],
        "E_t", p=1, m_dir=str(manifest["m_dir"]), r_dir=str(native_results),
        file_format="h5", overwrite_results=True, overwrite_mesh=False,
        debug_level=30, export_domains=False)


def run(args, comm):
    import numpy as np
    import dolfin as df
    from mpi4py import MPI
    from petsc4py import PETSc
    import custEM
    from runs.case3_mr3d.mr3d_native import MU0, PreparedConductivity, source_current
    from supermor.materials import assign_native_vti

    output = args.output.resolve()
    source = ROOT
    def prepare_output():
        output.mkdir(parents=True, exist_ok=True)
        if (output / "stats.json").exists() or (output / "native_results").exists():
            raise FileExistsError("Output already contains a run; choose a new --output: " + str(output))
    identity = on_root(comm, lambda: input_identity(args))
    on_root(comm, prepare_output)
    source_identity = release_source_identity()
    stats = dict(schema_version=1, status="failed", method=args.method, level=args.level,
        repeat=args.repeat, method_label={"SUMOR": "SUPERMOR", "CPC": "RK"}.get(args.method, args.method),
        mpi_ranks=comm.size, threads_per_rank=1,
        real_block_dof=None, complex_dof=None, frequencies_hz=FREQUENCIES,
        completed_indices=[], completed_frequencies=[], independent_residuals=[],
        max_independent_residual=None, finite_receivers=False, wall_time_s=None,
        end_to_end_wall_s=None, phase_time_s={}, mor_stats={}, options=METHOD_OPTIONS[args.method],
        time_metric="end_to_end_wall", failure="Run did not finish; inspect launcher log",
        source_root=str(source), source_identity=source_identity,
        input_files=identity, input_sha256={key: row["sha256"] for key, row in identity.items()},
        measurement_profile=args.measurement_profile,
        paper_configuration=dict(ranks=8, threads_per_rank=1, model="original"),
        historical_results_reused=False,
        residual_definition="||i*omega*q-K*E-i*omega*M*E||_2/(omega*||q||_2)",
        timing_policy="barrier after source/runtime/binding validation; before load_project_inputs through E/H and output; maximum rank wall")
    on_root(comm, lambda: atomic_json(output/"stats.json", stats))
    started = None
    old_cwd, old_hook = Path.cwd(), sys.excepthook
    try:
        if comm.size != 8:
            raise ValueError("Exactly eight MPI ranks are required")
        from supermor import mor
        if not Path(mor.__file__).resolve().is_relative_to(ROOT / "core" / "supermor"):
            raise ValueError("MR3D worker must load the release core from this checkout")
        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise ValueError("A real PETSc build is required")
        for name in THREAD_VARIABLES:
            if os.environ.get(name) != "1":
                raise ValueError(f"{name} must equal 1")
        stats["threadpools"] = numerical_runtimes()
        ranks = comm.gather(process_identity(comm), root=0)
        on_root(comm, lambda: atomic_json(output/"rank_pids.json", dict(
            mpi_ranks=comm.size, threads_per_rank=1, ranks=ranks)))
        def check_hardware():
            cores = {(row["hostname"], row["socket"], row["physical_core"]) for row in ranks}
            if len(cores) != 8 or any(len(row["affinity"]) != 1 for row in ranks):
                raise ValueError("Ranks must bind to eight distinct physical cores")
        on_root(comm, check_hardware)
        stats["custem_file"] = custEM.__file__
        df.set_log_level(40)
        comm.Barrier()
        started = time.perf_counter()
        tick = started
        manifest, prepared, inline, broadside, endpoints = load_project_inputs(
            args.data_root, args.level, args.material_dir)
        stats["input_workdir"] = str(prepared.parent)
        for line in (inline, broadside):
            if line.shape != (204, 3) or not np.isfinite(line).all():
                raise ValueError("Each survey line must contain 204 finite XYZ points")
        current = source_current(endpoints, 1.)
        if not (np.allclose(endpoints[1]-endpoints[0], [10.,0.,0.], rtol=0., atol=1e-10)
                and math.isclose(current, .1, rel_tol=0., abs_tol=1e-12)):
            raise ValueError("Source must be the frozen +x-directed 10 m line with 1 A m moment")
        stats.update(source_current_A=current, source_moment_Am=1., source_endpoints=endpoints)
        sampler = PreparedConductivity(prepared)
        original_paths = get_case_mesh("mr3d_" + args.level, data_root=args.data_root, require=True)
        view = on_root(comm, lambda: runtime_mesh(original_paths, output))
        manifest["m_dir"] = str(view["m_dir"])
        stats["runtime_mesh_directory"] = str(view["m_dir"])
        stats["phase_time_s"]["model_load"] = time.perf_counter()-tick
        tick = time.perf_counter()
        os.chdir(output)
        model = create_native_model(manifest, output, args.method, args.level, args.repeat, comm)
        model.MP.update_model_parameters(frequencies=FREQUENCIES, current=current,
            sigma_air=1e-8, sigma_ground=[1.]*(model.FS.DOM.n_domains-1), mu=MU0)
        actual = dict(cells=int(model.FS.mesh.num_entities_global(3)),
                      complex_dof=int(model.FS.V.dim()), real_block_dof=int(model.FS.M.dim()))
        for key, value in actual.items():
            if key in manifest and value != int(manifest[key]):
                raise ValueError(f"Mesh manifest {key} mismatch")
        if (actual["real_block_dof"] != EXPECTED_REAL_DOF[args.level]
                or actual["cells"] != EXPECTED_CELLS[args.level]):
            raise ValueError("Actual mesh cells/DOFs do not match the fixed benchmark level")
        stats.update(actual)
        model.MP.rx, model.MP.n_rx = [inline.tolist(), broadside.tolist()], 2
        model.MP.tx, model.MP.n_tx = [endpoints.tolist()], 1
        model.MP.topo, model.MP.grounding = "None", [True]
        model.FE.build_var_form(quasi_static=True, bc=None)
        stats["phase_time_s"]["mesh_and_source_setup"] = time.perf_counter()-tick
        tick = time.perf_counter()
        assign_native_vti(model, sampler)
        stats["phase_time_s"]["vti_assignment"] = time.perf_counter()-tick
        solve_native_batch(model, FREQUENCIES, args.method, inline, broadside, output, stats, comm)
        stats.update(status="passed", failure=None)
    except (Exception, SystemExit) as error:
        stats.update(status="failed", failure=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc())
        if comm.rank == 0:
            if started is not None:
                stats["wall_time_s"] = time.perf_counter()-started
            atomic_json(output/"stats.json", stats)
        print(f"MR3D_FAILED rank={comm.rank}: {stats['failure']}", flush=True)
        comm.Abort(1)
        return 1
    finally:
        os.chdir(old_cwd)
        sys.excepthook = old_hook
    on_root(comm, lambda: atomic_json(output/"stats.json", stats))
    on_root(comm, lambda: atomic_json(output/"progress.json", dict(status=stats["status"],
        total_frequencies=19, completed_indices=stats["completed_indices"],
        completed_frequencies=stats["completed_frequencies"],
        independent_residuals=stats["independent_residuals"])))
    comm.Barrier()
    stats["wall_time_s"] = stats["end_to_end_wall_s"] = comm.allreduce(
        time.perf_counter()-started, op=MPI.MAX)
    stats["phase_time_s"] = {key: comm.allreduce(value, op=MPI.MAX)
                              for key, value in stats["phase_time_s"].items()}
    on_root(comm, lambda: atomic_json(output/"stats.json", stats))
    if comm.rank == 0:
        print("MR3D_FINISHED " + json.dumps(dict(status=stats["status"],
            wall_time_s=stats["wall_time_s"], real_block_dof=stats["real_block_dof"],
            max_independent_residual=stats["max_independent_residual"])), flush=True)
    return 0


def method_name(value):
    value = value.upper()
    value = {"SUPERMOR": "SUMOR", "RK": "CPC"}.get(value, value)
    if value not in METHOD_OPTIONS:
        raise argparse.ArgumentTypeError("Choose SUPERMOR, IFOM, RK, or DFOM")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=method_name, required=True)
    parser.add_argument("--level", choices=list(EXPECTED_CELLS), required=True)
    parser.add_argument("--repeat", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data",
                        help="Input data root containing mesh_data and model_data")
    parser.add_argument("--material-dir", type=Path,
                        help="Prepared axes/log-resistivity directory; survey stays with the mesh inputs")
    parser.add_argument("--output", type=Path,
                        help="New run directory (default: outputs/case3_mr3d/mesh_scaling/LEVEL/METHOD/run_TIMESTAMP)")
    parser.add_argument("--measurement-profile", choices=("batch_r03", "fat09", "later"), default="later")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate input availability without importing FEM or starting a solve")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    if args.method in ("CPC", "DFOM") and args.repeat != 1:
        parser.error("RK and DFOM have exactly one repeat in the paper")
    if args.check_inputs:
        try:
            manifest, prepared, inline, broadside, endpoints = load_project_inputs(
                args.data_root, args.level, args.material_dir)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(dict(status="inputs_available", mesh=manifest["mesh_name"],
            material_dir=str(prepared), inline_shape=list(inline.shape),
            broadside_shape=list(broadside.shape), source_endpoints=endpoints.tolist()), indent=2))
        return 0
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") if comm.rank == 0 else None
        stamp = comm.bcast(stamp, root=0)
        args.output = ROOT / "outputs" / "case3_mr3d" / "mesh_scaling" / args.level / args.method / ("run_" + stamp)
    args.output = args.output.expanduser().resolve()
    input_roots = [args.data_root, ROOT / "data"]
    if args.material_dir:
        input_roots.append(args.material_dir.expanduser().resolve())
    if any(args.output.is_relative_to(path) for path in input_roots):
        parser.error("Run output must be outside input data directories")
    return run(args, comm)


if __name__ == "__main__":
    sys.exit(main())
