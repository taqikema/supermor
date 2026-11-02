"""One formal shifted-block strategy on the released mesh and numerical core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

import numpy as np

from mesh_gen.case1_block.strategy.strategy_contract import FREQUENCIES, receivers as expected_receivers
from runs.common.mesh_inputs import get_case_mesh, runtime_mesh
from runs.common.result_io import input_identity, save_receivers, write_json

SINGLE_POLES = [-71.08612701053386]
DUAL_POLES = [-2274.7560643370834, -2.221441469079183]


def strategy_options(name):
    if name == "reference":
        return {}
    if name not in ("split_u_pcg", "total_field_pcg", "total_field_lu", "single_pole", "dual_pole"):
        raise ValueError("Unknown strategy: " + name)
    options = dict(max_basis=400, pole_every=20, pole_schedule="hold_first",
        basis_strategy="residual_mr", residual_correction=False,
        field_formulation="total_field" if name.startswith("total_field") else "split_u",
        direct=name == "total_field_lu", poles=list(SINGLE_POLES if name == "single_pole" else DUAL_POLES))
    if not options["direct"]:
        options["inner_rtol"] = 1e-6
    return options


def check_inputs(data_root=None):
    paths = get_case_mesh("block_strategy", data_root=data_root)
    metadata = json.loads(paths["metadata"].read_text())
    parameters = json.loads(paths["parameters"].read_text())
    if metadata["cells"] != 309349 or metadata["edge_dofs"] != 360026:
        raise ValueError("Strategy metadata is not the paper mesh")
    if not np.array_equal(np.asarray(parameters["rx"][0]), expected_receivers()):
        raise ValueError("Expected the formal 3321 receivers")
    source = np.asarray(parameters["tx"][0])
    if source.shape != (21, 3) or not np.array_equal(source[[0, -1]], [[-2500, 0, 0], [-2000, 0, 0]]):
        raise ValueError("Wrong grounded finite source")
    if parameters["grounding"] != [True]:
        raise ValueError("Formal source must be grounded")
    return paths, metadata


def input_hashes(ops):
    # Original serial assembled-input identity from csem_mor_compare_methods.
    def digest(arrays):
        result = hashlib.sha256()
        for array in arrays:
            array = np.ascontiguousarray(array)
            result.update(str((array.shape, array.dtype.str)).encode())
            result.update(array.tobytes())
        return result.hexdigest()
    result = {key: digest((np.array(ops[key].getSize()),) + tuple(ops[key].getValuesCSR()))
              for key in ("K", "M")}
    result["q"] = digest((ops["q"].getArray(readonly=True),))
    result["receivers"] = digest((ops["rx"],))
    return result


def run(args):
    import dolfin as df
    import custEM
    from custEM.core import MOD
    import supermor
    from supermor import mor
    from supermor.operators import assemble_mor, sample_receivers

    if df.MPI.size(df.MPI.comm_world) != 1:
        raise ValueError("The paper block strategy fixes one MPI rank")
    df.set_log_level(40)
    paths, metadata = check_inputs(args.data_root)
    identity = input_identity(paths)
    source_identity = input_identity(dict(worker=Path(__file__),
        mor=Path(mor.__file__), operators=ROOT / "core/supermor/operators.py"))
    output = args.output.resolve()
    data_root = args.data_root.resolve() if args.data_root else ROOT / "data"
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError("Output must be outside the input data directory")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "stats.json").exists():
        raise FileExistsError("Existing numerical result: " + str(output))
    options = strategy_options(args.strategy)
    runtime = runtime_mesh(paths, output)
    write_json(output / "identity.json", dict(inputs=identity, strategy=args.strategy,
        options=options, frequencies_hz=FREQUENCIES, output=output, sources=source_identity))
    started = time.perf_counter()
    ops = model = None
    rows = {}
    fields = []
    stats = dict(status="failed", failure=None)
    assembly_seconds = None

    def emit(index, real, imag):
        rows[index] = sample_receivers(ops, real, imag, 2*np.pi*FREQUENCIES[index])
        if args.save_fields:
            np.savez(output / ("field_%02d.npz" % index),
                     real=real.getArray(readonly=True), imag=imag.getArray(readonly=True))
            fields.append(index)
        print(f"{args.strategy}: frequency {index+1}/20", flush=True)

    try:
        model = MOD("block_" + args.strategy, "csem_mor", "E_t", p=1, debug_level=30,
            overwrite_mesh=False, overwrite_results=True, file_format="h5",
            r_dir=str(output / "native_results"), m_dir=str(runtime["m_dir"]),
            para_dir=str(runtime["parameters"].parent), profiler=False)
        model.MP.update_model_parameters(frequencies=FREQUENCIES.tolist(), current=1.,
                                        sigma_air=1e-8, sigma_ground=[.01, 1.])
        model.FE.build_var_form(s_type="auto", quasi_static=True)
        ops = assemble_mor(model)
        if model.FS.mesh.num_cells() != 309349 or ops["q"].getSize() != 360026:
            raise ValueError("Loaded mesh dimensions differ from paper")
        if not np.array_equal(ops["rx"], expected_receivers()):
            raise ValueError("Assembled receiver order differs from paper")
        assembly_seconds = time.perf_counter() - started
        if args.strategy == "reference":
            stats = independent_reference(model, ops, FREQUENCIES, emit)
        else:
            stats = mor.solve_split_u(ops, FREQUENCIES, emit=emit, **options)
    except Exception as error:
        import traceback
        traceback.print_exc()
        stats.update(status="failed", failure=f"{type(error).__name__}: {error}")
    finally:
        save_receivers(output / "receivers.npz", FREQUENCIES, rows,
                       ops["rx"] if ops is not None else np.asarray(metadata["receiver_xyz"]))
        complete = time.perf_counter() - started
        stats.update(method="UFL_MUMPS" if args.strategy == "reference" else "SUPERMOR",
            strategy=args.strategy, frequencies=FREQUENCIES.tolist(), receiver_indices=sorted(rows),
            field_indices=fields, full_fields_saved=args.save_fields,
            complete_forward_s=complete, common_input_assembly_s=assembly_seconds,
            cells=model.FS.mesh.num_cells() if model is not None else None,
            edge_dofs=ops["q"].getSize() if ops is not None else None,
            real_P1_dofs=2*ops["q"].getSize() if ops is not None else None,
            mpi_ranks=1, threads=1, actual_inputs=identity, sources=source_identity,
            core_source=str(Path(supermor.__file__).resolve()),
            mor_source=str(Path(mor.__file__).resolve()),
            custem_source=str(Path(custEM.__file__).resolve()),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            field_output_policy="optional; paper serial workers saved all fields")
        try:
            stats["input_sha256"] = input_hashes(ops) if ops is not None else None
            write_json(output / "stats.json", stats)
        finally:
            if ops is not None:
                ops["destroy"]()
    if args.strategy == "reference":
        return 0 if len(rows) == 20 and max(r["residual"] for r in stats.get("per_frequency", [])) <= 1e-6 else 1
    return 0 if stats.get("status") == "converged" and len(rows) == 20 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("split_u_pcg", "total_field_pcg", "total_field_lu",
                        "single_pole", "dual_pole", "reference"), default="dual_pole")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if args.check_inputs:
        paths, metadata = check_inputs(args.data_root)
        print(json.dumps(dict(paths={k:str(v) for k,v in paths.items()}, cells=metadata["cells"],
                              real_dofs=metadata["mixed_dofs"], options=strategy_options(args.strategy)), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for a numerical worker; use run_strategy.py for an automatic run directory")
    return run(args)


# Reference routines retained from the formal independent UFL/MUMPS worker.
def refine_reference(initial, residual, correction, tol=1e-10):
    """At most three corrections through one LU; keep the best finite field."""
    best = np.array(initial, copy=True)
    best_rho, rhs = residual(best)
    history = [float(best_rho)]
    best_rho = best_rho if np.isfinite(best_rho) and np.isfinite(best).all() else np.inf
    errors, steps = [], 0
    for step in range(3):
        if best_rho <= tol:
            break
        steps += 1
        try:
            candidate = best+correction(rhs)
            rho, candidate_rhs = residual(candidate) if np.isfinite(candidate).all() else (np.inf, None)
            history.append(float(rho))
            if np.isfinite(rho) and rho < best_rho:
                best, best_rho, rhs = candidate.copy(), rho, candidate_rhs
        except Exception as error:
            errors.append('%s: %s' % (type(error).__name__, error))
            history.append(float('inf'))
    return best, dict(residual=float(best_rho), verified=bool(best_rho <= tol),
        refinement_residuals=history, refinement_steps=steps, refinement_failures=errors)

def independent_reference(model, ops, fs, emit):
    """The validation UFL mixed discretization with residual refinement on its LU."""
    import dolfin as df
    from petsc4py import PETSc
    from supermor import mor

    ur, ui = df.TrialFunctions(model.FS.M)
    vr, vi = df.TestFunctions(model.FS.M)
    sigma, mu = model.MP.sigma_func, model.MP.mu_inv_func
    qfield, zero = df.Function(model.FS.V), df.Function(model.FS.V)
    ops['q'].copy(df.as_backend_type(qfield.vector()).vec())
    rhs_field = df.Function(model.FS.M)
    to_mixed = df.FunctionAssigner(model.FS.M, [model.FS.V, model.FS.V])
    to_real = df.FunctionAssigner([model.FS.V, model.FS.V], model.FS.M)
    er, ei = df.Function(model.FS.V), df.Function(model.FS.V)
    field = model.FS.U[0] if isinstance(model.FS.U, list) else model.FS.U
    actions = [mor._Action(ops[key]) for key in ('K', 'M')]
    q = ops['q'].getArray(readonly=True)
    comm = ops['K'].comm.tompi4py()
    rows, factorizations, started = [], 0, time.perf_counter()
    try:
        for index, frequency in enumerate(fs):
            omega = 2*np.pi*frequency
            row = dict(index=index, frequency=float(frequency), verified=False,
                residual=float('inf'), refinement_residuals=[], refinement_steps=0, failure=None)
            solver = A = b = None
            best = np.zeros_like(field.vector().get_local())
            row['field_origin'] = 'zero_fallback_after_solve_error'
            try:
                a = (df.inner(mu*df.curl(ur), df.curl(vr))
                     - omega*df.inner(sigma*ui, vr)
                     + omega*df.inner(sigma*ur, vi)
                     + df.inner(mu*df.curl(ui), df.curl(vi)))*df.dx
                A = df.assemble(a)
                to_mixed.assign(rhs_field, [zero, qfield])
                b = rhs_field.vector().copy()
                b *= omega
                solver = PETSc.KSP().create(ops['K'].comm)
                solver.setOperators(df.as_backend_type(A).mat())
                solver.setType('preonly')
                solver.getPC().setType('lu')
                solver.getPC().setFactorSolverType('mumps')
                solver.setUp()
                factorizations += 1
                field.vector().zero()
                solver.solve(df.as_backend_type(b).vec(), df.as_backend_type(field.vector()).vec())
                initial = field.vector().get_local()
                if np.isfinite(initial).all():
                    best = initial.copy()
                    row['field_origin'] = 'independent_ufl_mumps'

                def residual(values):
                    field.vector().set_local(values)
                    field.vector().apply('insert')
                    r = b.copy()
                    A.mult(field.vector(), r)
                    r *= -1.
                    r.axpy(1., b)
                    ufl_rho = r.norm('l2')/max(b.norm('l2'), np.finfo(float).tiny)
                    to_real.assign([er, ei], field)
                    electric = er.vector().get_local()+1j*ei.vector().get_local()
                    unused, assembled, physical = mor._residual(comm, *actions,
                        q, electric, omega, np.linalg.norm(q))
                    row.update(ufl_residual=float(ufl_rho),
                               assembled_residual=float(assembled), physical_residual=float(physical))
                    return max(ufl_rho, assembled, physical), r.get_local()

                def correction(values):
                    rhs = b.copy()
                    rhs.set_local(values)
                    rhs.apply('insert')
                    delta = b.copy()
                    delta.zero()
                    solver.solve(df.as_backend_type(rhs).vec(), df.as_backend_type(delta).vec())
                    return delta.get_local()

                best, information = refine_reference(best, residual, correction)
                row.update(information)
                # Restore best even if later corrections were worse or nonfinite.
                residual(best)
                if not row['verified']:
                    row['failure'] = 'True relative residual exceeds 1e-10 after at most three LU corrections'
            except Exception as error:
                row['failure'] = '%s: %s' % (type(error).__name__, error)
            # Continue to later frequencies after an individual reference failure.
            if best is not None and np.isfinite(best).all():
                field.vector().set_local(best)
                field.vector().apply('insert')
                to_real.assign([er, ei], field)
                try:
                    emit(index, df.as_backend_type(er.vector()).vec(), df.as_backend_type(ei.vector()).vec())
                except Exception as error:
                    row.update(verified=False, failure='%s: %s' % (type(error).__name__, error))
            rows.append(row)
            print('reference frequency %d/%d: %s' % (index+1, len(fs),
                  'verified' if row['verified'] else 'unverified'), flush=True)
            if solver is not None:
                solver.destroy()
            del solver, A, b
    finally:
        for action in actions:
            action.destroy()
    verified = len(rows) == len(fs) and all(row['verified'] for row in rows)
    return dict(method='UFL_MUMPS', route='independent_mixed_ufl',
        status='converged' if verified else 'unverified', reference_verified=verified,
        failure=None if verified else 'Independent reference did not verify every frequency',
        per_frequency=rows, seconds=dict(total=time.perf_counter()-started),
        counts=dict(factorizations=factorizations, frequency_attempts=len(rows),
                    refinement_solves=sum(r['refinement_steps'] for r in rows)))


if __name__ == "__main__":
    raise SystemExit(main())
