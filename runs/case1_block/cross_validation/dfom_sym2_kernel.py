"""Isolated symmetric-indefinite MUMPS fallback kernel; not the production entrypoint.

Same original K/M/q equation, row-scaled into [K,-wM;-wM,-K], RHS [0,-wq].
No restart/resume policy and no production-freeze modifications are implemented here.
"""
import json
import time
import traceback
from mpi4py import MPI

COMM=MPI.COMM_WORLD
OPTIONS=dict(tol=1e-6,maximum_lu_corrections=3,mumps_icntl_14=50)


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def write(path, data):
    if COMM.rank == 0:
        def clean(value):
            if isinstance(value, dict):
                return {str(k):clean(v) for k,v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if hasattr(value, "tolist"):
                return clean(value.tolist())
            if isinstance(value, float) and not __import__("math").isfinite(value):
                return None
            return value
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+"\n")
        tmp.replace(path)


def independent_residual(ops, real, imag, omega):
    """Separate PETSc real/imag evaluation of the original assembled equation."""
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
        denominator = omega*ops["q"].norm()
        return float(squared**.5/denominator) if denominator else float(squared**.5)
    finally:
        residual.destroy()
        work.destroy()


def solve_dfom(model,ops,frequencies,emit,output,df,PETSc,np):
    """Use the established native mixed-UFL/MUMPS reference path, with the native q.

    The symmetric weak form matches native custEM E_t with system_it=False.
    Acceptance uses the independently evaluated original ops.K/M/q residual at 1e-6.
    Corrections reuse the same LU only when a measured residual exceeds this limit.
    """
    require(PETSc.Sys.hasExternalPackage("mumps"), "PETSc was built without MUMPS")
    ur,ui = df.TrialFunctions(model.FS.M)
    vr,vi = df.TestFunctions(model.FS.M)
    sigma,mu = model.MP.sigma_func,model.MP.mu_inv_func
    qfield,zero = df.Function(model.FS.V),df.Function(model.FS.V)
    ops["q"].copy(df.as_backend_type(qfield.vector()).vec())
    qfield.vector().apply("insert")
    rhs_field = df.Function(model.FS.M)
    to_mixed = df.FunctionAssigner(model.FS.M,[model.FS.V,model.FS.V])
    to_real = df.FunctionAssigner([model.FS.V,model.FS.V],model.FS.M)
    er,ei = df.Function(model.FS.V),df.Function(model.FS.V)
    field = model.FS.U[0] if isinstance(model.FS.U,list) else model.FS.U
    rows = []
    prefix = "hed_dfom_sym2_"
    option_key = prefix+"mat_mumps_icntl_14"
    options = PETSc.Options()
    options[option_key] = OPTIONS["mumps_icntl_14"]
    total = time.perf_counter()
    try:
        for index,frequency in enumerate(frequencies):
            omega = 2*np.pi*frequency
            row = dict(index=index,frequency_hz=float(frequency),status="running",
                       true_residual=None,mixed_matrix_residual=None,correction_residuals=[],
                       correction_solves=0,failure=None)
            solver = matrix = rhs = factor = None
            best = None
            started = time.perf_counter()
            if COMM.rank == 0:
                print("DFOM_FREQUENCY_START "+json.dumps(dict(index=index+1,total=len(frequencies),frequency_hz=frequency)),flush=True)
            write(output/"progress.json",dict(method="DFOM_MUMPS",phase="mixed_assembly",
                  current_frequency_hz=frequency,current_index=index+1,total_frequencies=len(frequencies),
                  completed_frequencies=len([r for r in rows if r["status"]=="converged"])))
            try:
                tick = time.perf_counter()
                form = (df.inner(mu*df.curl(ur),df.curl(vr))
                        -omega*df.inner(sigma*ui,vr)
                        -omega*df.inner(sigma*ur,vi)
                        -df.inner(mu*df.curl(ui),df.curl(vi)))*df.dx
                matrix = df.assemble(form)
                petsc_matrix = df.as_backend_type(matrix).mat()
                petsc_matrix.setOption(PETSc.Mat.Option.SYMMETRIC,True)
                petsc_matrix.setOption(PETSc.Mat.Option.SPD,False)
                to_mixed.assign(rhs_field,[zero,qfield])
                rhs = rhs_field.vector().copy()
                rhs *= -omega
                row["mixed_assembly_s"] = COMM.allreduce(time.perf_counter()-tick,op=MPI.MAX)
                solver = PETSc.KSP().create(ops["K"].comm)
                solver.setOptionsPrefix(prefix)
                solver.setOperators(df.as_backend_type(matrix).mat())
                solver.setType("preonly")
                solver.getPC().setType("cholesky")
                solver.getPC().setFactorSolverType("mumps")
                solver.setFromOptions()
                require(solver.getType()=="preonly" and solver.getPC().getType()=="cholesky",
                        "Expected MUMPS symmetric-indefinite LDLT via PC Cholesky, with SPD=False")
                write(output/"progress.json",dict(method="DFOM_MUMPS",phase="factorization",current_frequency_hz=frequency,
                      current_index=index+1,total_frequencies=len(frequencies)))
                if COMM.rank == 0:
                    print("DFOM_FACTOR_START "+str(frequency),flush=True)
                tick = time.perf_counter()
                solver.setUp()
                row["factorization_s"] = COMM.allreduce(time.perf_counter()-tick,op=MPI.MAX)
                try:
                    factor = solver.getPC().getFactorMatrix()
                    row["mumps_controls"] = {"icntl_14":int(factor.getMumpsIcntl(14)),
                                             "icntl_28":int(factor.getMumpsIcntl(28)),
                                             "icntl_29":int(factor.getMumpsIcntl(29)),
                                             "factorization_route":"MUMPS SYM=2 / LDLT (SPD=False)",
                                             "cntl_1":float(factor.getMumpsCntl(1))}
                except Exception as error:
                    row["mumps_controls"] = {"error":str(error)}
                if COMM.rank == 0:
                    print("DFOM_FACTOR_DONE "+json.dumps(dict(frequency_hz=frequency,seconds=row["factorization_s"])),flush=True)
                field.vector().zero()
                tick = time.perf_counter()
                solver.solve(df.as_backend_type(rhs).vec(),df.as_backend_type(field.vector()).vec())
                field.vector().apply("insert")
                row["initial_solve_s"] = COMM.allreduce(time.perf_counter()-tick,op=MPI.MAX)
                row["ksp_reason"] = int(solver.getConvergedReason())

                def residual():
                    field.vector().apply("insert")
                    to_real.assign([er,ei],field)
                    original = independent_residual(ops,df.as_backend_type(er.vector()).vec(),
                                                     df.as_backend_type(ei.vector()).vec(),omega)
                    remainder = rhs.copy()
                    matrix.mult(field.vector(),remainder)
                    remainder *= -1.
                    remainder.axpy(1.,rhs)
                    mixed = float(remainder.norm("l2")/rhs.norm("l2"))
                    finite = bool(COMM.allreduce(bool(np.isfinite(field.vector().get_local()).all()),
                                               op=MPI.LAND))
                    return max(original,mixed) if finite else float("inf"),original,mixed,remainder

                best_rho,original,mixed,remainder = residual()
                best = field.vector().copy()
                row["correction_residuals"].append(dict(original=original,mixed=mixed))
                correction_started = time.perf_counter()
                for correction in range(OPTIONS["maximum_lu_corrections"]):
                    if best_rho <= OPTIONS["tol"]:
                        break
                    delta = rhs.copy()
                    delta.zero()
                    solver.solve(df.as_backend_type(remainder).vec(),df.as_backend_type(delta).vec())
                    delta.apply("insert")
                    row["correction_solves"] += 1
                    field.vector().axpy(1.,delta)
                    rho,next_original,next_mixed,next_remainder = residual()
                    row["correction_residuals"].append(dict(original=next_original,mixed=next_mixed))
                    if np.isfinite(rho) and rho < best_rho:
                        best_rho,original,mixed,remainder = rho,next_original,next_mixed,next_remainder
                        best = field.vector().copy()
                    else:
                        field.vector().zero()
                        field.vector().axpy(1.,best)
                row["correction_s"] = COMM.allreduce(time.perf_counter()-correction_started,op=MPI.MAX)
                field.vector().zero()
                field.vector().axpy(1.,best)
                field.vector().apply("insert")
                to_real.assign([er,ei],field)
                row.update(true_residual=float(original),mixed_matrix_residual=float(mixed),
                           passed=bool(np.isfinite(best_rho) and best_rho<=OPTIONS["tol"]),
                           tolerance=OPTIONS["tol"])
                row["status"] = "converged" if row["passed"] else "unverified"
                if not row["passed"]:
                    row["failure"] = "Measured original/mixed residual exceeds 1e-6 after LU corrections"
                emit(index,df.as_backend_type(er.vector()).vec(),df.as_backend_type(ei.vector()).vec())
            except Exception as error:
                row.update(status="failed",failure=type(error).__name__+": "+str(error),traceback=traceback.format_exc())
            finally:
                # Release the borrowed factor Mat wrapper before the KSP/PC, so
                # the previous frequency's numeric LU/LDLT cannot outlive its job.
                if factor is not None:
                    factor.destroy()
                if solver is not None:
                    solver.destroy()
                del solver,matrix,rhs,best
            row["complete_frequency_s"] = COMM.allreduce(time.perf_counter()-started,op=MPI.MAX)
            rows.append(row)
            write(output/"frequency_stats.json",rows)
            if COMM.rank == 0:
                print("DFOM_FREQUENCY_FINISHED "+json.dumps(row),flush=True)
    finally:
        del options[option_key]
    verified = len(rows)==len(frequencies) and all(row.get("passed",False) for row in rows)
    return dict(method="DFOM_MUMPS",route="native_symmetric_mixed_ufl_mumps_sym2",status="converged" if verified else "unverified",
                residual_verified=verified,per_frequency=rows,seconds=dict(total=time.perf_counter()-total),
                counts=dict(factorizations=sum("factorization_s" in row for row in rows),
                            frequency_attempts=len(rows),correction_solves=sum(row["correction_solves"] for row in rows)),
                failure=None if verified else "DFOM did not pass every original/mixed 1e-6 residual check")
