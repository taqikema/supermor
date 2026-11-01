"""Emit tiny-model numerical baselines in an independent Python/MPI process.

Use --legacy-code /path/to/snapshot/core/code for the archived implementation;
otherwise import the installed supermor package (or set PYTHONPATH=core).
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import dolfin as df
from mpi4py import MPI
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-code", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output prefix for .npz and .json files")
    args = parser.parse_args()
    if args.legacy_code:
        sys.path.insert(0, str(args.legacy_code.resolve()))
        from custEM.core.methods import solve
        from custEM.fem.materials import assign_native_vti
        from custEM.fem.mor_operators import assemble_mor, sample_receivers
    else:
        from supermor import solve, assemble_mor, assign_native_vti
        from supermor.operators import sample_receivers

    frequencies = [.1, 1., 100.]
    arrays, summaries = {}, {}
    comm = MPI.COMM_WORLD

    class VTI:
        def sample(self, points, markers, chunk_size):
            horizontal = 1. + points[:, 0]
            return horizontal, .25 * horizontal

    for degree in (1, 2):
        for material in ("scalar", "vti"):
            mesh = df.UnitCubeMesh(2, 2, 2)
            V = df.FunctionSpace(mesh, "N1curl", degree)
            S = df.FunctionSpace(mesh, "CG", degree)
            W = df.FunctionSpace(mesh, df.MixedElement([V.ufl_element()] * 2))
            markers = df.MeshFunction("size_t", mesh, 3, 0)
            model = SimpleNamespace(
                FS=SimpleNamespace(mesh=mesh, V=V, S=S, M=W, U=df.Function(W),
                    p=degree, test_mode=True, DOM=SimpleNamespace(domain_func=markers)),
                FE=SimpleNamespace(quasi_static=True, ip=False, n_tx=1, bc=None,
                    s_type="path", tx=[[[0., 0., 0.], [1., 0., 0.]]],
                    grounding=True, a_tol=1e-8),
                MP=SimpleNamespace(approach="E_t", file_format="h5",
                    frequencies=frequencies, currents=[[1.]] * len(frequencies),
                    sigma_func=df.Constant(1.), mu_inv_func=df.Constant(1.),
                    rx=[[[.2, .3, .4], [.5, .5, .5], [.73, .41, .38]]]))
            if material == "vti":
                assign_native_vti(model, VTI(), chunk_size=3)
            ops = assemble_mor(model)
            try:
                methods = ("DFOM", "IFOM", "RK", "SUPERMOR") if degree == 1 else (
                    "DFOM", "RK", "SUPERMOR")
                for method in methods:
                    key = f"p{degree}_{material}_{method}"
                    fields, electric, magnetic, residuals = [], [], [], []

                    def emit(index, real, imag):
                        omega = 2 * np.pi * frequencies[index]
                        local = real.getArray(readonly=True).copy() + 1j * imag.getArray(readonly=True)
                        fields.append(np.concatenate(comm.allgather(local)))
                        received = sample_receivers(ops, real, imag, omega)
                        electric.append(received["E"])
                        magnetic.append(received["H"])
                        r, t = real.duplicate(), real.duplicate()
                        try:
                            ops["K"].mult(real, r)
                            ops["M"].mult(imag, t)
                            r.axpy(-omega, t)
                            squared = r.norm() ** 2
                            ops["K"].mult(imag, r)
                            ops["M"].mult(real, t)
                            r.axpy(omega, t)
                            r.axpy(-omega, ops["q"])
                            residuals.append(float(np.sqrt(squared + r.norm() ** 2)
                                                   / (omega * ops["q"].norm())))
                        finally:
                            r.destroy()
                            t.destroy()

                    stats = solve(ops, frequencies, method=method, emit=emit)
                    if stats["status"] != "converged" or len(fields) != len(frequencies):
                        raise RuntimeError(f"{key}: {stats}")
                    if max(residuals) > 1.01e-6:
                        raise RuntimeError(f"{key} original residuals: {residuals}")
                    arrays[key + "_fields"] = np.asarray(fields)
                    arrays[key + "_E"] = np.asarray(electric)
                    arrays[key + "_H"] = np.asarray(magnetic)
                    arrays[key + "_residuals"] = np.asarray(residuals)
                    summaries[key] = {name: stats[name] for name in (
                        "route", "status", "counts", "basis_columns", "algorithm",
                        "basis_strategy", "field_formulation") if name in stats}
            finally:
                ops["destroy"]()
    if comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(args.output) + ".npz", **arrays)
        metadata = dict(mpi_ranks=comm.size, frequencies=frequencies,
                        legacy_code=str(args.legacy_code) if args.legacy_code else None,
                        summaries=summaries)
        Path(str(args.output) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{len(summaries)} method/material/degree cases saved to {args.output}")


if __name__ == "__main__":
    main()
