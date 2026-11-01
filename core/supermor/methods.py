"""Common operator-level interface to the four paper methods.

SUPERMOR and RK use the unchanged native MOR kernels. DFOM and IFOM use
our production MPI block layout and original-matrix residual checks.
"""
from copy import deepcopy

from . import mor
from .full_order import full_order


SUPERMOR_OPTIONS = dict(
    poles=[-2274.7560643370834, -2.221441469079183],
    pole_every=20, pole_schedule="hold_first", max_basis=400,
    basis_strategy="residual_mr", field_formulation="split_u",
    inner_rtol=1e-6, tol=1e-6, direct=False,
    residual_correction=False,
)
RK_OPTIONS = dict(steps=80, max_cycles=10, tol=1e-6)

__all__ = ["solve", "solve_model", "SUPERMOR_OPTIONS", "RK_OPTIONS"]


def _method_name(method):
    if not isinstance(method, str):
        raise ValueError("method must be DFOM, IFOM, RK, or SUPERMOR")
    selected = method.strip().upper()
    selected = {"CPC": "RK", "SUMOR": "SUPERMOR"}.get(selected, selected)
    if selected not in ("DFOM", "IFOM", "RK", "SUPERMOR"):
        raise ValueError("Unknown method: " + method)
    return selected


def solve(ops, frequencies, method="SUPERMOR", *, emit=None, **options):
    """Solve assembled operators with DFOM, IFOM, RK, or SUPERMOR.

    ``CPC`` aliases RK and ``SUMOR`` aliases SUPERMOR; names are case
    insensitive. ``emit(index, real, imag)`` receives each frequency field.
    The returned native statistics retain their original route names.

    SUPERMOR defaults match the P1 terrain/sphere production configuration;
    callers may override any ``solve_split_u`` option, including its poles.
    RK defaults match the production 80-column, ten-cycle configuration and
    accept the native ``solve_cpc`` options. DFOM and IFOM retain their fixed
    production tolerances and accept no extra options. Their equation gate
    and DFOM's strict reference qualification remain separate.
    """
    selected = _method_name(method)
    if selected == "IFOM" and ops.get("fe_degree", 1) == 2:
        raise NotImplementedError("P2 IFOM is not supported by the paper MPI solver")
    if selected == "SUPERMOR":
        settings = deepcopy(SUPERMOR_OPTIONS)
        settings.update(options)
        return mor.solve_split_u(ops, frequencies, emit=emit, **settings)
    if selected == "RK":
        settings = dict(RK_OPTIONS)
        settings.update(options)
        return mor.solve_cpc(ops, frequencies, emit=emit, **settings)
    if selected in ("DFOM", "IFOM"):
        if options:
            raise TypeError(selected + " uses the fixed production options; unexpected: "
                            + ", ".join(sorted(options)))
        callback = emit if emit is not None else (lambda index, real, imag: None)
        return full_order(ops, frequencies, callback, direct=selected == "DFOM")
    raise ValueError("Unknown method: " + method)


def solve_model(model, method="SUPERMOR", *, receiver_domain=None, **options):
    """Assemble and solve a custEM model without changing its implementation.

    Initialize the upstream total-E model with ``iterative=False``, fixed real
    current, quasi-static physics and natural boundaries. Assign any native
    VTI material *after* upstream ``build_var_form`` and before this call.
    No legacy postprocessing or upstream solver is invoked. P2 model solves
    retain the existing SUPERMOR residual-MR scope; DFOM/RK remain available
    through the operator-level interface.

    Return the native statistics, also stored as ``model.mor_stats``, including
    per-frequency E/H receiver rows. The last emitted field stays in ``FS.U``.
    A numerical failure records its statistics before raising RuntimeError.
    """
    import numpy as np

    from .operators import assemble_mor, sample_receivers, store_field

    selected = _method_name(method)
    if model.FE.bc is not None:
        raise ValueError("Model solves require natural boundary conditions (bc=None)")
    if model.FS.V.ufl_element().degree() == 2:
        if selected != "SUPERMOR":
            raise NotImplementedError("P2 model solves support SUPERMOR only")
        if (options.get("basis_strategy", "residual_mr") != "residual_mr"
                or options.get("residual_correction", False)):
            raise ValueError("P2 requires residual_mr without residual_correction")
    frequencies = np.asarray(model.MP.frequencies, dtype=float)
    ops = assemble_mor(model, receiver_domain=receiver_domain)
    rows = []

    def emit(index, real, imag):
        store_field(model, real, imag)
        rows.append(dict(index=int(index), frequency=float(frequencies[index]),
                         data=sample_receivers(ops, real, imag,
                                               2 * np.pi * frequencies[index])))

    try:
        model.mor_stats = solve(ops, frequencies, method=selected, emit=emit, **options)
        model.mor_stats["receivers"] = rows
        if model.mor_stats["status"] != "converged":
            raise RuntimeError(model.mor_stats["failure"])
        return model.mor_stats
    finally:
        ops["destroy"]()
