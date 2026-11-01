"""Analytic P2 source/receiver checks, also runnable with two MPI ranks."""
from types import SimpleNamespace

import dolfin as df
import numpy as np
import pytest
from mpi4py import MPI

from supermor.operators import assemble_mor, sample_receivers


def p2_model(n=2):
    mesh = df.UnitCubeMesh(n, n, n)
    V = df.FunctionSpace(mesh, "N1curl", 2)
    S = df.FunctionSpace(mesh, "CG", 2)
    return SimpleNamespace(
        FS=SimpleNamespace(mesh=mesh, V=V, S=S, p=2, test_mode=True),
        FE=SimpleNamespace(quasi_static=True, ip=False, n_tx=1, bc=None,
                           s_type="path", tx=[[[0., 0., 0.], [1., 1., 1.]]],
                           grounding=True, a_tol=1e-8),
        MP=SimpleNamespace(approach="E_t", file_format="h5", frequencies=[1.],
                           currents=[[1.]], sigma_func=df.Constant(1.),
                           mu_inv_func=df.Constant(1.), rx=[]))


def _field_vector(model, components, degree):
    field = df.interpolate(df.Expression(components, degree=degree), model.FS.V)
    return field, df.as_backend_type(field.vector()).vec()


@pytest.mark.parametrize("current", [1., -2.5])
def test_p2_line_functional_matches_polynomial_work_and_gradient_endpoints(current):
    model = p2_model()
    model.MP.currents = [[current]]
    ops = assemble_mor(model)
    work_field, work = _field_vector(
        model, ("x[0]+2*x[1]", "3*x[0]+x[2]", "x[1]-2*x[2]"), 1)
    potential = df.interpolate(df.Expression(
        "x[0]*x[0]+2*x[0]*x[1]+3*x[1]*x[2]+x[2]*x[2]", degree=2), model.FS.S)
    gradient = ops["G"].createVecLeft()
    try:
        # Along x=y=z=t, E dot dl = 6*t dt, and phi(B)-phi(A)=7.
        assert ops["q"].dot(work) == pytest.approx(-3. * current, abs=2e-12)
        ops["G"].mult(df.as_backend_type(potential.vector()).vec(), gradient)
        assert ops["q"].dot(gradient) == pytest.approx(-7. * current, abs=2e-12)
        assert ops["fe_degree"] == 2
        assert ops["ams_interpolation"].getSize() == (
            model.FS.V.dim(), 3 * model.FS.S.dim())
        assert ops["constants"] == []
    finally:
        gradient.destroy()
        ops["destroy"]()


@pytest.mark.parametrize("path,expected", [
    ([[0, 0, 0], [1, 1, 1], [0, 0, 0]], 0.),
    ([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]], -3.),
    ([[1, 1, 1], [0, 0, 0]], 3.),
])
def test_p2_source_repeated_and_reversed_segments_keep_oriented_multiplicity(path, expected):
    model = p2_model()
    model.FE.tx = [path]
    ops = assemble_mor(model)
    work_field, work = _field_vector(
        model, ("x[0]+2*x[1]", "3*x[0]+x[2]", "x[1]-2*x[2]"), 1)
    try:
        assert ops["q"].dot(work) == pytest.approx(expected, abs=2e-12)
    finally:
        ops["destroy"]()


def test_p2_closed_loop_has_analytic_circulation_and_unfitted_path_is_rejected():
    model = p2_model()
    model.FE.tx = [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]]
    ops = assemble_mor(model)
    field, vector = _field_vector(model, ("-x[1]/2", "x[0]/2", "0"), 1)
    try:
        assert ops["q"].dot(vector) == pytest.approx(-1., abs=2e-12)
    finally:
        ops["destroy"]()
    model.FE.tx = [[[0, 0, 0], [1, .25, 0]]]
    with pytest.raises(ValueError, match="complete mesh edges"):
        assemble_mor(model)


def test_p2_receivers_use_pointwise_linear_curl_with_cell_permeability():
    model = p2_model()
    model.MP.rx = [[[.45, .35, .1], [.4, .3, .05], [.95, .85, .6]]]
    scalar = df.FunctionSpace(model.FS.mesh, "DG", 0)
    model.MP.mu_inv_func = df.interpolate(
        df.Expression("x[2] < .5 ? 2. : 7.", degree=0), scalar)
    ops = assemble_mor(model)
    field, real = _field_vector(model, ("x[1]*x[2]", "-x[0]*x[2]", "0"), 2)
    imag = real.copy()
    imag.scale(2.)
    try:
        result = sample_receivers(ops, real, imag, omega=2.)
        x, y, z = result["coordinates"].T
        electric = np.column_stack((y*z, -x*z, np.zeros(len(x))))
        curl = np.column_stack((x, y, -2*z))
        mu = np.where(z < .5, 2., 7.)
        np.testing.assert_allclose(result["E"], (1+2j)*electric, atol=2e-12)
        np.testing.assert_allclose(result["H"], (-1+.5j)*mu[:, None]*curl,
                                   atol=2e-12)
        # These two points are deliberately in one tetrahedron: curl is not DG0.
        assert ops["receiver_cell_global_indices"][0] == ops["receiver_cell_global_indices"][1]
        assert not np.allclose(result["H"][0], result["H"][1])
        with pytest.raises(NotImplementedError, match="P2"):
            ops["compute_curl"]()
        with pytest.raises(NotImplementedError, match="P2"):
            ops["compute_mass_lower_bound"]()
    finally:
        imag.destroy()
        ops["destroy"]()


def test_p2_z0_receivers_use_ground_trace_for_normal_E_and_mu():
    model = p2_model()
    mesh = model.FS.mesh
    mesh.coordinates()[:, 2] -= .5
    markers = df.MeshFunction("size_t", mesh, 3)
    for cell in df.cells(mesh):
        markers[cell] = 2 if cell.midpoint().z() < 0 else 1
    model.FS.DOM = SimpleNamespace(domain_func=markers)
    model.FE.tx = [[[0, 0, -.5], [1, 0, -.5]]]
    model.MP.rx = [[[.35, .2, 0.], [.8, .65, 0.]]]
    scalar = df.FunctionSpace(mesh, "DG", 0)
    model.MP.mu_inv_func = df.interpolate(
        df.Expression("x[2] < 0 ? 5. : 2.", degree=0), scalar)
    ops = assemble_mor(model, receiver_domain=2)
    field, real = _field_vector(model,
        ("x[1]*x[2]", "-x[0]*x[2]", "x[2] < 0 ? 13. : 29."), 2)
    imag = real.duplicate()
    imag.set(0.)
    try:
        result = sample_receivers(ops, real, imag, omega=2.)
        np.testing.assert_array_equal(result["coordinates"], model.MP.rx[0])
        np.testing.assert_allclose(result["E"], [[0, 0, 13], [0, 0, 13]], atol=2e-11)
        x, y, z = result["coordinates"].T
        np.testing.assert_allclose(result["H"],
            2.5j*np.column_stack((x, y, np.zeros(len(x)))), atol=2e-11)
        chosen = ops["receiver_cell_global_indices"]
    finally:
        imag.destroy()
        ops["destroy"]()
    model.FS.V = df.FunctionSpace(mesh, "N1curl", 1)
    model.FS.S = df.FunctionSpace(mesh, "CG", 1)
    model.FS.p = 1
    p1 = assemble_mor(model, receiver_domain=2)
    try:
        assert p1["receiver_cell_global_indices"] == chosen
    finally:
        p1["destroy"]()


def test_p2_auxiliary_destroy_precedes_clear_and_handles_source_failure(monkeypatch):
    from supermor import mor_high_order
    original = mor_high_order.build_p2_auxiliary
    allocations, destructions, active = [], [], []

    def tracked(*args):
        auxiliary = original(*args)
        allocations.append(auxiliary)
        actual_destroy = auxiliary["destroy"]

        def destroy():
            if active:
                assert "G" in active[0]
            destructions.append(auxiliary)
            actual_destroy()

        auxiliary["destroy"] = destroy
        return auxiliary

    monkeypatch.setattr(mor_high_order, "build_p2_auxiliary", tracked)
    model = p2_model()
    ops = assemble_mor(model)
    active.append(ops)
    borrowed_owner = ops["_owners"][0]
    source = ops["q"]
    cleanup = ops["destroy"]
    cleanup()
    cleanup()
    active.clear()
    assert ops == {}
    assert len(destructions) == 1
    assert allocations[0]["G"].handle == allocations[0]["Pi"].handle == 0
    assert source.handle == 0
    assert df.as_backend_type(borrowed_owner).mat().handle != 0
    model.FE.tx = [[[0, 0, 0], [1, .25, 0]]]
    with pytest.raises(ValueError, match="complete mesh edges"):
        assemble_mor(model)
    assert len(destructions) == 2
    assert allocations[1]["G"].handle == allocations[1]["Pi"].handle == 0




@pytest.mark.parametrize("attribute,phase", [("edges", "candidate scan"),
                                           ("Edge", "coverage check")])
def test_p2_source_reports_one_rank_edge_scan_failure_collectively(monkeypatch, attribute, phase):
    model = p2_model()
    comm = model.FS.mesh.mpi_comm()
    original = getattr(df, attribute)

    def faulty_edges(*args, **kwargs):
        if comm.rank == 0:
            raise ArithmeticError("Injected P2 source-edge scan failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(df, attribute, faulty_edges)
    with pytest.raises(RuntimeError, match="P2 source %s failed.*Injected" % phase):
        assemble_mor(model)


