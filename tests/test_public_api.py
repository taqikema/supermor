"""Official MOD/H5 integration with independent UFL and dense references."""
import json
from types import SimpleNamespace

import dolfin as df
import numpy as np
import pytest

from custEM.core import MOD
from supermor import methods
from supermor import operators
from supermor.materials import assign_native_vti
from supermor.operators import assemble_mor

from .fe_helpers import full_vector, independent_matrices, reference_receivers


class DomainVTI:
    def sample(self, points, markers, chunk_size):
        horizontal = np.where(markers == 1, 2., 1.)
        return horizontal, .25 * horizontal


def official_h5_model(path, degree, material):
    """Create a genuine upstream model, keeping all generated files in /tmp."""
    comm = df.MPI.comm_world
    meshes = path / "meshes"
    if comm.rank == 0:
        (meshes / "_h5").mkdir(parents=True)
        (meshes / "para").mkdir()
        parameters = dict(topo=None, centering=False, easting_shift=None,
            northing_shift=None, rotation=None, n_layers=1, layer_depths=[],
            zeroing=False, bathymetry=False, domains=[0, 1], n_domains=2,
            rx=[[[.22, .37, 0.], [.47, .22, 0.], [.73, .41, 0.]]],
            tx=[[[0., 0., 0.], [1., 0., 0.]]], grounding=[True],
            post_rotate=None, overwrite_markers=None)
        (meshes / "para" / "public_parameters.json").write_text(json.dumps(parameters))
    comm.barrier()
    mesh = df.BoxMesh(df.Point(0., 0., -1.), df.Point(1., 1., 1.), 2, 2, 2)
    domains = df.MeshFunction("size_t", mesh, 3, 0)
    for cell in df.cells(mesh):
        domains[cell] = int(cell.midpoint().z() < 0.)
    with df.HDF5File(comm, str(meshes / "_h5" / "public.h5"), "w") as handle:
        handle.write(mesh, "/mesh")
        handle.write(domains, "/domains")
    model = MOD("public_api", "public", "E_t", p=degree, iterative=False,
        debug_level=50, profiler=False, file_format="h5", overwrite_mesh=False,
        overwrite_results=True, m_dir=str(meshes), para_dir=str(meshes / "para"),
        r_dir=str(path / "results"))
    assert model.FS.mesh.num_entities_global(3) == 48
    assert model.FS.V.ufl_element().degree() == degree
    model.MP.update_model_parameters(frequencies=[.1, 1., 100.], current=1.,
        sigma_air=1., sigma_ground=[1.])
    model.MP.mu_inv_func, model.MP.mu = df.Constant(1.), 1.
    # Upstream builds only scalar placeholders; our assignment follows it.
    model.FE.build_var_form(s_type="auto", quasi_static=True)
    if material == "vti":
        assign_native_vti(model, DomainVTI(), chunk_size=7)
    return model


@pytest.mark.parametrize("degree,method", [
    (1, "DFOM"), (1, "IFOM"), (1, "RK"), (1, "SUPERMOR"), (2, "SUPERMOR"),
])
@pytest.mark.parametrize("material", ["scalar", "vti"])
def test_real_h5_model_loads_and_solves_via_public_entries(
        mpi_tmp_path, monkeypatch, degree, method, material):
    model = official_h5_model(mpi_tmp_path, degree, material)
    frequencies = model.MP.frequencies
    fields = {}
    original_solve = methods.solve

    def observe_solve(ops, values, method="SUPERMOR", *, emit=None, **options):
        def observe(index, real, imag):
            fields[index] = full_vector(real) + 1j * full_vector(imag)
            if emit is not None:
                emit(index, real, imag)
        return original_solve(ops, values, method=method, emit=observe, **options)

    monkeypatch.setattr(methods, "solve", observe_solve)
    stats = methods.solve_model(model, method=method, receiver_domain=1)
    assert stats is model.mor_stats
    assert stats["status"] == "converged", stats
    assert set(fields) == set(range(len(frequencies)))
    assert len(stats["receivers"]) == len(frequencies)
    ops = assemble_mor(model, receiver_domain=1)
    try:
        stiffness, mass = independent_matrices(model)
        q = full_vector(ops["q"])
        for index, frequency in enumerate(frequencies):
            s = 2j * np.pi * frequency
            rhs, matrix = s * q, stiffness + s * mass
            reference = np.linalg.solve(matrix, rhs)
            reference_residual = np.linalg.norm(rhs - matrix @ reference) / np.linalg.norm(rhs)
            assert reference_residual < 1e-10
            actual = fields[index]
            residual = np.linalg.norm(rhs - matrix @ actual) / np.linalg.norm(rhs)
            assert residual <= 1.01e-6, (degree, method, material, frequency, residual)
            if method == "DFOM":
                assert residual < 1e-10
            difference = actual - reference
            error = np.sqrt(np.vdot(difference, mass @ difference).real /
                np.vdot(reference, mass @ reference).real)
            assert error < 1e-3, (degree, method, material, frequency, error)
            expected = reference_receivers(ops, reference, s.imag)
            row = stats["receivers"][index]
            assert row["index"] == index
            assert row["frequency"] == frequency
            for name in ("E", "H"):
                relative = np.linalg.norm(row["data"][name] - expected[name]) / np.linalg.norm(expected[name])
                assert relative < 1e-3, (degree, method, material, frequency, name, relative)
        if degree == 2:
            assert stats["counts"]["ams_setups"] >= 1
            assert stats["counts"]["shift_factorizations"] == 0
    finally:
        ops["destroy"]()


def test_model_failure_preserves_statistics_and_releases_operators(monkeypatch):
    model = SimpleNamespace(FE=SimpleNamespace(bc=None),
        FS=SimpleNamespace(V=SimpleNamespace(
            ufl_element=lambda: SimpleNamespace(degree=lambda: 1))),
        MP=SimpleNamespace(frequencies=[1.]))
    released = []
    monkeypatch.setattr(operators, "assemble_mor", lambda *args, **kwargs:
        {"destroy": lambda: released.append(True)})
    failed = dict(status="failed", failure="injected numerical failure")
    monkeypatch.setattr(methods, "solve", lambda *args, **kwargs: failed)
    with pytest.raises(RuntimeError, match="injected numerical failure"):
        methods.solve_model(model)
    assert model.mor_stats is failed
    assert model.mor_stats["receivers"] == []
    assert released == [True]


@pytest.mark.parametrize("method,options,exception", [
    ("IFOM", {}, NotImplementedError),
    ("DFOM", {}, NotImplementedError),
    ("RK", {}, NotImplementedError),
    ("SUPERMOR", {"basis_strategy": "chain"}, ValueError),
    ("SUPERMOR", {"residual_correction": True}, ValueError),
])
def test_p2_model_scope_is_rejected_before_assembly(monkeypatch, method, options, exception):
    model = SimpleNamespace(FE=SimpleNamespace(bc=None),
        FS=SimpleNamespace(V=SimpleNamespace(
            ufl_element=lambda: SimpleNamespace(degree=lambda: 2))))

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported P2 model path reached assembly")

    monkeypatch.setattr(operators, "assemble_mor", forbidden)
    with pytest.raises(exception, match="P2"):
        methods.solve_model(model, method=method, **options)
