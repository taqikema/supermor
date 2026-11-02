"""DFOM startup policy survives the removal of its separate wrapper."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from petsc4py import PETSc

from runs.case1_block.cross_validation import custem_worker as worker
from runs.case1_block.cross_validation.run_cross_validation import commands


@pytest.fixture
def dfom_options():
    options = PETSc.Options()
    keys = ("hed_dfom_sym2_mat_mumps_icntl_4", "hed_dfom_sym2_mat_mumps_icntl_10",
            "hed_dfom_sym2_mat_mumps_icntl_22")
    previous = {key: options.getString(key) if options.hasName(key) else None for key in keys}
    for key in keys:
        if options.hasName(key):
            del options[key]
    try:
        yield options
    finally:
        for key, value in previous.items():
            if options.hasName(key):
                del options[key]
            if value is not None:
                options[key] = value


def test_dfom_cli_applies_formal_mumps_policy_before_worker(monkeypatch, dfom_options):
    def inspect_start(args):
        assert args.stage == "dfom"
        assert dfom_options.getInt("hed_dfom_sym2_mat_mumps_icntl_4") == 2
        assert dfom_options.getInt("hed_dfom_sym2_mat_mumps_icntl_10") == 0
        assert dfom_options.getInt("hed_dfom_sym2_mat_mumps_icntl_22") == 0
        return 23
    monkeypatch.setattr(worker, "run_solver", inspect_start)
    assert worker.main(["--stage", "dfom"]) == 23


def test_sumor_cli_keeps_mumps_policy_untouched(monkeypatch, dfom_options):
    dfom_options["hed_dfom_sym2_mat_mumps_icntl_22"] = 7
    def inspect_start(args):
        assert args.stage == "sumor"
        assert not dfom_options.hasName("hed_dfom_sym2_mat_mumps_icntl_4")
        assert not dfom_options.hasName("hed_dfom_sym2_mat_mumps_icntl_10")
        assert dfom_options.getInt("hed_dfom_sym2_mat_mumps_icntl_22") == 7
        return 0
    monkeypatch.setattr(worker, "run_solver", inspect_start)
    assert worker.main(["--stage", "sumor"]) == 0


def test_dfom_input_check_never_imports_numerical_runtime(monkeypatch, tmp_path, capsys):
    mesh, parameters, metadata = (tmp_path / name for name in ("model.h5", "parameters.json", "mesh.json"))
    mesh.write_bytes(b"input fixture")
    parameters.write_text("{}")
    metadata.write_text(json.dumps(dict(input_sha256=dict(h5=worker.sha256(mesh),
        parameters=worker.sha256(parameters)), dimensions={"cells": 4})))
    monkeypatch.setattr(worker, "get_case_mesh", lambda *a, **kw:
        dict(mesh=mesh, parameters=parameters, metadata=metadata))
    original_import = builtins.__import__
    def reject_runtime(name, *args, **kwargs):
        if name.split(".")[0] in ("petsc4py", "dolfin", "mpi4py", "custEM", "supermor"):
            raise AssertionError("Input-only audit imported " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_runtime)
    monkeypatch.setattr(worker, "run_solver", lambda args: pytest.fail("Input check started a solve"))
    assert worker.main(["--stage", "dfom", "--check-inputs"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


@pytest.mark.parametrize("method,stage,ranks", [("DFOM", "dfom", 16), ("SUPERMOR", "sumor", 32)])
def test_cross_validation_launches_shared_worker_with_formal_stage(tmp_path, method, stage, ranks):
    args = SimpleNamespace(method=method, indices="0,1,2,3", data_root=None,
        emg3d_python=None, custem_python=None, emg3d_source=None,
        check_inputs=False, mpiexec="mpiexec")
    command = commands(args, tmp_path)[method]
    assert command[:3] == ["mpiexec", "-n", str(ranks)]
    script = command[command.index("--stage") - 1]
    assert Path(script).name == "custem_worker.py" and Path(script).is_file()
    assert command[command.index("--stage") + 1] == stage


def test_dfom_uncaught_worker_error_aborts_all_ranks(monkeypatch, dfom_options):
    import mpi4py
    class AbortObserved(Exception):
        pass
    def abort(code):
        assert code == 1
        raise AbortObserved
    monkeypatch.setattr(mpi4py, "MPI", SimpleNamespace(COMM_WORLD=SimpleNamespace(Abort=abort)))
    def fail(args):
        raise RuntimeError("Synthetic startup error")
    monkeypatch.setattr(worker, "run_solver", fail)
    with pytest.raises(AbortObserved):
        worker.main(["--stage", "dfom"])
