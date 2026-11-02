"""Portable paper inputs, receiver error definitions and process measurement.

These tests use small arrays and short child processes. They never assemble a
paper mesh, invoke a forward solver or start an MPI campaign.
"""
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.common import mesh_inputs, result_io
from runs.common.measurement import run_monitored
from runs.common.receiver_metrics import compare_archives, nrmsd, relative_metrics
from runs.case1_block.strategy.worker import strategy_options
from runs.case3_mr3d.run_methods_mpi import validate_native_stats


def test_total_data_root_and_legacy_mesh_root_resolve_same_files(tmp_path):
    data = tmp_path / "relocated" / "data"
    for name in ("block_cross", "block_emg3d", "block_strategy", "sphere", "mr3d_M3"):
        actual = mesh_inputs.get_case_mesh(name, data_root=data, require=False)
        legacy = mesh_inputs.get_mesh(name, data_root=data / "mesh_data", require=False)
        assert actual == legacy
        assert all(path.is_relative_to(data / "mesh_data") for path in actual.values())
    assert mesh_inputs.material_directory("case3_mr3d", data_root=data) == data / "model_data/case3_mr3d"


def test_default_input_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    before = mesh_inputs.get_case_mesh("sphere", require=False)
    monkeypatch.chdir(tmp_path)
    assert mesh_inputs.get_case_mesh("sphere", require=False) == before
    assert all(path.is_absolute() for path in before.values())
    assert mesh_inputs.material_directory("case3_mr3d") == ROOT / "data/model_data/case3_mr3d"


def test_missing_inputs_fail_without_creating_directories(tmp_path):
    missing = tmp_path / "not_downloaded"
    with pytest.raises(FileNotFoundError, match="Missing mesh inputs"):
        mesh_inputs.get_case_mesh("mr3d_M5", data_root=missing)
    assert not missing.exists()


def test_runtime_mesh_keeps_auxiliary_writes_outside_input_data(tmp_path):
    source = tmp_path / "data/mesh_data/case1_block"
    h5 = source / "_h5/model.h5"
    parameters = source / "para/model_parameters.json"
    auxiliary_parameter = source / "para/surface.xyz"
    h5.parent.mkdir(parents=True)
    parameters.parent.mkdir()
    h5.write_bytes(b"immutable HDF5 input fixture")
    parameters.write_text('{"receiver":[0,0,0]}')
    auxiliary_parameter.write_text("0 0 0\n")
    original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    paths = dict(mesh=h5, parameters=parameters, m_dir=source)
    run = tmp_path / "outputs/run_001"
    runtime = mesh_inputs.runtime_mesh(paths, run)
    assert runtime["m_dir"] == run / "runtime_mesh"
    assert set(path.name for path in runtime["m_dir"].iterdir()) == {"_h5", "para"}
    assert runtime["mesh"].is_symlink() and runtime["mesh"].resolve() == h5
    assert runtime["parameters"].is_symlink() and runtime["parameters"].resolve() == parameters
    assert (runtime["parameters"].parent / "surface.xyz").is_symlink()
    # Simulate the upstream loader's auxiliary-directory creation.
    (runtime["m_dir"] / "_xml").mkdir()
    (runtime["m_dir"] / "_xml/generated.xml").write_text("auxiliary output")
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == original
    assert not (source / "_xml").exists()
    assert paths["m_dir"] == source
    # Loading the same attempt again may reuse exactly the same links.
    assert mesh_inputs.runtime_mesh(paths, run)["mesh"].resolve() == h5


def test_explicit_nonempty_output_is_rejected_without_touching_results(tmp_path):
    directory = tmp_path / "previous_run"
    directory.mkdir()
    old = directory / "stats.json"
    old.write_text('{"complete":true}')
    with pytest.raises(FileExistsError, match="not empty"):
        result_io.output_directory("case1_block", "strategy", output=directory)
    assert old.read_text() == '{"complete":true}'
    assert result_io.output_directory("case1_block", "strategy", output=tmp_path / "new_run").is_dir()


def test_default_outputs_are_unique_and_stay_in_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(result_io, "ROOT", tmp_path)
    first = result_io.output_directory("case1_block", "strategy")
    second = result_io.output_directory("case1_block", "strategy")
    assert first != second
    assert first.parent == second.parent == tmp_path / "outputs/case1_block/strategy"


def test_output_cannot_be_created_inside_input_data(tmp_path, monkeypatch):
    monkeypatch.setattr(result_io, "ROOT", tmp_path)
    forbidden = tmp_path / "data/mesh_data/case1_block/new_result"
    with pytest.raises(ValueError, match="[Ii]nput|data"):
        result_io.output_directory("case1_block", "strategy", output=forbidden)
    assert not forbidden.exists()


def test_complex_global_l2_and_max_pointwise_are_distinct():
    # Only the weak receiver has 100% error. The strong receiver dominates
    # the global norm, so averaging receiver percentages would be wrong.
    reference = np.array([[100j, 0, 0], [1j, 0, 0]])
    actual = np.array([[100j, 0, 0], [2j, 0, 0]])
    metrics = relative_metrics(actual, reference)
    assert metrics["relative_L2_pct"] == pytest.approx(100 / np.sqrt(10001))
    assert metrics["max_pointwise_pct"] == pytest.approx(100.)
    assert metrics["zero_reference_points"] == 0


def test_zero_reference_discrepancy_is_preserved():
    reference = np.array([[0., 0, 0], [1., 0, 0]], dtype=complex)
    actual = np.array([[1., 0, 0], [1., 0, 0]], dtype=complex)
    metrics = relative_metrics(actual, reference)
    assert np.isinf(metrics["max_pointwise_pct"])
    assert metrics["relative_L2_pct"] == pytest.approx(100.)
    assert metrics["zero_reference_points"] == 1
    zero = np.zeros((2, 3), dtype=complex)
    assert relative_metrics(zero, zero)["relative_L2_pct"] == 0
    assert relative_metrics(zero, zero)["max_pointwise_pct"] == 0
    assert np.isinf(relative_metrics(actual, zero)["relative_L2_pct"])


def test_nrmsd_uses_complex_values_and_is_symmetric():
    actual = np.array([0, 1j, 2 + 2j], complex)
    reference = np.array([0, -1j, 1 + 1j], complex)
    assert nrmsd(actual, reference) == pytest.approx([0, 200, 200 / 3])
    assert np.array_equal(nrmsd(actual, reference), nrmsd(reference, actual))


def archive(path, *, coordinates=None, fields=None, frequencies=None, indices=None):
    coordinates = np.array([[0., 0, 0], [1., 0, 0]]) if coordinates is None else coordinates
    fields = np.ones((2, 2, 3), complex) if fields is None else fields
    np.savez(path, frequencies_hz=np.array([1., 10.]) if frequencies is None else frequencies,
             indices=np.array([0, 1]) if indices is None else indices,
             rx_xyz=coordinates, E=fields, H=2 * fields)


def test_archive_comparison_rejects_receiver_order_mismatch(tmp_path):
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    archive(actual)
    # Even constant fields must not conceal a different point order.
    archive(reference, coordinates=np.array([[1., 0, 0], [0., 0, 0]]))
    with pytest.raises(ValueError, match="rx_xyz"):
        compare_archives(actual, reference)


@pytest.mark.parametrize("bad_side", ["actual", "reference"])
def test_archive_rejects_field_count_inconsistent_with_frequency_identity(tmp_path, bad_side):
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    archive(actual)
    archive(reference)
    archive(actual if bad_side == "actual" else reference, fields=np.ones((3, 2, 3), complex))
    with pytest.raises(ValueError, match="shape"):
        compare_archives(actual, reference)


def test_partial_frequency_archive_is_not_reported_as_full_comparison(tmp_path):
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    for path in (actual, reference):
        archive(path, fields=np.ones((1, 2, 3), complex), frequencies=np.array([10.]), indices=np.array([1]))
    with pytest.raises(ValueError, match="ordered frequency"):
        compare_archives(actual, reference)


def test_saved_receiver_rows_are_sorted_before_comparison(tmp_path):
    path = tmp_path / "receivers.npz"
    coordinates = np.array([[0., 0, 0], [1., 0, 0]])
    rows = {1: {"E": np.full((2, 3), 2j), "H": np.full((2, 3), 3j)},
            0: {"E": np.full((2, 3), 1j), "H": np.full((2, 3), 2j)}}
    result_io.save_receivers(path, [1., 10.], rows, coordinates)
    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data["indices"], [0, 1])
        assert np.array_equal(data["frequencies_hz"], [1., 10.])
        assert np.array_equal(data["E"][:, 0, 0], [1j, 2j])
    assert all(row["relative_L2_pct"] == 0 for row in compare_archives(path, path))


def test_nonfinite_metric_is_explicitly_recorded_in_json(tmp_path):
    path = tmp_path / "summary.json"
    result_io.write_json(path, {"error": np.inf, "indices": np.array([0, 1]), "path": tmp_path})
    raw = path.read_text()
    assert "Infinity" not in raw and "NaN" not in raw
    data = json.loads(raw)
    assert data["error"] is None and ".error" in data["nonfinite_values"]
    assert data["indices"] == [0, 1]
    assert data["path"] == str(tmp_path)


def test_strategy_dual_pole_contract_retains_paper_schedule():
    options = strategy_options("dual_pole")
    assert options["poles"] == pytest.approx([-2274.7560643370834, -2.221441469079183])
    assert options["pole_every"] == 20 and options["pole_schedule"] == "hold_first"
    assert options["max_basis"] == 400 and options["inner_rtol"] == 1e-6
    assert options["field_formulation"] == "split_u" and not options["direct"]
    assert strategy_options("single_pole")["poles"] == pytest.approx([-71.08612701053386])
    assert strategy_options("total_field_lu")["direct"] is True


def sumor_native_stats():
    return dict(route="split_u", status="converged", residual_verified=True,
                test_direct_inverse=False, counts={"poisson_rhs": 2},
                linear_solves=[dict(name="poisson", relative_residual=value,
                                   rtol=1e-10, max_it=300, direct=False,
                                   require_true_residual=True, reason=2)
                               for value in (2e-12, 7e-12)])


def test_mr3d_poisson_aggregate_is_derived_from_all_rhs():
    native = sumor_native_stats()
    validate_native_stats("SUMOR", native)
    assert native["poisson_maximum_true_residual"] == 7e-12


@pytest.mark.parametrize("defect", [
    "nan_residual", "missing_rhs", "wrong_route", "unverified_maxwell",
    "direct_inverse", "wrong_tolerance", "unverified_poisson",
])
def test_mr3d_rejects_incomplete_or_invalid_native_acceptance_evidence(defect):
    native = deepcopy(sumor_native_stats())
    if defect == "nan_residual":
        native["linear_solves"][1]["relative_residual"] = np.nan
    elif defect == "missing_rhs":
        native["linear_solves"].pop()
    elif defect == "wrong_route":
        native["route"] = "cpc"
    elif defect == "unverified_maxwell":
        native["residual_verified"] = False
    elif defect == "direct_inverse":
        native["test_direct_inverse"] = True
    elif defect == "wrong_tolerance":
        native["linear_solves"][0]["rtol"] = 1e-6
    elif defect == "unverified_poisson":
        native["linear_solves"][0]["require_true_residual"] = False
    with pytest.raises(RuntimeError, match="acceptance|evidence"):
        validate_native_stats("SUMOR", native)


def test_monitor_captures_short_child_and_preserves_nonzero_exit_code(tmp_path):
    command = [sys.executable, "-B", "-c",
               "import sys,time; data=bytearray(8*1024**2); print('short-worker',flush=True); time.sleep(.12); sys.exit(7)"]
    record = run_monitored(command, tmp_path, interval=.01, profile="later")
    assert record["returncode"] == 7 and record["samples"] >= 1
    assert record["peak_job_pss_bytes"] > 0 and record["monitor_excluded"]
    assert "short-worker" in (tmp_path / "run.log").read_text()
    persisted = json.loads((tmp_path / "resources.json").read_text())
    assert persisted["returncode"] == 7
    with (tmp_path / "memory_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(int(row["processes"]) >= 1 and int(row["job_pss_bytes"]) > 0 for row in rows)


def test_monitor_includes_live_descendants_in_the_job_tree(tmp_path):
    grandchild = "import time; data=bytearray(4*1024**2); time.sleep(.15)"
    parent = ("import subprocess,sys; "
              f"p=subprocess.Popen([sys.executable,'-B','-c',{grandchild!r}]); p.wait()")
    record = run_monitored([sys.executable, "-B", "-c", parent], tmp_path, interval=.01)
    assert record["returncode"] == 0
    with (tmp_path / "memory_samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(int(row["processes"]) >= 2 for row in rows)
