"""Mesh portability and independent topology checks; no paper meshes required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesh_gen.paths import DATA_ROOT, get_mesh, mesh_names  # noqa: E402
from mesh_gen.verify import tetrahedral_counts  # noqa: E402


def write_mesh(path, topology):
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                            [0., 0., 1.], [0., 0., -1.]])
    with h5py.File(path, "w") as stream:
        stream["mesh/coordinates"] = coordinates
        stream["mesh/topology"] = np.asarray(topology, dtype=np.int64)


@pytest.mark.parametrize("topology", [
    [[0, 1, 2, 3], [0, 1, 2, 4]],
    [[4, 2, 1, 0], [3, 0, 2, 1]],
])
def test_independent_counter_counts_shared_face_once(tmp_path, topology):
    path = tmp_path / "two_tets.h5"
    write_mesh(path, topology)
    assert tetrahedral_counts(path) == {
        "cells": 2, "vertices": 5, "edges": 9, "real_P1_dof": 18,
    }


@pytest.mark.parametrize("topology,message", [
    ([[0, 1, 2, 9]], "invalid vertex"),
    ([[0, 1, 1, 3]], "duplicate vertices"),
])
def test_independent_counter_rejects_bad_topology(tmp_path, topology, message):
    path = tmp_path / "bad.h5"
    write_mesh(path, topology)
    with pytest.raises(ValueError, match=message):
        tetrahedral_counts(path)


def test_runtime_paths_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in mesh_names():
        paths = get_mesh(name)
        assert all(path.is_absolute() and path.is_relative_to(DATA_ROOT) for path in paths.values())
    with pytest.raises(FileNotFoundError, match="Missing mesh inputs"):
        get_mesh("sphere", require=True, data_root=tmp_path / "empty")


def test_mr3d_manifest_resolves_relative_data_directory(tmp_path, monkeypatch):
    source = ROOT / "mesh_gen/case3_mr3d/experiments/mr3d_custem/mesh.py"
    spec = importlib.util.spec_from_file_location("mesh_generation_test_mr3d", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    level = tmp_path / "model/meshes/M1"
    level.mkdir(parents=True)
    (level / "manifest.json").write_text(json.dumps({"m_dir": "meshes/M1/attempt_01"}))
    monkeypatch.chdir(tmp_path)
    actual = module.load_mesh_manifest(tmp_path / "model", "M1")
    assert Path(actual["m_dir"]) == (level / "attempt_01").resolve()


@pytest.mark.parametrize("case_args", [[], ["all"]])
def test_dry_run_uses_formal_parameters_and_explicit_data_root(tmp_path, case_args):
    process = subprocess.run([sys.executable, "-B", str(ROOT / "mesh_gen/generate.py"),
                              *case_args, "--dry-run", "--data-root", str(tmp_path / "data")],
                             cwd=tmp_path, text=True, capture_output=True, check=True)
    stages = json.loads(process.stdout)
    audit = json.loads((ROOT / "MESH_AUDIT_20261101.json").read_text())
    for formal in audit["mr3d_evidence"]["levels"]:
        command = stages["mr3d_" + formal["level"]][0]
        for flag, value in [("--initial-scale", formal["scale"]),
                            ("--quality", formal["quality"]),
                            ("--half-extent", formal["half_extent"])]:
            assert float(command[command.index(flag) + 1]) == value
        assert command[command.index("--max-attempts") + 1] == "1"
    initial, first, second = stages["sphere"]
    assert initial[initial.index("--quality") + 1] == "1.7"
    assert first[first.index("--max-added-dofs") + 1] == "905028"
    assert second[second.index("--max-added-dofs") + 1] == "505909"
    for command_list in stages.values():
        for command in command_list:
            assert not any("/mnt/d/" in part or "/public/home/" in part for part in command)
            for flag in ("--root", "--workdir", "--input", "--output", "--template"):
                if flag in command:
                    assert Path(command[command.index(flag) + 1]).is_relative_to(tmp_path / "data")


@pytest.mark.parametrize("case_args", [[], ["block_strategy"]])
def test_missing_mesh_check_fails_without_generating_data(tmp_path, case_args):
    location = tmp_path / "not_generated"
    process = subprocess.run([sys.executable, "-B", str(ROOT / "mesh_gen/generate.py"),
                              *case_args, "--check", "--data-root", str(location)],
                             cwd=tmp_path, text=True, capture_output=True)
    assert process.returncode == 2
    assert "Missing mesh inputs" in process.stderr
    assert not location.exists()


def test_small_sphere_input_is_available_outside_code_tree():
    path = DATA_ROOT / "case2_layered_sphere/inputs/sphere_surface_template.npz"
    with np.load(path, allow_pickle=False) as template:
        assert template["points"].shape == (642, 3)
        assert template["faces"].shape == (1280, 3)
        assert np.isfinite(template["points"]).all()
    assert not (ROOT / "mesh_gen/case2_layered_sphere/sphere_surface_template.npz").exists()


@pytest.mark.parametrize("missing", [("receiver_support.npz", "receiver_support.json"),
                                    ("receiver_support.npz",), ("receiver_support.json",)])
def test_emg_generator_restores_missing_sampling_inputs(tmp_path, missing):
    from mesh_gen.case1_block.cross_validation.emg3d_worker import prepare_mesh, prepare_supports
    from mesh_gen.verify import sha256

    data_root = tmp_path / "data"
    location = data_root / "case1_block/cross_validation/emg3d"
    arrays, _ = prepare_mesh(location, create=True)
    prepare_supports(location, arrays, create=True)
    files = get_mesh("block_emg3d", data_root=data_root)
    original_mesh_sha = sha256(files["arrays"])
    for name in missing:
        (files["m_dir"] / name).unlink()
    with pytest.raises(FileNotFoundError, match="Missing mesh inputs"):
        get_mesh("block_emg3d", require=True, data_root=data_root)

    process = subprocess.run([sys.executable, "-B", str(ROOT / "mesh_gen/generate.py"),
                              "block_emg3d", "--data-root", str(data_root)], cwd=tmp_path,
                             text=True, capture_output=True, check=True)
    assert '"receiver_support_checked": true' in process.stdout
    get_mesh("block_emg3d", require=True, data_root=data_root)
    assert sha256(files["arrays"]) == original_mesh_sha
    prepare_supports(location, arrays, create=False)


def test_emg_check_rejects_changed_sampling_weights(tmp_path):
    from mesh_gen.case1_block.cross_validation.emg3d_worker import prepare_mesh, prepare_supports
    from mesh_gen.verify import check_mesh

    location = tmp_path / "case1_block/cross_validation/emg3d"
    arrays, _ = prepare_mesh(location, create=True)
    prepare_supports(location, arrays, create=True)
    path = get_mesh("block_emg3d", data_root=tmp_path)["receiver_support"]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["Ex_weights"][0, 0] += 0.01
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="receiver support audit changed"):
        check_mesh("block_emg3d", data_root=tmp_path)
