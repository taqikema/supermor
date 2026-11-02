"""MR3D 几何与按实虚块自由度校准的四面体网格。"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


TARGET_REAL_DOF = {
    "M1": 936398, "M2": 5372032, "M3": 10399316,
    "M4": 15677176, "M5": 20350580,
}
INTERFACES = (
    "Sea_Bottom-mr3d_reverted.xyz", "Miocene-mr3d_reverted.xyz",
    "Oligocene-mr3d_reverted.xyz", "Blue_mark-mr3d_reverted.xyz",
    "Top_of_Salt-mr3d_reverted.xyz", "Base_of_salt-mr3d_reverted.xyz",
)


def load_mesh_manifest(workdir, level):
    """Read a level manifest and resolve its paths against the MR3D data root.

    Archived manifests use relative ``m_dir`` paths so the data directory can
    move with the checkout. Original absolute generation paths also remain
    accepted for manifests produced by ``build_mesh``.
    """
    if level not in TARGET_REAL_DOF:
        raise ValueError(f"Unknown level: {level}; expected M1 through M5")
    workdir = Path(workdir).expanduser().resolve()
    manifest = json.loads((workdir / "meshes" / level / "manifest.json").read_text(encoding="utf-8"))
    for record in [manifest, *manifest.get("attempts", [])]:
        for key in ("m_dir", "log"):
            value = record.get(key)
            if value is not None:
                path = Path(value).expanduser()
                record[key] = str(path.resolve() if path.is_absolute() else (workdir / path).resolve())
    return manifest


def _next_scale(attempts, target):
    """优先在实测夹逼区间内插，否则按实测增长率外推线性尺度。"""
    brackets = [
        (left, right) for left in attempts for right in attempts
        if left["scale"] < right["scale"]
        and left["real_block_dof"] > target > right["real_block_dof"]
    ]
    if brackets:
        left, right = min(
            brackets, key=lambda pair: math.log(pair[1]["scale"] / pair[0]["scale"]))
        weight = math.log(target / left["real_block_dof"]) / math.log(
            right["real_block_dof"] / left["real_block_dof"])
        # Keep each new measurement away from both endpoints so the log-scale
        # bracket shrinks even when TetGen's count growth is unexpectedly steep.
        weight = min(.8, max(.2, weight))
        return math.exp(math.log(left["scale"]) + weight * math.log(
            right["scale"] / left["scale"]))

    current = attempts[-1]
    exponent = 3.
    for previous in reversed(attempts[:-1]):
        if previous["scale"] == current["scale"]:
            continue
        measured = math.log(current["real_block_dof"] / previous["real_block_dof"]) / math.log(
            previous["scale"] / current["scale"])
        if measured > 0:
            exponent = min(3., max(.5, measured))
        break
    ratio = (current["real_block_dof"] / target) ** (1 / exponent)
    return current["scale"] * min(2., max(.5, ratio))


def _initial_scale(workdir, target, quality, half_extent):
    """同几何和质量的已校准 M1 可缩短后续档位的校准过程。"""
    baseline = Path(workdir) / "meshes" / "M1" / "manifest.json"
    if baseline.is_file():
        manifest = json.loads(baseline.read_text(encoding="utf-8"))
        parameters = manifest["parameters"]
        if (manifest["within_tolerance"] and parameters["quality"] == quality
                and parameters["half_extent"] == half_extent):
            return parameters["scale"] * (manifest["real_block_dof"] / target) ** (1 / 3)
    return (TARGET_REAL_DOF["M1"] / target) ** (1 / 3)


def read_tetgen_counts(base):
    """读取 TetGen 实际边数；一阶复 Nédélec 每条边一个自由度。"""
    def header_count(suffix):
        with Path(str(base) + suffix).open(encoding="utf-8") as stream:
            for line in stream:
                data = line.split("#", 1)[0].split()
                if data:
                    return int(data[0])
        raise ValueError(f"Empty TetGen header: {base}{suffix}")

    edges = header_count(".edge")
    return {"complex_dof": edges, "real_block_dof": 2 * edges,
            "n_elements": header_count(".ele")}


@contextmanager
def _stage_log(path):
    """同时保存 Python、Triangle 和 TetGen 的阶段输出。"""
    sys.stdout.flush()
    sys.stderr.flush()
    original = [os.dup(1), os.dup(2)]
    with path.open("w", encoding="utf-8") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, saved in zip((1, 2), original):
                os.dup2(saved, fd)
                os.close(saved)


def _generate(prepared, m_dir, name, survey, scale, quality, half_extent):
    from custEM.meshgen import meshgen_utils as mu
    from custEM.meshgen.meshgen_tools import BlankWorld
    from custEM.misc.synthetic_definitions import flat_topo

    world = BlankWorld(
        name=name, m_dir=str(m_dir), backup_script=False,
        topo=flat_topo, x_dim=[-half_extent, half_extent],
        y_dim=[-half_extent, half_extent], z_dim=[-half_extent, half_extent],
        boundary_mesh_cell_size=0., outer_area_cell_size=1e7 * scale ** 2,
        layer_cell_sizes=(np.array([1e9, 1e9, 1e9, 1e8, 1e9, 1e9, 1e9])
                          * scale ** 3).tolist(),
        interface_cell_sizes=(np.array([1e7, 1e7, 1e6, 1e6, 1e7, 1e7])
                              * scale ** 2).tolist(),
        centering=False, t_dir=str(prepared / "interfaces"),
        easting_shift=-390275., northing_shift=-7518065., preserve_edges=True,
    )
    world.build_surface()
    # Source endpoints retain the supplied elevation.  The projected segment
    # subdivides the seafloor triangles as in the public author geometry.
    source = survey["source_endpoints"].copy()
    world.build_layered_earth_mesh(
        7, layer_depths=[-1., -2., -3., -4., -5., -6.],
        subsurface_topos=list(INTERFACES), insert_struct=[source.copy()],
    )
    for key in ("inline", "broadside"):
        positions = survey[key]
        world.add_paths(mu.refine_rx(positions, 5.))
        world.add_rx(positions.copy())
    world.add_tx(source)
    world.extend_world(1.1, 1.1, 1.1, cell_sizes=[0., 1e9 * scale ** 3])
    world.extend_world(2., 2., 2., initial=False)
    # Keep .node/.ele as well as the actual .1.edge count.  TetGen's F switch
    # suppresses both face and edge output, so no export suppression is used.
    world.call_tetgen(tet_param=f"-pq{quality:g}aA", export_format=" -ke",
                      suppress="", ignore_tx_z=True)
    return read_tetgen_counts(m_dir / "mesh_create" / f"{name}.1")


def build_mesh(workdir, level="M1", target_dof=None, tolerance=.15,
               max_attempts=4, *, initial_scale=None, quality=2.,
               half_extent=11400.):
    """串行建网格并保留最接近目标的结果；target_dof 为实虚合计 2N。

    初始几何参照公开作者脚本，分辨率按线性尺度 h 调整。档位为近似
    DOF 匹配，不代表恢复作者原始 M1–M5 配置。未进入容差仍保存最佳
    可用网格，manifest 的 within_tolerance 明确记录结果。
    """
    if level not in TARGET_REAL_DOF:
        raise ValueError(f"Unknown level: {level}; expected M1 through M5")
    target = TARGET_REAL_DOF[level] if target_dof is None else target_dof
    if not isinstance(target, (int, np.integer)) or target <= 0:
        raise ValueError("target_dof must be a positive integer (real-block DOF)")
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for key, value in (("quality", quality), ("half_extent", half_extent)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    scale = _initial_scale(workdir, target, quality, half_extent) if initial_scale is None else initial_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("initial_scale must be positive and finite")
    from dolfin import MPI
    if MPI.size(MPI.comm_world) != 1:
        raise RuntimeError("Mesh generation must run serially, without mpirun")

    workdir = Path(workdir).resolve()
    prepared = workdir / "prepared"
    from experiments.mr3d_custem.prepare import require_prepared
    require_prepared(prepared)
    for filename in INTERFACES:
        if not (prepared / "interfaces" / filename).is_file():
            raise FileNotFoundError(prepared / "interfaces" / filename)
    with np.load(prepared / "survey.npz", allow_pickle=False) as data:
        survey = {key: np.asarray(data[key], dtype=float) for key in
                  ("inline", "broadside", "receiver", "source_endpoints")}
    for key in ("inline", "broadside"):
        if survey[key].ndim != 2 or survey[key].shape[1] != 3 or not len(survey[key]):
            raise ValueError(f"survey {key} must have shape (n, 3), with n > 0")
    if survey["receiver"].shape != (3,) or survey["source_endpoints"].shape != (2, 3):
        raise ValueError("survey receiver/source_endpoints must have shapes (3,) and (2, 3)")
    if not all(np.isfinite(values).all() for values in survey.values()):
        raise ValueError("survey coordinates must be finite")

    level_dir = workdir / "meshes" / level
    level_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for attempt in range(1, max_attempts + 1):
        m_dir = level_dir / f"attempt_{attempt:02d}"
        # Repeated invocations must not mistake an earlier TetGen result for
        # this invocation if TetGen fails (its upstream wrapper uses os.system).
        if m_dir.exists():
            suffix = 2
            while (level_dir / f"attempt_{attempt:02d}_run{suffix}").exists():
                suffix += 1
            m_dir = level_dir / f"attempt_{attempt:02d}_run{suffix}"
        m_dir.mkdir()
        name = f"mr3d_{level.lower()}_a{attempt:02d}"
        log = m_dir / "mesh.log"
        print(f"{level} attempt {attempt}/{max_attempts}: h={scale:.6g}; log={log}", flush=True)
        started = time.monotonic()
        with _stage_log(log):
            counts = _generate(prepared, m_dir, name, survey, scale,
                               quality, half_extent)
        record = {"attempt": attempt, "mesh_name": name, "m_dir": str(m_dir),
                  "scale": float(scale), "quality": float(quality), **counts,
                  "relative_error": abs(counts["real_block_dof"] / target - 1),
                  "elapsed_seconds": time.monotonic() - started,
                  "log": str(log)}
        attempts.append(record)
        print(f"  real DOF={record['real_block_dof']:,}; error={record['relative_error']:.2%}", flush=True)
        if record["relative_error"] <= tolerance:
            break
        scale = _next_scale(attempts, target)

    best = min(attempts, key=lambda item: item["relative_error"])
    # Convert only the chosen mesh; conversion is materialized for MOD to read
    # directly and avoids holding every calibration mesh in dolfin memory.
    # Run conversion in a fresh interpreter: the installed pyGIMLi/Triangle
    # and dolfin HDF5 libraries abort when meshing and HDF5 writing share a
    # process.  XML/HDF5 are still produced by custEM's own converter.
    with (Path(best["m_dir"]) / "conversion.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c",
             "import sys; from custEM.meshgen import vtk2xml2h5; "
             "vtk2xml2h5(sys.argv[1], sys.argv[2], None, None)",
             best["mesh_name"], best["m_dir"]],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    manifest = {
        "schema_version": 1, "level": level, "mesh_name": best["mesh_name"],
        "m_dir": best["m_dir"], "complex_dof": best["complex_dof"],
        "real_block_dof": best["real_block_dof"], "n_elements": best["n_elements"],
        "target_real_dof": int(target), "tolerance": float(tolerance),
        "within_tolerance": best["relative_error"] <= tolerance,
        "dof_measurement": "TetGen output .1.edge count; N1curl degree 1; real block is 2N",
        "parameters": {"scale": best["scale"], "quality": float(quality),
                       "half_extent": float(half_extent), "source_length": 10.,
                       "receiver_refinement_radius": 5.,
                       "easting_shift": -390275., "northing_shift": -7518065.,
                       "outer_area_cell_size": 1e7 * best["scale"] ** 2,
                       "layer_cell_sizes": [v * best["scale"] ** 3 for v in
                                            (1e9, 1e9, 1e9, 1e8, 1e9, 1e9, 1e9)],
                       "interface_cell_sizes": [a * best["scale"] ** 2 for a in
                                                (1e7, 1e7, 1e6, 1e6, 1e7, 1e7)],
                       "extension_factors": [1.1, 2.],
                       "extension_inner_cell_sizes": [0., 1e9 * best["scale"] ** 3],
                       "tetgen_options": f"-pq{quality:g}aA -ke",
                       "interfaces": list(INTERFACES)},
        "attempts": attempts,
    }
    (level_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not manifest["within_tolerance"]:
        print(f"{level}: target tolerance not reached; retained closest mesh "
              f"({best['relative_error']:.2%} error)", flush=True)
    return manifest
