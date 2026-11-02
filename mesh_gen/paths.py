"""Resolve mesh inputs relative to the checkout, independent of the cwd.

Runtime scripts should call get_mesh() instead of reconstructing paths or
reading historical absolute paths from the original production metadata.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "mesh_data"

# cell count, unique edge count, real P1 degrees of freedom
EXPECTED = {
    "block_cross": (5256131, 6126084, 12252168),
    "block_strategy": (309349, 360026, 720052),
    "sphere": (2519112, 2952721, 5905442),
    "mr3d_M1": (385245, 448995, 897990),
    "mr3d_M2": (2168870, 2521000, 5042000),
    "mr3d_M3": (4780659, 5554916, 11109832),
    "mr3d_M4": (6409790, 7447120, 14894240),
    "mr3d_M5": (7877059, 9151292, 18302584),
}
MR3D_SCALES = {
    "M1": 0.401957,
    "M2": 0.19081228839710815,
    "M3": 0.14204016427579944,
    "M4": 0.12776329735875352,
    "M5": 0.11866730553215037,
}


def mesh_names():
    return ("block_cross", "block_emg3d", "block_strategy", "sphere", *(
        f"mr3d_{level}" for level in MR3D_SCALES
    ))


def get_mesh(name, *, require=False, data_root=None):
    """Return absolute runtime paths; require=True checks required files.

    data_root is an optional explicit replacement for data/mesh_data, useful
    when generating/verifying a fresh tree without replacing existing meshes.
    """
    base = Path(data_root).expanduser().resolve() if data_root is not None else DATA_ROOT
    if name == "block_cross":
        directory = base / "case1_block/cross_validation/custem/final/meshes/custem_L2"
        result = dict(mesh=directory / "_h5/custem_L2.h5",
                      parameters=directory / "para/custem_L2_parameters.json",
                      metadata=directory / "mesh.json", m_dir=directory)
    elif name == "block_emg3d":
        directory = base / "case1_block/cross_validation/emg3d/meshes/emg3d_fixed"
        result = dict(arrays=directory / "axes_sigma.npz", metadata=directory / "mesh.json",
                      receiver_support=directory / "receiver_support.npz",
                      receiver_support_metadata=directory / "receiver_support.json", m_dir=directory)
    elif name == "block_strategy":
        directory = base / "case1_block/strategy/mesh"
        result = dict(mesh=directory / "meshes/_h5/csem_mor.h5",
                      parameters=directory / "meshes/para/csem_mor_parameters.json",
                      metadata=directory / "mesh.json", m_dir=directory / "meshes")
    elif name == "sphere":
        directory = base / "case2_layered_sphere/q17_rx_plaza_earth_round_02"
        result = dict(mesh=directory / "meshes/_h5/terrain_sphere.h5",
                      parameters=directory / "meshes/para/terrain_sphere_parameters.json",
                      metadata=directory / "mesh.json", receivers=directory / "receivers.csv",
                      source=directory / "source.csv", m_dir=directory / "meshes")
    elif name.startswith("mr3d_") and name[5:] in MR3D_SCALES:
        level = name[5:]
        workdir = base / "case3_mr3d"
        directory = workdir / "meshes" / level / "attempt_01"
        stem = f"mr3d_{level.lower()}_a01"
        result = dict(mesh=directory / "_h5" / f"{stem}.h5",
                      parameters=directory / "para" / f"{stem}_parameters.json",
                      metadata=directory.parent / "manifest.json", m_dir=directory,
                      prepared=workdir / "prepared", survey=workdir / "prepared/survey.npz")
    else:
        raise ValueError(f"Unknown mesh {name!r}; choose from {', '.join(mesh_names())}")
    if require:
        # Regenerable sidecars are still required by the forward sampling path.
        missing = [path for key, path in result.items()
                   if key not in {"m_dir", "prepared"} and not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing mesh inputs: " + ", ".join(map(str, missing)))
    return result
