#!/usr/bin/env python3
"""Generate the paper meshes with explicit formal parameters and portable paths."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesh_gen.paths import DATA_ROOT, MR3D_SCALES, get_mesh, mesh_names  # noqa: E402


def commands(name, data_root):
    """Return the exact stages for one final mesh, without executing them."""
    base = Path(data_root).expanduser().resolve()
    code = ROOT / "mesh_gen"
    python = [sys.executable, "-u", "-B"]
    if name == "block_cross":
        location = base / "case1_block/cross_validation/custem"
        scripts = code / "case1_block/cross_validation"
        return [python + [str(scripts / "mesh_custem.py"), "--root", str(location / "parent"),
                          "--level", "2", "--volume-scale", "1", "--mode", "build"],
                python + [str(scripts / "refine_receiver_mesh.py"), "--parent",
                          str(location / "parent/meshes/custem_L2"), "--root", str(location / "final"),
                          "--passes", "1"]]
    if name == "block_emg3d":
        return [python + [str(code / "case1_block/cross_validation/emg3d_worker.py"),
                          "--root", str(base / "case1_block/cross_validation/emg3d")]]
    if name == "block_strategy":
        return [python + [str(code / "case1_block/strategy/mesh_strategy.py"), "--root",
                          str(base / "case1_block/strategy"), "--volume-scale", "1", "--mode", "build"]]
    if name == "sphere":
        location = base / "case2_layered_sphere"
        scripts = code / "case2_layered_sphere"
        return [python + [str(scripts / "build_terrain_mesh_sensitivity.py"), "--output",
                          str(location / "candidate_q17_local025"), "--mode", "formal",
                          "--outer-km", "80", "--scale", "4", "--local-volume-factor", "0.25",
                          "--quality", "1.7", "--max-attempts", "1", "--template",
                          str(location / "inputs/sphere_surface_template.npz")],
                python + [str(scripts / "refine_terrain_rx_star_dolfin.py"), "--input",
                          str(location / "candidate_q17_local025/attempt_01"), "--output",
                          str(location / "q17_rx_plaza_round_01"), "--max-added-dofs", "905028"],
                python + [str(scripts / "refine_terrain_rx_earth_star_dolfin.py"), "--input",
                          str(location / "q17_rx_plaza_round_01"), "--output",
                          str(location / "q17_rx_plaza_earth_round_02"), "--max-added-dofs", "505909"]]
    if name.startswith("mr3d_") and name[5:] in MR3D_SCALES:
        level = name[5:]
        return [python + [str(code / "case3_mr3d/experiments/mr3d_custem/mr3d.py"),
                          "--workdir", str(base / "case3_mr3d"), "mesh", "--level", level,
                          "--initial-scale", repr(MR3D_SCALES[level]), "--quality", "2",
                          "--half-extent", "11400", "--max-attempts", "1", "--tolerance", "0.15"]]
    raise ValueError(f"Unknown mesh {name!r}")


def prepare_inputs(name, data_root):
    """Copy required local small/Zenodo inputs into a fresh explicit data tree."""
    if name == "sphere":
        relative = [Path("case2_layered_sphere/inputs/sphere_surface_template.npz")]
    elif name.startswith("mr3d_"):
        prepared = Path("case3_mr3d/prepared")
        relative = [prepared / "manifest.json", prepared / "survey.npz"]
        relative.extend(prepared / "interfaces" / filename for filename in (
            "Sea_Bottom-mr3d_reverted.xyz", "Miocene-mr3d_reverted.xyz", "Oligocene-mr3d_reverted.xyz",
            "Blue_mark-mr3d_reverted.xyz", "Top_of_Salt-mr3d_reverted.xyz", "Base_of_salt-mr3d_reverted.xyz"))
    else:
        return
    for path in relative:
        target, source = data_root / path, DATA_ROOT / path
        if target.is_file():
            continue
        if source.is_file() and source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            raise FileNotFoundError(f"Missing mesh-generation input {target}. See the root README for the MR3D interface download/layout.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=(*mesh_names(), "mr3d", "all"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List final paths without reading data")
    mode.add_argument("--dry-run", action="store_true", help="Print the explicit generation stages")
    mode.add_argument("--check", action="store_true", help="Independently recount all final H5 edges")
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    selected = list(mesh_names()) if args.case in (None, "all") else (
        [f"mr3d_{level}" for level in MR3D_SCALES] if args.case == "mr3d" else [args.case])
    if args.list or (args.case is None and not (args.check or args.dry_run)):
        print(json.dumps({name: {key: str(value) for key, value in get_mesh(name, data_root=data_root).items()}
                          for name in selected}, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({name: commands(name, data_root) for name in selected}, indent=2))
        return 0
    from mesh_gen.verify import check_mesh
    results = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(key, "1")
    for name in selected:
        if not args.check:
            paths = get_mesh(name, data_root=data_root)
            existing = paths.get("mesh", paths.get("arrays"))
            # The mesh-only emg3d entry also validates/restores receiver sidecars.
            if name == "block_emg3d" or not existing.exists():
                prepare_inputs(name, data_root)
                for command in commands(name, data_root):
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
        result = check_mesh(name, data_root=data_root)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Mesh generation/check failed: {error}", file=sys.stderr)
        raise SystemExit(2)
