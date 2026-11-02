#!/usr/bin/env python3
"""Generate the MR3D meshes and prepare their inputs outside the code tree."""

import argparse
import importlib
import json
import math
from pathlib import Path
import sys


# Keep both direct-script and `python -m` execution usable from any directory.
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

LEVELS = ("M1", "M2", "M3", "M4", "M5")
DEFAULT_WORKDIR = Path(__file__).resolve().parents[4] / "data" / "mesh_data" / "case3_mr3d"


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR,
                        help="MR3D data directory containing prepared/ and meshes/")
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("fetch", "prepare"):
        command = stages.add_parser(stage, help=f"{stage} official model inputs")
        command.add_argument("--model", choices=("extended", "original"), default="original")
    mesh = stages.add_parser("mesh", help="generate and measure approximate M1–M5 meshes")
    mesh.add_argument("--level", choices=LEVELS + ("all",), default="M1")
    mesh.add_argument("--target-dof", type=int)
    mesh.add_argument("--tolerance", type=positive_number, default=0.15)
    mesh.add_argument("--max-attempts", type=int, default=4)
    mesh.add_argument("--initial-scale", type=positive_number)
    mesh.add_argument("--quality", type=positive_number, default=2.0)
    mesh.add_argument("--half-extent", type=positive_number, default=11400.0)
    stages.add_parser("check", help="check the meshing environment and prepared geometry inputs")
    return parser


def check_runtime(workdir):
    dependencies = {}
    for name in ("numpy", "scipy", "h5py", "pygimli", "dolfin", "custEM", "mpi4py"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as error:
            dependencies[name] = {"ok": False, "error": str(error)}
    from experiments.mr3d_custem.mesh import INTERFACES
    required = ("survey.npz", "manifest.json", *("interfaces/" + name for name in INTERFACES))
    prepared = {name: (workdir / "prepared" / name).is_file() for name in required}
    from experiments.mr3d_custem.prepare import require_prepared
    try:
        require_prepared(workdir / "prepared")
        prepared["complete_manifest"] = True
    except ValueError:
        prepared["complete_manifest"] = False
    try:
        import numpy as np
        with np.load(workdir / "prepared" / "survey.npz", allow_pickle=False) as survey:
            values = {key: np.asarray(survey[key], dtype=float)
                      for key in ("inline", "broadside", "receiver", "source_endpoints")}
        valid = all(np.isfinite(value).all() for value in values.values())
        valid = valid and all(values[key].ndim == 2 and values[key].shape[1] == 3
                              and len(values[key]) > 0 for key in ("inline", "broadside"))
        prepared["survey_geometry_valid"] = bool(valid and values["receiver"].shape == (3,)
                                                 and values["source_endpoints"].shape == (2, 3))
    except (ImportError, OSError, ValueError, KeyError):
        prepared["survey_geometry_valid"] = False
    report = {"workdir": str(workdir), "dependencies": dependencies, "prepared": prepared}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(v["ok"] for v in dependencies.values()) and all(prepared.values()) else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    workdir = args.workdir.expanduser().resolve()
    try:
        if args.stage == "fetch":
            from experiments.mr3d_custem.prepare import fetch_data
            print(fetch_data(workdir, model=args.model))
        elif args.stage == "prepare":
            from experiments.mr3d_custem.prepare import prepare_data
            print(prepare_data(workdir, model=args.model))
        elif args.stage == "mesh":
            from experiments.mr3d_custem.mesh import build_mesh
            if args.level == "all" and args.target_dof is not None:
                raise ValueError("--target-dof requires one --level, not all")
            for level in LEVELS if args.level == "all" else (args.level,):
                result = build_mesh(
                    workdir, level=level, target_dof=args.target_dof,
                    tolerance=args.tolerance, max_attempts=args.max_attempts,
                    initial_scale=args.initial_scale, quality=args.quality,
                    half_extent=args.half_extent,
                )
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            return check_runtime(workdir)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"MR3D: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
