"""Prepare original MR3D material arrays under data/model_data/case3_mr3d.

Only the two original SEG-Y resistivity volumes are needed for this stage.
Survey/mesh geometry remains under data/mesh_data/case3_mr3d. The pinned
mesh preparation module supplies the unchanged SEG-Y decoding/axis rules.
Fetching is explicit (--fetch); --check only inspects local availability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--raw-dir", type=Path,
                        help="Directory containing Horizontal_resistivity.sgy and Vertical_Resistivity.sgy")
    parser.add_argument("--output", type=Path, help="Prepared material directory (default: data/model_data/case3_mr3d)")
    parser.add_argument("--fetch", action="store_true", help="Explicitly download the two original SEG-Y volumes")
    parser.add_argument("--check", action="store_true", help="Report existing files without fetching/preparing")
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    raw = args.raw_dir.expanduser().resolve() if args.raw_dir else data_root / "raw_data/case3_mr3d"
    output = args.output.expanduser().resolve() if args.output else data_root / "model_data/case3_mr3d"
    filenames = ("Horizontal_resistivity.sgy", "Vertical_Resistivity.sgy")
    products = ("manifest.json", "axes.npz", "log_rho_h.npy", "log_rho_v.npy")
    if args.check:
        print(json.dumps(dict(raw_dir=str(raw), material_dir=str(output),
            raw_files={name: (raw / name).is_file() for name in filenames},
            material_files={name: (output / name).is_file() for name in products}), indent=2))
        return 0 if all((output / name).is_file() for name in products) else 1
    from mesh_gen.case3_mr3d.experiments.mr3d_custem import prepare
    if args.fetch:
        raw.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            prepare._download(f"{prepare.ORIGINAL_URL}/files/{name}?download=1", raw / name)
    missing = [str(raw / name) for name in filenames if not (raw / name).is_file()]
    if missing:
        parser.error("Missing original MR3D SEG-Y inputs; supply --raw-dir or use --fetch: " + ", ".join(missing))
    if output.exists() and any(output.iterdir()):
        parser.error("Material output is not empty; select a new --output to preserve prepared inputs")
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from runs.common.result_io import input_identity, write_json
    config = prepare.MODELS["original"]
    axes = None
    for filename, component in zip(filenames, ("h", "v")):
        current = prepare._write_log_model(raw / filename, output / f"log_rho_{component}.npy",
                                           config["widths"], model="original")
        if axes is not None and not all(np.array_equal(a, b) for a, b in zip(axes, current)):
            raise ValueError("Horizontal and vertical SEG-Y coordinate grids do not match")
        axes = current
    np.savez(output / "axes.npz", **dict(zip("xyz", axes)))
    # The completion marker is published last; no mesh/survey products are replaced.
    manifest = dict(status="complete", model="original", model_url=prepare.ORIGINAL_URL,
        model_files=list(filenames), shape=[len(axis) for axis in axes],
        array_order="x,y,z; all axes ascending", cell_widths_m=list(config["widths"]),
        resistivity_encoding="log10(ohm m), float32, uncompressed .npy",
        translation_m=prepare.TRANSLATION.tolist(),
        coordinate_positions="actual SEG-Y samples", coordinate_spacing_tolerance_m=.100001,
        z_convention="positive upwards; negative actual sample depth",
        axis_bounds_m={key: [float(axis[0]), float(axis[-1])] for key, axis in zip("xyz", axes)},
        reference_cube_transform=config["reference_transform"],
        outside_model="clip x/y to nearest model boundary; use deep resistivity below bottom",
        deep_resistivity_ohm_m=dict(horizontal=38, vertical=76),
        source_inputs=input_identity({name: raw / name for name in filenames}),
        preparation_source_sha256=hashlib.sha256(Path(prepare.__file__).read_bytes()).hexdigest())
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(status="prepared", material_dir=str(output), shape=manifest["shape"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
