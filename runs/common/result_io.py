"""Small result files and source/input identities, independent of FEniCS."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    nonfinite = []

    def clean(item, location=""):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            nonfinite.append(location)
            return None
        if isinstance(item, dict):
            return {str(k): clean(v, location + "." + str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v, location + "." + str(i)) for i, v in enumerate(item)]
        return item

    result = clean(value)
    if nonfinite and isinstance(result, dict):
        result["nonfinite_values"] = nonfinite
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def output_directory(case, experiment, *, output=None):
    if output is None:
        parent = ROOT / "outputs" / case / experiment
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(tempfile.mkdtemp(prefix=stamp + "_", dir=parent))
    destination = Path(output).expanduser().resolve()
    if destination == ROOT / "data" or destination.is_relative_to(ROOT / "data"):
        raise ValueError("Numerical output must be outside the input data directory")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output directory is not empty: " + str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_receivers(path, frequencies, receivers, coordinates):
    indices = sorted(receivers)
    shape = (0, len(coordinates), 3)
    np.savez(path, indices=np.asarray(indices, dtype=int),
             frequencies_hz=np.asarray(frequencies)[indices], rx_xyz=coordinates,
             E=np.asarray([receivers[i]["E"] for i in indices]) if indices else np.empty(shape, complex),
             H=np.asarray([receivers[i]["H"] for i in indices]) if indices else np.empty(shape, complex))


def input_identity(paths):
    return {key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items() if Path(path).is_file()}
