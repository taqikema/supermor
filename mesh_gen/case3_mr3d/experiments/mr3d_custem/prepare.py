"""Download and prepare the two official MR3D resistivity models.

The output arrays are ordered (x, y, z) and use local metres with z positive
upwards. Original data retain the actual SEG-Y sample coordinates; extended
data retain the reference cell-centre convention. Large arrays are streamed.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile

import h5py
import numpy as np


ORIGINAL_URL = "https://zenodo.org/records/400233"
EXTENDED_URL = "https://zenodo.org/records/3748492"
SURVEY_URL = (
    "https://raw.githubusercontent.com/swung-research/"
    "3d-csem-open-source-landscape/main/model-marlim/marlim_survey.nc"
)
TRANSLATION = np.array([-390275.0, -7518065.0, 0.0])
OFFICIAL_FREQUENCY = np.array([0.125, 0.25, 0.5, 0.75, 1.0, 1.25])
FREQUENCY = np.arange(2, 21, dtype=float) / 16.0
INTERFACES = (
    "Sea_Bottom-mr3d.xyz", "Miocene-mr3d.xyz", "Oligocene-mr3d.xyz",
    "Blue_mark-mr3d.xyz", "Top_of_Salt-mr3d.xyz", "Base_of_salt-mr3d.xyz",
)
MODELS = {
    "original": {
        "url": ORIGINAL_URL,
        "filenames": ("Horizontal_resistivity.sgy", "Vertical_Resistivity.sgy"),
        "widths": (25, 75, 5),
        "reference_transform": "cube[::-1, :, ::-1].transpose(1, 0, 2)",
    },
    "extended": {
        "url": EXTENDED_URL,
        "archive": "MR3D_20meters_Mesh.zip",
        "filenames": ("Mrl3d_Rh_20metersmesh.segy", "Mrl3d_Rv_20metersmesh.segy"),
        "widths": (100, 100, 20),
        "reference_transform": "cube[:, :, ::-1]",
    },
}


def _model_config(model):
    try:
        return MODELS[model]
    except KeyError as error:
        raise ValueError("model must be 'original' or 'extended'") from error


def _report_progress(partial: Path) -> None:
    if partial.exists():
        print(f"[fetch] {partial.name.removesuffix('.part')}: "
              f"{partial.stat().st_size / 1024**2:.1f} MiB", flush=True)


def _powershell_download(url: str, partial: Path) -> None:
    """Use Windows networking when the WSL HTTPS stack is unavailable."""
    windows_path = subprocess.run(
        ["wslpath", "-w", str(partial.resolve())], check=True,
        capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    # PowerShell single-quoted literals cannot interpolate $, backticks, or $().
    literal_url = "'" + url.replace("'", "''") + "'"
    literal_path = "'" + windows_path.replace("'", "''") + "'"
    script = (
        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
        f"Invoke-WebRequest -UseBasicParsing -Uri {literal_url} "
        f"-OutFile {literal_path} -TimeoutSec 3600"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    stopped = threading.Event()
    def monitor():
        while not stopped.wait(10):
            _report_progress(partial)
    progress = threading.Thread(target=monitor, daemon=True)
    progress.start()
    try:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=True, capture_output=True, text=True, errors="replace", timeout=3660,
        )
    finally:
        stopped.set()
        progress.join()


def _download(url: str, destination: Path) -> None:
    """Stream to a temporary sibling, then publish only a complete download."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        print(f"[fetch] reuse {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"[fetch] downloading {destination.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "MR3D-custEM/1.0"})
    windows_available = bool(os.environ.get("WSL_DISTRO_NAME")) and url.startswith("https://")
    started = last_report = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            expected = response.headers.get("Content-Length")
            total = 0
            with partial.open("wb") as stream:
                while chunk := response.read(64 * 1024):
                    stream.write(chunk)
                    total += len(chunk)
                    now = time.monotonic()
                    elapsed = now - started
                    if windows_available and elapsed > 15 and total / elapsed < 128 * 1024:
                        raise TimeoutError("WSL transfer is slower than 128 KiB/s")
                    if now - last_report > 10:
                        _report_progress(partial)
                        last_report = now
            if expected is not None and total != int(expected):
                raise OSError(f"Incomplete download: {url} ({total}/{expected} bytes)")
    except OSError as error:
        if not windows_available:
            raise
        print(f"[fetch] {destination.name}: Windows network fallback ({error})", flush=True)
        _powershell_download(url, partial)
    if not partial.is_file() or not partial.stat().st_size:
        raise OSError(f"Empty download: {url}")
    partial.replace(destination)
    print(f"[fetch] complete {destination.name} "
          f"({destination.stat().st_size / 1024**2:.1f} MiB)", flush=True)

def _extract_models(archive: Path, destination: Path, names) -> None:
    """Flatten only the required SEG-Y members after validating archive paths."""
    destination = Path(destination)
    with zipfile.ZipFile(archive) as zipped:
        members = {}
        for info in zipped.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
                raise ValueError(f"unsafe archive member: {info.filename}")
            if not info.is_dir() and path.name in names:
                if path.name in members:
                    raise ValueError(f"duplicate archive model: {path.name}")
                members[path.name] = info
        missing = set(names) - members.keys()
        if missing:
            raise ValueError(f"Archive is missing required models: {sorted(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = destination / name
            if target.is_file() and target.stat().st_size == members[name].file_size:
                continue
            partial = target.with_name(target.name + ".part")
            with zipped.open(members[name]) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            partial.replace(target)


def fetch_data(workdir: Path, model: str = "extended") -> Path:
    """Fetch public files with at most three simultaneous transfers."""
    config = _model_config(model)
    raw = Path(workdir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    downloads = []
    archive = None
    if "archive" in config:
        if not all((raw / name).is_file() and (raw / name).stat().st_size
                   for name in config["filenames"]):
            archive = raw / config["archive"]
            downloads.append((f"{config['url']}/files/{archive.name}?download=1", archive))
    else:
        downloads.extend((f"{config['url']}/files/{name}?download=1", raw / name)
                         for name in config["filenames"])
    downloads.append((SURVEY_URL, raw / "marlim_survey.nc"))
    downloads.extend((f"{ORIGINAL_URL}/files/{name}?download=1", raw / name)
                     for name in INTERFACES)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_download, url, target) for url, target in downloads]
        for future in as_completed(futures):
            future.result()
    if archive is not None:
        print(f"[fetch] extracting {archive.name}", flush=True)
        _extract_models(archive, raw, config["filenames"])
    return raw

def scale_segy_coordinates(coordinates, scalars):
    """Apply SEG-Y coordinate scalars: multiply positive, divide negative."""
    scalars = np.asarray(scalars, dtype=float)
    factor = np.where(scalars < 0, 1. / np.maximum(abs(scalars), 1),
                      np.maximum(scalars, 1))
    return np.asarray(coordinates, dtype=float) * factor


def _trace_layout(cdp_x, cdp_y, widths, scalars=-10, model="extended"):
    """Validate trace geometry; return local sample coordinates and indices.

    Original coordinates retain the native header positions and their 0.1 m
    quantization. Extended coordinates keep the reference half-cell offsets.
    Header mapping supports either trace order without loading the full cube.
    """
    trace_x = scale_segy_coordinates(cdp_x, scalars)
    trace_y = scale_segy_coordinates(cdp_y, scalars)
    if trace_x.size == 0 or trace_x.shape != trace_y.shape:
        raise ValueError("SEG-Y coordinate arrays must be nonempty and have equal shape")
    if not np.isfinite(trace_x).all() or not np.isfinite(trace_y).all():
        raise ValueError("SEG-Y coordinates must be finite")
    x, ix = np.unique(trace_x, return_inverse=True)
    y, iy = np.unique(trace_y, return_inverse=True)
    tolerance = .100001 if model == "original" else 1e-6
    for axis, spacing in zip((x, y), widths[:2]):
        if not np.allclose(np.diff(axis), spacing, rtol=0, atol=tolerance):
            raise ValueError(f"SEG-Y coordinate spacing does not match {spacing} m")
    if np.unique(ix * y.size + iy).size != trace_x.size:
        raise ValueError("SEG-Y contains duplicate x/y traces")
    if x.size * y.size != trace_x.size:
        raise ValueError("SEG-Y traces do not form a complete rectangular grid")
    offset = 0 if model == "original" else .5
    x = x + offset * widths[0] + TRANSLATION[0]
    y = y + offset * widths[1] + TRANSLATION[1]
    return x, y, ix, iy


def _survey_arrays(source: Path) -> dict:
    with h5py.File(source, "r") as handle:
        repeated_x = np.asarray(handle["src_x"], dtype=float)
        if repeated_x.size % 2 or not np.array_equal(repeated_x[::2], repeated_x[1::2]):
            raise ValueError("Official survey src_x must contain real/imaginary pairs")
        x = repeated_x[::2]
        frequency = np.asarray(handle["freqs"], dtype=float)
        if (frequency.shape != OFFICIAL_FREQUENCY.shape or
                not np.allclose(frequency, OFFICIAL_FREQUENCY)):
            raise ValueError("Official survey must contain the six MR3D frequencies")
        if not np.isclose(float(handle.attrs["strength"]), 1.0):
            raise ValueError("Official survey must be normalized to source strength 1")
        arrays = {}
        for name, key in (("data_il", "inline"), ("data_bs", "broadside")):
            attrs = handle[name].attrs
            y = np.broadcast_to(np.asarray(attrs["src_y"], dtype=float), x.shape)
            z = np.broadcast_to(np.asarray(attrs["src_z"], dtype=float), x.shape)
            arrays[key] = np.column_stack((x, y, z)) + TRANSLATION
        receiver = np.array([float(handle.attrs[f"rec_{axis}"]) for axis in "xyz"])
    receiver += TRANSLATION
    arrays["receiver"] = receiver
    arrays["source_endpoints"] = receiver + np.array([[-5, 0, 0], [5, 0, 0]])
    arrays["frequency"] = FREQUENCY.copy()
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Survey coordinates must be finite")
    return arrays


def _write_log_model(source: Path, destination: Path, widths, model="extended"):
    try:
        import segyio
    except ImportError as error:
        raise ImportError(
            "SEG-Y preparation requires segyio. Install it in the active Python "
            "environment with: python -m pip install segyio"
        ) from error
    with segyio.open(str(source), "r", ignore_geometry=True) as segy:
        x, y, ix, iy = _trace_layout(
            segy.attributes(segyio.TraceField.CDP_X)[:],
            segy.attributes(segyio.TraceField.CDP_Y)[:], widths,
            scalars=segy.attributes(segyio.TraceField.SourceGroupScalar)[:], model=model,
        )
        samples = np.asarray(segy.samples, dtype=float)
        depth_start = 60 if model == "original" else 0
        if (samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all()
                or not np.isclose(samples[0], depth_start, rtol=0, atol=1e-6)
                or not np.all(np.diff(samples) > 0)
                or not np.allclose(np.diff(samples), widths[2], rtol=0, atol=1e-6)):
            raise ValueError(
                f"SEG-Y sample axis must be finite, start at {depth_start}, and increase by "
                f"{widths[2]} m: {source}"
            )
        nz = samples.size
        offset = 0 if model == "original" else .5
        z = -(samples[::-1] + offset * widths[2])
        partial = destination.with_name(destination.name + ".part")
        output = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(x.size, y.size, nz),
        )
        # A block bounds RAM use to about 16 MiB even for the original fine model.
        traces_per_block = max(1, 4 * 1024 * 1024 // nz)
        for start in range(0, segy.tracecount, traces_per_block):
            stop = min(start + traces_per_block, segy.tracecount)
            values = segy.trace.raw[start:stop]
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"SEG-Y resistivity must be finite and positive: {source}")
            np.log10(values, out=values)
            output[ix[start:stop], iy[start:stop], :] = values[:, ::-1]
        output.flush()
        del output
    partial.replace(destination)
    return x, y, z


def require_prepared(prepared: Path) -> dict:
    """Reject missing, interrupted, or invalid preparation completion markers."""
    marker = Path(prepared) / "manifest.json"
    message = f"Prepared MR3D data are incomplete or invalid; rerun prepare: {marker}"
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(message) from error
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(message)
    shape = manifest.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError(message)
    return manifest

def prepare_data(workdir: Path, model: str = "extended") -> Path:
    """Prepare existing raw inputs; fetch_data is a separate explicit operation."""
    config = _model_config(model)
    workdir = Path(workdir)
    raw, prepared = workdir / "raw", workdir / "prepared"
    required = ["marlim_survey.nc", *INTERFACES, *config["filenames"]]
    missing = [name for name in required if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw MR3D inputs; run fetch first: {', '.join(missing)}")
    prepared.mkdir(parents=True, exist_ok=True)
    # Invalidate the completion marker before replacing any prepared products.
    (prepared / "manifest.json").unlink(missing_ok=True)
    np.savez(prepared / "survey.npz", **_survey_arrays(raw / "marlim_survey.nc"))
    interface_dir = prepared / "interfaces"
    interface_dir.mkdir(exist_ok=True)
    for name in INTERFACES:
        xyz = np.loadtxt(raw / name, ndmin=2)
        if xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError(f"Interface must contain three finite XYZ columns: {name}")
        xyz[:, 2] *= -1
        np.savetxt(interface_dir / f"{Path(name).stem}_reverted.xyz", xyz, fmt="%.10g")
    axes = None
    for filename, component in zip(config["filenames"], ("h", "v")):
        current = _write_log_model(raw / filename, prepared / f"log_rho_{component}.npy",
                                   config["widths"], model=model)
        if axes is not None and not all(np.array_equal(a, b) for a, b in zip(axes, current)):
            raise ValueError("Horizontal and vertical SEG-Y coordinate grids do not match")
        axes = current
    np.savez(prepared / "axes.npz", **dict(zip("xyz", axes)))
    manifest = {
        "status": "complete",
        "model": model,
        "model_url": config["url"],
        "interfaces_url": ORIGINAL_URL,
        "survey_url": SURVEY_URL,
        "model_files": list(config["filenames"]),
        "cell_widths_m": list(config["widths"]),
        "shape": [len(axis) for axis in axes],
        "array_order": "x,y,z; all axes ascending",
        "coordinate_positions": ("actual SEG-Y samples" if model == "original" else
                                 "reference cell centres with half-cell offsets"),
        "resistivity_encoding": "log10(ohm m), float32, uncompressed .npy",
        "translation_m": TRANSLATION.tolist(),
        "coordinate_scale": "CDP_X and CDP_Y decoded using each trace SourceGroupScalar",
        "coordinate_spacing_tolerance_m": .100001 if model == "original" else 1e-6,
        "z_convention": "positive upwards; negative actual sample depth" if model == "original" else "positive upwards; reference model top z=0",
        "axis_bounds_m": {key: [float(axis[0]), float(axis[-1])]
                          for key, axis in zip("xyz", axes)},
        "reference_cube_transform": config["reference_transform"],
        "trace_order": "validated and mapped by actual CDP_X/CDP_Y headers",
        "interfaces_coordinates": "original UTM x/y; third column negated",
        "source_strength": 1.0,
        "source_length_m": 10.0,
        "frequencies_hz": FREQUENCY.tolist(),
        "deep_resistivity_ohm_m": {"horizontal": 38, "vertical": 76},
        "outside_model": "clip x/y to nearest model boundary; use deep resistivity below bottom",
    }
    marker = prepared / "manifest.json"
    partial = marker.with_name(marker.name + ".part")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.replace(marker)
    return prepared
