"""Minimal frozen MR3D model sampling rules from the established experiment."""
from pathlib import Path
import json
import numpy as np

MU0 = 4 * np.pi * 1e-7

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

def source_current(endpoints, source_moment):
    """Convert an electric source moment in A m to actual line current in A."""
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
        raise ValueError("source_endpoints must be a finite (2, 3) array")
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    if length <= 0:
        raise ValueError("Source length must be positive")
    if not np.isfinite(source_moment) or source_moment <= 0:
        raise ValueError("source_moment must be finite and positive")
    return float(source_moment / length)

class PreparedConductivity:
    """Read-only mmap model; interpolate log10 resistivity in bounded chunks."""

    def __init__(self, prepared):
        from scipy.interpolate import RegularGridInterpolator

        prepared = Path(prepared)
        manifest = require_prepared(prepared)
        with np.load(prepared / "axes.npz") as axes:
            self.axes = tuple(np.array(axes[key], dtype=float) for key in "xyz")
        if any(axis.ndim != 1 or len(axis) < 2 or
               not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0)
               for axis in self.axes):
            raise ValueError("Model axes must be finite, strictly increasing 1D vectors")
        self.log_rho_h = np.load(prepared / "log_rho_h.npy", mmap_mode="r")
        self.log_rho_v = np.load(prepared / "log_rho_v.npy", mmap_mode="r")
        shape = tuple(map(len, self.axes))
        if tuple(manifest["shape"]) != shape:
            raise ValueError("Prepared manifest shape disagrees with model axes; rerun prepare")
        if self.log_rho_h.shape != shape or self.log_rho_v.shape != shape:
            raise ValueError("Resistivity array shape must match model axes (x, y, z)")
        # Float32 mmap stays shared: never construct a whole-model float64/log copy.
        self.interp_h = RegularGridInterpolator(self.axes, self.log_rho_h)
        self.interp_v = RegularGridInterpolator(self.axes, self.log_rho_v)

    def sample(self, points, markers, chunk_size=100_000):
        """Return horizontal/vertical S/m using author air/water extension rules."""
        points = np.asarray(points, dtype=float)
        markers = np.asarray(markers)
        if points.ndim != 2 or points.shape[1] != 3 or len(markers) != len(points):
            raise ValueError("Expected points (n, 3) and markers (n,)")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        horizontal = np.empty(len(points), dtype=float)
        vertical = np.empty(len(points), dtype=float)
        lower = np.array([axis[0] for axis in self.axes])
        upper = np.array([axis[-1] for axis in self.axes])
        for first in range(0, len(points), chunk_size):
            last = min(first + chunk_size, len(points))
            xyz = points[first:last]
            effective = markers[first:last].copy()
            radius = np.max(np.abs(xyz[:, :2]), axis=1)
            # Original overwrite_markers(): material beneath the outward-sloping
            # bathymetry is ground even if the coarse outer mesh marks it water.
            ground_extension = ((radius > 1.14e4) &
                                (xyz[:, 2] < -radius / 20. - 200.))
            effective[ground_extension | (xyz[:, 2] < -1.14e4)] = 99
            sh = np.full(len(xyz), 1e-8)
            sv = sh.copy()
            water = effective == 1
            ground = effective > 1
            sh[water] = sv[water] = 1 / .32
            deep = ground & (xyz[:, 2] < lower[2])
            sh[deep], sv[deep] = 1 / 38., 1 / 76.
            sampled = ground & ~deep
            if sampled.any():
                clipped = np.clip(xyz[sampled], lower, upper)
                sh[sampled] = np.power(10., -self.interp_h(clipped))
                sv[sampled] = np.power(10., -self.interp_v(clipped))
            if not (np.isfinite(sh).all() and np.isfinite(sv).all() and
                    (sh > 0).all() and (sv > 0).all()):
                raise ValueError("Interpolated conductivity must be finite and positive")
            horizontal[first:last], vertical[first:last] = sh, sv
        return horizontal, vertical
