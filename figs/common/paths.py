"""Checkout-relative figure paths and bounded-memory input fingerprints."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
