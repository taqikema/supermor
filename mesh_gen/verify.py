"""Independent bounded-memory checks of the meshes used by the paper."""
import gc
import hashlib
from pathlib import Path
import tempfile

from .paths import EXPECTED, get_mesh


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tetrahedral_counts(path):
    """Count all undirected edges directly from H5 tetrahedral connectivity."""
    import h5py
    import numpy as np

    chunk = 150000
    with h5py.File(path, "r") as stream:
        cells, xyz = stream["mesh/topology"], stream["mesh/coordinates"]
        nc, nv = len(cells), len(xyz)
        if nc == 0 or nv == 0 or cells.shape != (nc, 4) or xyz.shape != (nv, 3):
            raise ValueError("Expected a nonempty three-dimensional tetrahedral mesh")
        if nv >= 2**32:
            raise ValueError("Mesh exceeds the edge counter's uint32 vertex range")
        for start in range(0, nv, chunk):
            if not np.isfinite(xyz[start:start + chunk]).all():
                raise ValueError("Nonfinite mesh coordinates")
        with tempfile.TemporaryDirectory(prefix="supermor-edge-check-") as scratch:
            packed = np.memmap(Path(scratch) / "edges.dat", mode="w+", dtype=np.uint64,
                               shape=(6 * nc,))
            cursor = 0
            for start in range(0, nc, chunk):
                block = np.asarray(cells[start:start + chunk], dtype=np.int64)
                if block.min() < 0 or block.max() >= nv:
                    raise ValueError("Mesh connectivity contains an invalid vertex index")
                if np.any(np.diff(np.sort(block, axis=1), axis=1) <= 0):
                    raise ValueError("A tetrahedron contains duplicate vertices")
                for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                    low = np.minimum(block[:, a], block[:, b]).astype(np.uint64)
                    high = np.maximum(block[:, a], block[:, b]).astype(np.uint64)
                    packed[cursor:cursor + len(block)] = (low << np.uint64(32)) | high
                    cursor += len(block)
            packed.sort(kind="quicksort")
            edges, previous = 0, None
            for start in range(0, len(packed), 6 * chunk):
                block = np.asarray(packed[start:start + 6 * chunk])
                edges += int(np.count_nonzero(block[1:] != block[:-1]))
                edges += int(previous is None or block[0] != previous)
                previous = int(block[-1])
            del block, packed
            gc.collect()
    return {"cells": nc, "vertices": nv, "edges": edges, "real_P1_dof": 2 * edges}


def check_mesh(name, *, data_root=None):
    """Check a final generated mesh; raise on any size or input error."""
    files = get_mesh(name, require=True, data_root=data_root)
    if name == "block_emg3d":
        from .case1_block.cross_validation.emg3d_worker import prepare_mesh, prepare_supports
        generation_root = files["m_dir"].parents[1]
        arrays, metadata = prepare_mesh(generation_root, create=False)
        _, supports = prepare_supports(generation_root, arrays, create=False)
        result = {"shape": metadata["shape_cells"], "cells": metadata["cells"],
                  "arrays_sha256": sha256(files["arrays"]),
                  "receiver_support_sha256": supports["npz_sha256"],
                  "receiver_support_checked": True}
    else:
        result = tetrahedral_counts(files["mesh"])
        actual = result["cells"], result["edges"], result["real_P1_dof"]
        if actual != EXPECTED[name]:
            raise ValueError(f"{name}: actual counts {actual} differ from paper {EXPECTED[name]}")
        result["mesh_sha256"] = sha256(files["mesh"])
    return {"name": name, "passed": True, **result,
            "files": {key: str(value) for key, value in files.items()}}
