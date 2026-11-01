"""Keep numerical tests on the intended package and an external custEM tree."""
import inspect
import os
from pathlib import Path
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "core"
INSTALLED = os.environ.get("SUPERMOR_TEST_INSTALLED") == "1"
if not INSTALLED:
    sys.path.insert(0, str(CODE))


def pytest_sessionstart(session):
    """Reject local custEM copies and mixed source/installed namespaces."""
    import custEM
    from custEM.core import MOD
    import supermor
    from supermor import full_order, materials, methods, mor, mor_high_order, operators

    own_root = Path(supermor.__file__).resolve().parent
    if INSTALLED:
        if own_root.is_relative_to(ROOT):
            raise pytest.UsageError(f"Installed validation loaded repository source: {own_root}")
    elif own_root != CODE / "supermor":
        raise pytest.UsageError(f"Tests must use {CODE / 'supermor'}; loaded {own_root}")
    for module in (supermor, full_order, materials, methods, mor, mor_high_order, operators):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(own_root):
            raise pytest.UsageError(f"Mixed supermor source trees: {module.__name__} from {path}")

    upstream_root = Path(custEM.__file__).resolve().parent
    if upstream_root.is_relative_to(ROOT):
        raise pytest.UsageError(f"custEM must be an external official dependency: {upstream_root}")
    expected = os.environ.get("CUSTEM_EXPECTED_ROOT")
    if expected:
        expected = Path(expected).resolve()
        if (expected / "custEM" / "__init__.py").is_file():
            expected = expected / "custEM"
        if upstream_root != expected:
            raise pytest.UsageError(f"Expected custEM from {expected}; loaded {upstream_root}")
    parameters = inspect.signature(MOD.solve_main_problem).parameters
    if "mor" in parameters or "mor_options" in parameters:
        raise pytest.UsageError("custEM MOD has the research checkout's modified MOR interface")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("custEM.") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(upstream_root):
                raise pytest.UsageError(f"Mixed custEM source trees: {name} from {path}")


@pytest.fixture
def mpi_tmp_path():
    """One shared temporary directory for collective mesh/model file I/O."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    temporary = tempfile.TemporaryDirectory(prefix="supermor-core-") if comm.rank == 0 else None
    path = comm.bcast(temporary.name if temporary is not None else None, root=0)
    try:
        yield Path(path)
    finally:
        comm.Barrier()
        if temporary is not None:
            temporary.cleanup()
        comm.Barrier()
