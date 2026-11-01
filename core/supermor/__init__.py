"""Paper solvers extending an unmodified, separately installed custEM."""

from .materials import assign_native_vti
from .methods import RK_OPTIONS, SUPERMOR_OPTIONS, solve, solve_model
from .operators import assemble_mor

__all__ = [
    "solve", "solve_model", "assemble_mor", "assign_native_vti",
    "SUPERMOR_OPTIONS", "RK_OPTIONS",
]
