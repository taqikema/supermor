"""Local P2 exact-sequence transfers for tetrahedral MOR auxiliary spaces."""
from functools import lru_cache
from time import perf_counter

import dolfin as df
import numpy as np
from ffc.fiatinterface import create_element
from mpi4py import MPI
from petsc4py import PETSc


def _collective_check(comm, error, exception=RuntimeError):
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        details = "; ".join("rank %d: %s" % (rank, message)
                            for rank, message in enumerate(errors) if message is not None)
        raise exception("P2 auxiliary transfer construction failed: " + details)


def _require_connected(mesh):
    """Check the scalar gauge premise; this does not check topological holes."""
    comm, error = mesh.mpi_comm(), None
    sentinel = np.iinfo(np.int64).max
    try:
        parent = np.arange(mesh.num_vertices(), dtype=np.int64)

        def find(vertex):
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = int(parent[vertex])
            return vertex

        # Three unions per tetrahedron, without retaining a cell-edge graph.
        for cell in df.cells(mesh, "all"):
            vertices = cell.entities(0)
            root = find(int(vertices[0]))
            for vertex in vertices[1:]:
                parent[find(int(vertex))] = root
        for vertex in range(len(parent)):
            parent[vertex] = find(vertex)
        global_ids = mesh.topology().global_indices(0)
        if len(global_ids) != len(parent):
            raise ValueError("mesh global vertex indices are unavailable")
        labels = np.full(len(parent), sentinel, dtype=np.int64)
        np.minimum.at(labels, parent, global_ids)
        boundary, outgoing = {}, [[] for unused in range(comm.size)]
        for vertex, ranks in mesh.topology().shared_entities(0).items():
            gid, root = int(global_ids[vertex]), int(parent[vertex])
            boundary[gid] = root
            for rank in ranks:
                outgoing[rank].append((gid, root))
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error)

    # Only interface vertex labels travel between neighbors. Empty ranks
    # participate in the same collectives and contribute the neutral label.
    while True:
        messages = [[(gid, int(labels[root])) for gid, root in records]
                    for records in outgoing]
        received = comm.alltoall(messages)
        changed, error = False, None
        try:
            for records in received:
                for gid, label in records:
                    root = boundary[gid]
                    if label < labels[root]:
                        labels[root], changed = label, True
        except Exception as exc:
            error = str(exc)
        _collective_check(comm, error)
        if not comm.allreduce(changed, op=MPI.LOR):
            break
    minimum = comm.allreduce(int(labels.min(initial=sentinel)), op=MPI.MIN)
    different = bool(np.any(labels[parent] != minimum))
    if minimum == sentinel or comm.allreduce(different, op=MPI.LOR):
        raise ValueError("P2 auxiliary transfers require a single connected mesh component")


@lru_cache(maxsize=1)
def _reference_transfers():
    """Apply FIAT tangential point duals once on the reference tetrahedron."""
    scalar = create_element(df.FiniteElement("CG", df.tetrahedron, 2))
    vector = create_element(df.FiniteElement("N1curl", df.tetrahedron, 2))
    gradient = np.zeros((20, 10))
    interpolation = np.zeros((20, 10, 3))
    nodes = vector.dual.get_nodes()
    if scalar.space_dimension() != 10 or vector.space_dimension() != 20:
        raise ValueError("unsupported P2 reference element dimensions")
    for i, node in enumerate(nodes):
        if (type(node).__name__ not in ("PointEdgeTangentEvaluation", "PointFaceTangentEvaluation")
                or node.deriv_dict or len(node.get_point_dict()) != 1):
            raise ValueError("P2 N1curl requires single-point tangential FIAT duals")
        point, terms = next(iter(node.get_point_dict().items()))
        table = scalar.tabulate(1, [point])
        values = table[(0, 0, 0)][:, 0].copy()
        # Remove reference-tabulation roundoff only. Physical Pi coefficients
        # must retain their scale and are never subjected to an absolute cutoff.
        tolerance = 128*np.finfo(float).eps
        values[np.abs(values) <= tolerance*max(1., np.max(np.abs(values)))] = 0.
        for weight, component in terms:
            c = component[0]
            derivative = tuple(int(k == c) for k in range(3))
            gradient[i] += weight*table[derivative][:, 0]
            interpolation[i, :, c] += weight*values
    gradient[np.abs(gradient) <= tolerance*max(1., np.max(np.abs(gradient)))] = 0.
    gradient.setflags(write=False)
    interpolation.setflags(write=False)
    return gradient, interpolation


def build_p2_auxiliary(V, S):
    """Build owned AIJ transfers G: CG2 -> N1curl2 and Pi: [CG2]^3 -> N1curl2.

    Pi columns are node-major (``3*scalar_global_dof + component``). The
    returned ``destroy`` callback owns both matrices and is idempotent. Only
    local tetrahedra and arrays indexed by owned rows are retained during setup.
    """
    started = perf_counter()
    mesh, comm = V.mesh(), V.mesh().mpi_comm()
    error = None
    try:
        if V.ufl_element().family() != "Nedelec 1st kind H(curl)" or V.ufl_element().degree() != 2:
            raise ValueError("V must be N1curl degree 2")
        if (S.ufl_element().family() != "Lagrange" or S.ufl_element().degree() != 2
                or S.ufl_element().value_shape()):
            raise ValueError("S must be scalar CG degree 2")
        if S.mesh().id() != mesh.id():
            raise ValueError("V and S must use the same mesh")
        if (mesh.ufl_cell().cellname() != "tetrahedron"
                or mesh.geometry().dim() != 3 or mesh.topology().dim() != 3):
            raise ValueError("P2 transfers require a three-dimensional tetrahedral mesh")
        if not mesh.ordered():
            raise ValueError("P2 transfers require an ordered mesh before space creation")
        if V.element().space_dimension() != 20 or S.element().space_dimension() != 10:
            raise ValueError("P2 transfers require local N1curl2/CG2 dimensions 20/10")
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error, ValueError)

    _require_connected(mesh)
    error = None
    try:
        reference_g, reference_pi = _reference_transfers()
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error, ValueError)

    vdm, sdm = V.dofmap(), S.dofmap()
    vmap = vdm.tabulate_local_to_global_dofs()
    smap = sdm.tabulate_local_to_global_dofs()
    vfirst, vlast = vdm.ownership_range()
    sfirst, slast = sdm.ownership_range()
    nrows = vlast-vfirst
    chosen_cell = np.full(nrows, -1, dtype=np.int64)
    diagonal = [np.zeros(nrows, dtype=PETSc.IntType) for unused in range(2)]
    off_diagonal = [np.zeros(nrows, dtype=PETSc.IntType) for unused in range(2)]

    def cell_data(cell):
        rows = vmap[vdm.cell_dofs(cell.index())]
        columns = smap[sdm.cell_dofs(cell.index())]
        coordinates = np.asarray(cell.get_vertex_coordinates()).reshape(4, 3)
        jacobian = (coordinates[1:] - coordinates[0]).T
        pi = np.einsum("ijr,cr->ijc", reference_pi, jacobian).reshape(20, 30)
        return rows, (columns, (3*columns[:, None]+np.arange(3)).reshape(-1)), (reference_g, pi)

    # First traversal selects one incident cell per owned row and determines
    # its exact diagonal/off-diagonal support, without a global row dictionary.
    error = None
    try:
        for cell in df.cells(mesh, "all"):
            rows, columns, blocks = cell_data(cell)
            for i, row in enumerate(rows):
                row = int(row)
                if not vfirst <= row < vlast:
                    continue
                local_row = row-vfirst
                if chosen_cell[local_row] >= 0:
                    continue
                chosen_cell[local_row] = cell.index()
                for k, (cols, block) in enumerate(zip(columns, blocks)):
                    support = cols[block[i] != 0.]
                    factor = 1 if k == 0 else 3
                    inside = (support >= factor*sfirst) & (support < factor*slast)
                    diagonal[k][local_row] = np.count_nonzero(inside)
                    off_diagonal[k][local_row] = len(support)-diagonal[k][local_row]
        if np.any(chosen_cell < 0):
            raise ValueError("local cells do not cover every owned N1curl2 row")
    except Exception as exc:
        error = str(exc)
    _collective_check(comm, error)

    matrices = []
    try:
        for k in range(2):
            factor = 1 if k == 0 else 3
            matrices.append(PETSc.Mat().createAIJ(
                size=((nrows, V.dim()), (factor*(slast-sfirst), factor*S.dim())),
                nnz=(diagonal[k], off_diagonal[k]), comm=comm))
        error = None
        try:
            for cell in df.cells(mesh, "all"):
                rows, columns, blocks = cell_data(cell)
                for i, row in enumerate(rows):
                    row = int(row)
                    if not vfirst <= row < vlast or chosen_cell[row-vfirst] != cell.index():
                        continue
                    for matrix, cols, block in zip(matrices, columns, blocks):
                        support = block[i] != 0.
                        matrix.setValues(int(row), cols[support].astype(PETSc.IntType), block[i, support],
                                         addv=PETSc.InsertMode.INSERT_VALUES)
        except Exception as exc:
            error = str(exc)
        _collective_check(comm, error)
        for matrix in matrices:
            matrix.assemble()
        nnz = {name: int(matrix.getInfo()["nz_used"])
               for name, matrix in zip(("G", "Pi"), matrices)}
    except Exception:
        for matrix in matrices:
            matrix.destroy()
        raise

    G, Pi = matrices

    def destroy():
        G.destroy()
        Pi.destroy()

    return {"G": G, "Pi": Pi, "setup_seconds": perf_counter()-started,
            "nnz": nnz, "destroy": destroy}
