"""Total-electric-field operators for the two MOR solvers.

All large objects stay distributed.  The source is integrated on mesh edges;
neither source assembly nor field assignment assumes mixed-space numbering.
"""

from time import perf_counter

import numpy as np
import dolfin as df
from mpi4py import MPI


def _positive_scalar(coefficient, comm, name):
    """Keep scalar materials positive, including rank-local invalid shapes."""
    local_error = None
    try:
        if getattr(coefficient, "ufl_shape", ()):
            raise ValueError("MOR requires isotropic scalar " + name)
        if isinstance(coefficient, df.Function):
            values = coefficient.vector().get_local()
        elif isinstance(coefficient, df.Constant):
            values = coefficient.values()
        else:
            if np.iscomplexobj(coefficient):
                raise ValueError("MOR requires real " + name)
            values = np.asarray([float(coefficient)])
        if np.iscomplexobj(values):
            raise ValueError("MOR requires real " + name)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("MOR requires finite positive " + name)
    except Exception as error:
        local_error = "rank %d: %s" % (comm.rank, error)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise ValueError("MOR material validation failed: " +
                         "; ".join(error for error in errors if error is not None))


def _validate_conductivity(coefficient, mesh):
    """Accept scalar sigma or a real SPD Constant/native DG0 tensor.

    Tensor values are reconstructed through scalar component dofmaps. This
    handles distributed numbering and symmetry-compressed TensorElements;
    reshaping a local coefficient vector into nine entries per cell does not.
    All failures, including incompatible shapes, are reported on every rank
    before the caller reaches collective finite-element assembly.
    """
    comm = mesh.mpi_comm()
    shape, space, local_error = None, None, None
    try:
        shape = tuple(getattr(coefficient, "ufl_shape", ()))
        if shape not in ((), (3, 3)):
            raise ValueError("MOR conductivity must be scalar or a 3x3 tensor")
        if shape:
            if isinstance(coefficient, df.Function):
                space = coefficient.function_space()
                element = space.ufl_element()
                if (space.mesh().id() != mesh.id() or
                        element.family() != "Discontinuous Lagrange" or
                        element.degree() != 0):
                    raise ValueError("MOR tensor conductivity must be a native DG0 function on this mesh")
            elif not isinstance(coefficient, df.Constant):
                raise ValueError("MOR tensor conductivity must be a real Constant or native DG0 function")
    except Exception as error:
        local_error = "rank %d: %s" % (comm.rank, error)
    records = comm.allgather((shape, local_error))
    errors = [error for _, error in records if error is not None]
    if errors:
        raise ValueError("MOR conductivity validation failed: " + "; ".join(errors))
    if any(other_shape != shape for other_shape, _ in records):
        raise ValueError("MOR conductivity shape must be identical on all ranks")
    if not shape:
        _positive_scalar(coefficient, comm, "conductivity")
        return

    def check_tensor(values):
        if np.iscomplexobj(values):
            raise ValueError("MOR tensor conductivity must be real")
        if not np.all(np.isfinite(values)):
            raise ValueError("MOR tensor conductivity must be finite")
        if not np.allclose(values, values.T, rtol=1e-12, atol=0.0):
            raise ValueError("MOR tensor conductivity must be symmetric")
        try:
            # Cholesky reads only one triangle. Check the actual symmetric
            # part so near-symmetry tolerance cannot hide an indefinite mode.
            # Scale first to avoid overflowing when finite entries are large.
            np.linalg.cholesky(.5*values + .5*values.T)
        except np.linalg.LinAlgError as error:
            raise ValueError("MOR tensor conductivity must be positive definite") from error

    local_error = None
    try:
        if space is None:
            check_tensor(np.asarray(coefficient.values()).reshape(3, 3))
        else:
            element = space.ufl_element()
            components = [element.extract_subelement_component((i, j))[0]
                          for i in range(3) for j in range(3)]
            component_maps = {index: space.sub(index).dofmap()
                              for index in set(components)}
            dofmap = space.dofmap()
            first, last = dofmap.ownership_range()
            local_values = coefficient.vector().get_local()
            if np.iscomplexobj(local_values):
                raise ValueError("MOR tensor conductivity must be real")
            for cell in df.cells(mesh):
                if cell.is_ghost():
                    continue
                entries = {}
                for index, component_map in component_maps.items():
                    dofs = component_map.cell_dofs(cell.index())
                    if len(dofs) != 1:
                        raise ValueError("Invalid MOR tensor DG0 component layout")
                    global_dof = dofmap.local_to_global_index(int(dofs[0]))
                    if not first <= global_dof < last:
                        raise ValueError("MOR tensor DG0 cell has a non-owned material dof")
                    entries[index] = local_values[global_dof - first]
                values = np.asarray([entries[index] for index in components]).reshape(3, 3)
                try:
                    check_tensor(values)
                except ValueError as error:
                    raise ValueError("cell %d: %s" % (cell.index(), error)) from error
    except Exception as error:
        local_error = "rank %d: %s" % (comm.rank, error)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise ValueError("MOR conductivity validation failed: " +
                         "; ".join(error for error in errors if error is not None))


def _assemble_mor_form(integrand, mesh):
    """Accumulate owned-cell contributions with a fixed summation order."""
    from petsc4py import PETSc

    comm = mesh.mpi_comm()
    if comm.size == 1:
        return df.assemble(integrand * df.dx(domain=mesh))
    markers = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    owned = np.array([not df.Cell(mesh, i).is_ghost()
                      for i in range(mesh.num_cells())], dtype=bool)
    if comm.allreduce(int(owned.sum()), op=MPI.SUM) != mesh.num_entities_global(mesh.topology().dim()):
        raise ValueError('MOR assembly owned cells do not cover the global mesh exactly once')
    dx = df.Measure('dx', domain=mesh, subdomain_data=markers)
    form = df.Form(integrand * dx(1))
    matrix = df.PETScMatrix(comm)
    coverage = np.zeros(len(owned), dtype=np.int8)
    for source in range(comm.size):
        markers.array()[:] = 0
        if comm.rank == source:
            markers.array()[owned] = 1
            coverage[owned] += 1
        error = None
        try:
            df.assemble(form, tensor=matrix, add_values=source > 0, finalize_tensor=False)
            backend = df.as_backend_type(matrix).mat()
        except Exception as caught:
            error = 'rank %d: %s' % (comm.rank, caught)
        errors = comm.allgather(error)
        if any(message is not None for message in errors):
            raise RuntimeError('MOR phased assembly failed: ' + str(errors))
        mode = PETSc.Mat.AssemblyType.FINAL if source+1 == comm.size else PETSc.Mat.AssemblyType.FLUSH
        backend.assemblyBegin(mode)
        backend.assemblyEnd(mode)
    valid = bool(np.all(coverage[owned] == 1) and np.all(coverage[~owned] == 0))
    if not comm.allreduce(valid, op=MPI.LAND):
        raise RuntimeError('MOR assembly must include each owned cell once and no ghost cells')
    return matrix


def _assemble_mass(V, sigma):
    return _assemble_mor_form(df.inner(sigma * df.TrialFunction(V), df.TestFunction(V)), V.mesh())


def _p1_edge_constants(V, S, G):
    """Represent Cartesian constant fields by the gradient of CG1 coordinates."""
    comm, error, coordinates = V.mesh().mpi_comm(), None, None
    try:
        lo, hi = S.dofmap().ownership_range()
        if V.mesh().geometry().degree() != 1:
            raise ValueError('P1 edge constants require affine coordinate geometry')
        if (V.ufl_element().degree() != 1 or S.ufl_element().degree() != 1 or
                S.ufl_element().family() != 'Lagrange' or V.mesh().geometry().dim() != 3):
            raise ValueError('P1 edge constants require scalar CG1 coordinates')
        if (G.getSize() != (V.dim(), S.dim()) or G.getOwnershipRange() != V.dofmap().ownership_range()
                or G.getOwnershipRangeColumn() != (lo, hi)):
            raise ValueError('P1 gradient and scalar/edge ownership layout differ')
        points = np.asarray(S.tabulate_dof_coordinates()).reshape(-1, 3)
        ids = np.asarray(S.dofmap().tabulate_local_to_global_dofs())
        if len(points) != len(ids):
            if len(points) != hi-lo or not np.array_equal(ids[:hi-lo], np.arange(lo, hi)):
                raise ValueError('CG1 coordinate and local-to-global layout differ')
            ids = ids[:hi-lo]
        selected = (ids >= lo) & (ids < hi)
        rows = ids[selected]-lo
        if not np.array_equal(np.sort(rows), np.arange(hi-lo)):
            raise ValueError('CG1 coordinate ownership is incomplete or duplicated')
        coordinates = np.empty((hi-lo, 3))
        coordinates[rows] = points[selected]
        if not np.isfinite(coordinates).all():
            raise ValueError('CG1 owned coordinates are nonfinite')
    except Exception as caught:
        error = 'rank %d: %s' % (comm.rank, caught)
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError('P1 edge coordinate validation failed: ' + str(errors))
    nodal, functions = G.createVecRight(), [df.Function(V) for unused in range(3)]
    try:
        for axis, function in enumerate(functions):
            nodal.getArray()[:] = coordinates[:, axis]
            G.mult(nodal, df.as_backend_type(function.vector()).vec())
            function.vector().update_ghost_values()
        return functions
    finally:
        nodal.destroy()


def _validate_cells(mesh, mu_inv):
    """Validate cell volumes and material averages with scalar DG0 temporaries."""
    comm = mesh.mpi_comm()
    Q = df.FunctionSpace(mesh, "DG", 0)
    z = df.TestFunction(Q)
    dx = df.Measure("dx", domain=mesh, metadata={"quadrature_degree": 0})
    volumes = df.assemble(z * dx).get_local()
    values = df.assemble(mu_inv * z * dx).get_local()
    local_error = None
    try:
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ArithmeticError("Invalid MOR cell volume")
        values /= volumes
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ArithmeticError("Invalid MOR inverse permeability")
    except Exception as error:
        local_error = "rank %d: %s: %s" % (comm.rank, type(error).__name__, error)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise ArithmeticError("MOR cell validation failed: " +
                              "; ".join(error for error in errors if error is not None))


def _receiver_markers(mesh, cell_markers, receiver_domain):
    """Validate optional material selection collectively before any assembly."""
    comm = mesh.mpi_comm()
    valid = (receiver_domain is None or
             (isinstance(receiver_domain, (int, np.integer)) and
              not isinstance(receiver_domain, (bool, np.bool_)) and
              0 <= receiver_domain <= np.iinfo(np.uint64).max))
    requested = (None if receiver_domain is None else int(receiver_domain)) if valid else 'invalid'
    domains = comm.allgather(requested)
    if 'invalid' in domains or any(domain != domains[0] for domain in domains):
        raise ValueError("MOR receiver domain must be one nonnegative integer, identical on all ranks")
    if receiver_domain is None:
        return None
    values, local_error = None, None
    try:
        if cell_markers is None:
            raise ValueError("receiver domain requires FS.DOM.domain_func cell markers")
        if cell_markers.dim() != mesh.topology().dim() or cell_markers.mesh().id() != mesh.id():
            raise ValueError("receiver domain markers must belong to this mesh's cells")
        values = cell_markers.array()
        if values.ndim != 1 or len(values) != mesh.num_cells() or values.dtype.kind not in 'iu':
            raise ValueError("receiver domain markers must contain one integer per local cell")
    except Exception as error:
        local_error = "rank %d: %s" % (comm.rank, error)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise ValueError("MOR receiver domain marker validation failed: " +
                         "; ".join(error for error in errors if error is not None))
    if not comm.allreduce(bool(np.any(values == receiver_domain)), op=MPI.LOR):
        raise ValueError("MOR receiver domain marker %d is absent from the mesh" % receiver_domain)
    return values


def _receiver_cells(mesh, V, rx, mu_inv, cell_markers=None, receiver_domain=None):
    """Cache owner cells and local curl coefficients at receiver points."""
    comm = mesh.mpi_comm()
    markers = _receiver_markers(mesh, cell_markers, receiver_domain)
    mesh.init_global(3)
    global_cells = mesh.topology().global_indices(3)
    local = np.full(len(rx), -1, dtype=np.int64)
    chosen = np.full(len(rx), np.iinfo(np.int64).max, dtype=np.int64)
    tree = mesh.bounding_box_tree()
    for j, point in enumerate(rx):
        for cell_index in tree.compute_entity_collisions(df.Point(*point)):
            if markers is not None and markers[cell_index] != receiver_domain:
                continue
            cell = df.Cell(mesh, int(cell_index))
            if not cell.is_ghost() and int(global_cells[cell_index]) < chosen[j]:
                local[j] = cell_index
                chosen[j] = global_cells[cell_index]
    minimum = np.empty_like(chosen)
    comm.Allreduce(chosen, minimum, op=MPI.MIN)
    if np.any(minimum == np.iinfo(np.int64).max):
        if receiver_domain is not None:
            raise ValueError("MOR receiver has no containing cell in domain %d" % receiver_domain)
        raise ValueError("MOR receiver lies outside the computational mesh")
    owner = np.where(chosen == minimum, comm.rank, comm.size).astype(np.int32)
    owners = np.empty_like(owner)
    comm.Allreduce(owner, owners, op=MPI.MIN)
    rows, local_error = [], None
    element = V.element()
    for j, point in enumerate(rx):
        if owners[j] != comm.rank:
            continue
        try:
            cell = df.Cell(mesh, int(local[j]))
            derivative = element.evaluate_basis_derivatives_all(
                1, point, cell.get_vertex_coordinates(), cell.orientation()).reshape(
                    element.space_dimension(), 3, 3)
            curl = np.column_stack((derivative[:, 2, 1] - derivative[:, 1, 2],
                                    derivative[:, 0, 2] - derivative[:, 2, 0],
                                    derivative[:, 1, 0] - derivative[:, 0, 1]))
            mu = float(mu_inv(cell.midpoint()))
            if not np.isfinite(curl).all() or not np.isfinite(mu) or mu <= 0:
                raise ArithmeticError("Invalid MOR receiver curl or inverse permeability")
            rows.append((j, cell, curl, mu))
        except Exception as error:
            local_error = "rank %d receiver %d: %s: %s" % (
                comm.rank, j, type(error).__name__, error)
            break
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise RuntimeError("MOR receiver setup failed: " +
                           "; ".join(error for error in errors if error is not None))
    return rows


def _source_path(fe):
    if fe.s_type == "path":
        path = np.asarray(fe.tx[0], dtype=float)
        grounded = fe.grounding
        if isinstance(grounded, (list, tuple, np.ndarray)):
            grounded = grounded[0]
        if not grounded and not np.array_equal(path[0], path[-1]):
            path = np.vstack((path, path[0]))
    elif fe.s_type == "line":
        path = np.asarray([fe.start, fe.stop], dtype=float)
    elif fe.s_type == "hed":
        direction = np.array([np.cos(np.deg2rad(fe.azimuth)),
                              np.sin(np.deg2rad(fe.azimuth)), 0.0])
        origin = np.asarray(fe.origin, dtype=float)
        path = np.array([origin - fe.length * direction / 2,
                         origin + fe.length * direction / 2])
    elif fe.s_type == "loop_r":
        a, b = np.asarray(fe.start, dtype=float), np.asarray(fe.stop, dtype=float)
        path = np.array([a, [b[0], a[1], a[2]], [b[0], b[1], a[2]],
                         [a[0], b[1], a[2]], a])
    else:
        raise ValueError("MOR supports one mesh-fitted line or path source")
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("Invalid MOR source path")
    if not np.all(np.isfinite(path)):
        raise ValueError("MOR source coordinates must be finite")
    return path


def _source_vector(V, constants, path, current, tolerance):
    """Negative line functional q, matching custEM's E_t source convention."""
    mesh, dm = V.mesh(), V.dofmap()
    comm = mesh.mpi_comm()
    mesh.init(1, 0)
    start, stop = dm.ownership_range()
    edge_dofs = dm.entity_dofs(mesh, 1)
    vertex_map = mesh.topology()(1, 0)
    owned, endpoints = [], []
    for edge in df.edges(mesh):
        global_dof = dm.local_to_global_index(int(edge_dofs[edge.index()]))
        if start <= global_dof < stop:
            owned.append(global_dof - start)
            endpoints.append(mesh.coordinates()[vertex_map(edge.index())])
    owned = np.asarray(owned, dtype=int)
    endpoints = np.asarray(endpoints, dtype=float).reshape(-1, 2, 3)
    edge_vectors = endpoints[:, 1] - endpoints[:, 0]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    constant_coefficients = np.column_stack([v.getArray(readonly=True)
                                             for v in constants])
    source = constants[0].duplicate()
    source.set(0.0)
    values = source.getArray()
    for a, b in zip(path[:-1], path[1:]):
        delta = b - a
        length = np.linalg.norm(delta)
        if length <= tolerance:
            raise ValueError("MOR source contains a zero-length segment")
        tangent = delta / length
        distances = np.linalg.norm(endpoints - a, axis=2) + \
            np.linalg.norm(endpoints - b, axis=2)
        selected = np.all(np.abs(distances - length) <= tolerance, axis=1)
        covered = comm.allreduce(float(np.sum(edge_lengths[selected])), op=MPI.SUM)
        if abs(covered - length) > max(10 * tolerance, length * 1e-10):
            raise ValueError("MOR source must follow complete mesh edges: "
                             "covered %.16g of %.16g" % (covered, length))
        rows = owned[selected]
        # Interpolation of a constant tangent gives its oriented edge moment.
        signs = constant_coefficients[rows].dot(tangent) / edge_lengths[selected]
        if comm.allreduce(bool(np.any(np.abs(np.abs(signs) - 1.0) > 1e-9)),
                          op=MPI.LOR):
            raise ArithmeticError("Inconsistent Nedelec edge orientation")
        # Native E_t uses (K+i*omega*M) E = -i*omega*J.  Our equation
        # writes its right-hand side as i*omega*q, hence q = -J.
        np.add.at(values, rows, -current * signs)
    source.assemble()
    return source


def _source_vector_p2(V, path, current, tolerance):
    """Integrate the negative line functional with every local P2 basis."""
    from petsc4py import PETSc

    mesh, dm, element = V.mesh(), V.dofmap(), V.element()
    comm = mesh.mpi_comm()
    mesh.init(1, 0)
    mesh.init(1, 3)
    mesh.init_global(1)
    mesh.init_global(3)
    global_edges = mesh.topology().global_indices(1)
    global_cells = mesh.topology().global_indices(3)
    source = df.as_backend_type(df.Function(V).vector()).vec().duplicate()
    source.set(0.)
    quadrature = (.5 - .5 / np.sqrt(3.), .5 + .5 / np.sqrt(3.))

    def check_error(error, phase):
        errors = comm.allgather(error)
        if any(error is not None for error in errors):
            raise RuntimeError("MOR P2 source %s failed: " % phase +
                               "; ".join(error for error in errors if error is not None))

    try:
        for segment, (a, b) in enumerate(zip(path[:-1], path[1:])):
            delta = b - a
            length = np.linalg.norm(delta)
            if length <= tolerance:
                raise ValueError("MOR source contains a zero-length segment")
            tangent = delta / length
            candidates = {}
            local_error = None
            try:
                for edge in df.edges(mesh):
                    endpoints = mesh.coordinates()[edge.entities(0)]
                    distances = np.linalg.norm(endpoints - a, axis=1) + \
                        np.linalg.norm(endpoints - b, axis=1)
                    if not np.all(np.abs(distances - length) <= tolerance):
                        continue
                    cells = [int(index) for index in edge.entities(3)
                             if not df.Cell(mesh, int(index)).is_ghost()]
                    if cells:
                        cell_index = min(cells, key=lambda index: int(global_cells[index]))
                        key = (segment, int(global_edges[edge.index()]))
                        candidates[key] = (int(global_cells[cell_index]), cell_index,
                                           edge.index())
            except Exception as error:
                local_error = "rank %d: %s: %s" % (comm.rank, type(error).__name__, error)
            check_error(local_error, "candidate scan")
            # Only source-edge metadata crosses ranks. A shared edge is integrated
            # by its smallest global non-ghost incident cell, once per segment.
            metadata = comm.allgather({key: (info[0], comm.rank)
                                       for key, info in candidates.items()})
            owners = {}
            for records in metadata:
                for key, owner in records.items():
                    owners[key] = min(owner, owners.get(key, owner))
            selected = [info for key, info in candidates.items()
                        if owners[key] == (info[0], comm.rank)]
            covered, local_error = 0., None
            try:
                for _, _, edge_index in selected:
                    endpoints = mesh.coordinates()[df.Edge(mesh, edge_index).entities(0)]
                    covered += np.linalg.norm(endpoints[1] - endpoints[0])
            except Exception as error:
                local_error = "rank %d: %s: %s" % (comm.rank, type(error).__name__, error)
            check_error(local_error, "coverage check")
            covered = comm.allreduce(float(covered), op=MPI.SUM)
            if abs(covered - length) > max(10 * tolerance, length * 1e-10):
                raise ValueError("MOR source must follow complete mesh edges: "
                                 "covered %.16g of %.16g" % (covered, length))
            local_error = None
            try:
                for _, cell_index, edge_index in selected:
                    cell = df.Cell(mesh, cell_index)
                    endpoints = mesh.coordinates()[df.Edge(mesh, edge_index).entities(0)]
                    if np.dot(endpoints[1] - endpoints[0], tangent) < 0:
                        endpoints = endpoints[::-1]
                    direction = endpoints[1] - endpoints[0]
                    values = np.zeros(element.space_dimension())
                    for point in quadrature:
                        basis = element.evaluate_basis_all(
                            endpoints[0] + point * direction,
                            cell.get_vertex_coordinates(), cell.orientation()).reshape(-1, 3)
                        values -= .5 * current * basis.dot(direction)
                    dofs = np.asarray([dm.local_to_global_index(int(dof))
                                       for dof in dm.cell_dofs(cell_index)], dtype=PETSc.IntType)
                    source.setValues(dofs, values, addv=PETSc.InsertMode.ADD_VALUES)
            except Exception as error:
                local_error = "rank %d: %s: %s" % (comm.rank, type(error).__name__, error)
            check_error(local_error, "assembly")
        source.assemble()
        return source
    except BaseException:
        source.destroy()
        raise


def assemble_mor(model, *, receiver_domain=None):
    """Assemble operators; optionally sample receivers from one material only.

    receiver_domain selects FS.DOM.domain_func cell markers. The default keeps
    the original minimum-global-cell policy across all materials.
    """
    started = perf_counter()
    fs, fe, mp = model.FS, model.FE, model.MP
    mesh, V, S = fs.mesh, fs.V, fs.S
    comm = mesh.mpi_comm()
    element = V.ufl_element()
    degree = element.degree()
    if (mp.approach != "E_t" or degree not in (1, 2) or
            element.family() != "Nedelec 1st kind H(curl)" or
            mesh.topology().dim() != 3 or mesh.ufl_cell().cellname() != "tetrahedron"):
        raise ValueError("MOR requires P1 or P2 first-kind tetrahedral E_t")
    if not fe.quasi_static or fe.ip or fe.n_tx != 1 or fe.bc is not None:
        raise ValueError("MOR requires quasi-static, no IP, one source and bc=None")
    if getattr(mp, "file_format", "h5") != "h5" and not getattr(fs, "test_mode", False):
        raise ValueError("MOR requires a parallel HDF5 mesh")
    frequencies = np.asarray(mp.frequencies, dtype=float)
    if frequencies.ndim != 1 or not frequencies.size or \
            not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("MOR frequencies must be finite and positive")
    currents = np.asarray(mp.currents)
    if np.iscomplexobj(currents) and np.any(currents.imag != 0):
        raise ValueError("MOR requires a fixed real current")
    currents = np.asarray(currents.real, dtype=float).ravel()
    if not currents.size or not np.all(np.isfinite(currents)) or \
            np.any(currents != currents[0]):
        raise ValueError("MOR requires one frequency-independent source current")
    _validate_conductivity(mp.sigma_func, mesh)
    _positive_scalar(mp.mu_inv_func, comm, "inverse permeability")

    assembly_seconds = dict(mass_lower_bound=0.0, curl=0.0)
    assembly_counts = dict(mass_lower_bound=0, curl=0)
    receivers = getattr(mp, "rx", [])
    rx = np.vstack(receivers).reshape(-1, 3) if len(receivers) else np.empty((0, 3))
    if not np.all(np.isfinite(rx)):
        raise ValueError("MOR receiver coordinates must be finite")
    phase_started = perf_counter()
    markers = getattr(getattr(fs, "DOM", None), "domain_func", None)
    receiver_cells = _receiver_cells(mesh, V, rx, mp.mu_inv_func, markers, receiver_domain)
    global_receiver_cells = np.full(len(rx), np.iinfo(np.int64).max, dtype=np.int64)
    for index, cell, _, _ in receiver_cells:
        global_receiver_cells[index] = cell.global_index()
    comm.Allreduce(MPI.IN_PLACE, global_receiver_cells, op=MPI.MIN)
    assembly_seconds["receivers"] = perf_counter() - phase_started
    phase_started = perf_counter()
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = df.Measure("dx", domain=mesh)
    kform = df.inner(mp.mu_inv_func * df.curl(u), df.curl(v)) * dx
    mform = df.inner(mp.sigma_func * u, v) * dx
    if degree == 1:
        kd = _assemble_mor_form(df.inner(mp.mu_inv_func * df.curl(u), df.curl(v)), mesh)
        md = _assemble_mass(V, mp.sigma_func)
    else:
        kd, md = df.assemble(kform), df.assemble(mform)
    auxiliary, q = None, None
    try:
        if degree == 1:
            gd = df.DiscreteOperators.build_gradient(V, S)
            gradient = df.as_backend_type(gd).mat()
            constant_functions = _p1_edge_constants(V, S, gradient)
            constants = [df.as_backend_type(f.vector()).vec() for f in constant_functions]
            owners = [kd, md, gd, constant_functions]
        else:
            from .mor_high_order import build_p2_auxiliary
            auxiliary = build_p2_auxiliary(V, S)
            gradient, constants, owners = auxiliary["G"], [], [kd, md]
        assembly_seconds["operators"] = perf_counter() - phase_started
        phase_started = perf_counter()
        path, tolerance = _source_path(fe), float(getattr(fe, "a_tol", 1e-8))
        q = (_source_vector(V, constants, path, currents[0], tolerance) if degree == 1 else
             _source_vector_p2(V, path, currents[0], tolerance))
        assembly_seconds["source"] = perf_counter() - phase_started

        phase_started = perf_counter()
        _validate_cells(mesh, mp.mu_inv_func)
        assembly_seconds["cell_validation"] = perf_counter() - phase_started
    except BaseException:
        if auxiliary is not None:
            auxiliary["destroy"]()
            if q is not None:
                q.destroy()
        raise

    def compute_curl():
        """Build the stable curl factor once for chain or correction modes."""
        if degree == 2:
            raise NotImplementedError("P2 MOR does not support chain or residual correction")
        if ops["curl"] is not None:
            return ops["curl"]
        curl_started = perf_counter()
        assembly_counts["curl"] += 1
        try:
            Q = df.VectorFunctionSpace(mesh, "DG", 0)
            z = df.TestFunction(Q)
            cd = df.assemble(df.inner(df.sqrt(mp.mu_inv_func / df.CellVolume(mesh)) *
                                     df.curl(u), z) * dx)
            # P1 curl is cellwise constant, so C.T*C = K without cancellation.
            ops["Q"], ops["curl"] = Q, df.as_backend_type(cd).mat()
            ops["_owners"].append(cd)
            return ops["curl"]
        finally:
            assembly_seconds["curl"] += perf_counter() - curl_started

    def compute_mass_lower_bound():
        """Compute the full-mass bound only for inverse-defect acceptance."""
        if degree == 2:
            raise NotImplementedError("P2 MOR does not support residual correction")
        if ops["mass_lower_bound"] is not None:
            return ops["mass_lower_bound"]
        bound_started = perf_counter()
        assembly_counts["mass_lower_bound"] += 1
        lower, local_error = 1.0, None
        try:
            local_mass_form = df.Form(mform)
            for cell in df.cells(mesh):
                if cell.is_ghost():
                    continue
                local_mass = df.assemble_local(local_mass_form, cell)
                diagonal = np.diag(local_mass)
                if (not np.isfinite(local_mass).all() or
                        not np.all(diagonal > 0)):
                    raise ArithmeticError("Invalid local MOR mass matrix")
                normalized = local_mass / np.sqrt(np.outer(diagonal, diagonal))
                c = 0.99 * float(np.linalg.eigvalsh(normalized)[0])
                if not np.isfinite(c) or c <= 0:
                    raise ArithmeticError("Non-positive local MOR mass matrix")
                np.linalg.cholesky(local_mass - c * np.diag(diagonal))
                lower = min(lower, c)
        except Exception as error:
            local_error = "rank %d: %s: %s" % (
                comm.rank, type(error).__name__, error)
        errors = comm.allgather(local_error)
        if any(error is not None for error in errors):
            assembly_seconds["mass_lower_bound"] += perf_counter() - bound_started
            raise ArithmeticError("MOR local mass validation failed: " +
                                  "; ".join(error for error in errors if error is not None))
        ops["mass_lower_bound"] = comm.allreduce(lower, op=MPI.MIN)
        assembly_seconds["mass_lower_bound"] += perf_counter() - bound_started
        return ops["mass_lower_bound"]

    ops = dict(K=df.as_backend_type(kd).mat(), M=df.as_backend_type(md).mat(),
               q=q, G=gradient, constants=constants,
               curl=None, mass_lower_bound=None,
               assembly_seconds=assembly_seconds, assembly_counts=assembly_counts,
               compute_mass_lower_bound=compute_mass_lower_bound, compute_curl=compute_curl,
               mesh=mesh, V=V, S=S, Q=None, rx=rx, model=model,
               receiver_domain_marker=None if receiver_domain is None else int(receiver_domain),
               receiver_cell_global_indices=global_receiver_cells.tolist(),
               receiver_cells=receiver_cells, er=df.Function(V), ei=df.Function(V),
               _owners=owners)

    if auxiliary is None:
        ops["destroy"] = ops.clear
    else:
        ops.update(fe_degree=2, ams_interpolation=auxiliary["Pi"],
                   auxiliary_setup=dict(setup_seconds=auxiliary["setup_seconds"],
                                        nnz=auxiliary["nnz"]))

        def destroy():
            if ops:
                auxiliary["destroy"]()
                q.destroy()
                ops.clear()

        ops["destroy"] = destroy
    assembly_seconds["total"] = perf_counter() - started
    return ops


def _copy_field(function, vector):
    vector.copy(df.as_backend_type(function.vector()).vec())
    function.vector().apply("insert")
    function.vector().update_ghost_values()


def store_field(model, real, imag):
    """Assign separate real/imag vectors using DOLFIN's actual subspace maps."""
    fs = model.FS
    real_function, imag_function = df.Function(fs.V), df.Function(fs.V)
    _copy_field(real_function, real)
    _copy_field(imag_function, imag)
    if not isinstance(fs.U, list):
        fs.U = [df.Function(fs.M)]
    df.FunctionAssigner(fs.M, [fs.V, fs.V]).assign(fs.U[0],
                                                 [real_function, imag_function])


def sample_receivers(ops, real, imag, omega):
    """Sample original edge E and pointwise H; communicate receiver arrays."""
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("Receiver omega must be finite and positive")
    mesh, rx = ops["mesh"], ops["rx"]
    comm = mesh.mpi_comm()
    _copy_field(ops["er"], real)
    _copy_field(ops["ei"], imag)
    values = np.zeros((len(rx), 4, 3), dtype=float)
    local_error = None
    element = ops["V"].element()
    for j, cell, curl, mu in ops["receiver_cells"]:
        try:
            for k, name in enumerate(("er", "ei")):
                ops[name].eval_cell(values[j, k], rx[j], cell)
            values[j, 2] = -(mu / omega) * (ops["ei"].restrict(element, cell) @ curl)
            values[j, 3] = (mu / omega) * (ops["er"].restrict(element, cell) @ curl)
        except Exception as error:
            local_error = "rank %d receiver %d: %s: %s" % (
                comm.rank, j, type(error).__name__, error)
            break
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        raise RuntimeError("MOR receiver evaluation failed: " +
                           "; ".join(error for error in errors if error is not None))
    result = np.empty_like(values)
    comm.Allreduce(values, result, op=MPI.SUM)
    return dict(E=result[:, 0] + 1j * result[:, 1],
                H=result[:, 2] + 1j * result[:, 3], coordinates=rx.copy())
