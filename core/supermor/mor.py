"""Distributed MOR for the quasi-static total electric field.

Only real PETSc matrices are required. Complex vectors are local owned numpy
arrays; scalar products reduce scalars or small matrices, never full fields.
"""
from datetime import datetime, timezone
import json
from numbers import Real
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.linalg.blas import daxpy
from scipy.linalg.lapack import dgeqrf, dormqr
from scipy.sparse import csr_matrix


class MORFailure(RuntimeError):
    """A numerical acceptance check failed without changing the algorithm."""


class _ConvergedBeforeExtension(Exception):
    """An existing partial basis converged before a rejected extension."""


def _cpc_trace(comm, event, **details):
    """Optional rank-local progress only; no collective or numerical side effects."""
    directory = os.environ.get('CUSTEM_CPC_TRACE_DIR')
    if not directory:
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   monotonic_s=perf_counter(), pid=os.getpid(), rank=comm.rank,
                   mpi_size=comm.size, event=event, **details)
        with (path / ('rank_%04d.jsonl' % comm.rank)).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
    except (OSError, TypeError, ValueError) as error:
        print('CPC_TRACE_WRITE_FAILED: %s' % error, file=sys.stderr, flush=True)


def _frequencies(values):
    if np.iscomplexobj(values):
        raise ValueError('MOR frequencies must be real')
    f = np.asarray(values, dtype=float)
    if f.ndim != 1 or not f.size or not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError('MOR requires a nonempty finite positive frequency list')
    return f



def _validate_ams_interpolation(matrix, gradient, interpolation):
    """Check a full high-order interpolation collectively, before AMS setup."""
    comm = matrix.comm.tompi4py()
    error = None
    try:
        if gradient is None:
            raise ValueError('P2 interpolation requires a discrete gradient')
        if interpolation is None:
            raise ValueError('P2 requires a full AMS interpolation matrix')
        for auxiliary in (gradient, interpolation):
            if MPI.Comm.Compare(comm, auxiliary.comm.tompi4py()) not in (MPI.IDENT, MPI.CONGRUENT):
                raise ValueError('P2 interpolation and gradient must use the operator communicator')
        rows, columns = gradient.getSize()
        if (rows != matrix.getSize()[0] or
                interpolation.getSize() != (rows, 3 * columns) or
                gradient.getOwnershipRange() != matrix.getOwnershipRange() or
                interpolation.getOwnershipRange() != matrix.getOwnershipRange()):
            raise ValueError('P2 interpolation must have matching rows and three columns per scalar DOF')
        if 3 * columns > np.iinfo(PETSc.IntType).max:
            raise ValueError('P2 interpolation exceeds the PETSc integer index range')
        lo, hi = gradient.getOwnershipRangeColumn()
        if interpolation.getOwnershipRangeColumn() != (3 * lo, 3 * hi):
            raise ValueError('P2 interpolation columns must use node-major scalar ownership')
        if not interpolation.getType().endswith('aij'):
            raise ValueError('P2 interpolation requires an explicit real AIJ matrix')
        values = interpolation.getValuesCSR()[2]
        if np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError('P2 interpolation must contain finite real entries')
        if not PETSc.Sys.hasExternalPackage('hypre') or np.issubdtype(PETSc.ScalarType, np.complexfloating):
            raise ValueError('P2 production requires a real PETSc build with HYPRE')
    except Exception as caught:
        error = 'rank %d: %s' % (comm.rank, caught)
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError('; '.join(message for message in errors if message is not None))


def _validate_ops(ops, need_gradient=False):
    """Check explicit matrix inputs before the zero-source shortcut."""
    K, M, q = ops['K'], ops['M'], ops['q']
    comm = K.comm.tompi4py()
    n = q.getSize()
    valid = (n > 0 and K.getSize() == (n, n) and M.getSize() == (n, n) and
             K.getOwnershipRange() == q.getOwnershipRange() and
             M.getOwnershipRange() == q.getOwnershipRange())
    if not comm.allreduce(valid, op=MPI.LAND):
        raise ValueError('K, M and q must have matching square sizes and ownership')
    if not comm.allreduce(bool(np.isfinite(q.getArray(readonly=True)).all()), op=MPI.LAND):
        raise ValueError('The source must contain finite real entries')
    degrees = comm.allgather(ops.get('fe_degree', 1))
    if (any(isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
            or degree not in (1, 2) for degree in degrees) or
            any(degree != degrees[0] for degree in degrees)):
        raise ValueError('MOR fe_degree must be P1 or P2, identical on every rank')
    if degrees[0] == 2:
        _validate_ams_interpolation(K, ops.get('G'), ops.get('ams_interpolation'))
    matrices = [K, M]
    if need_gradient:
        G = ops['G']
        if G.getSize()[0] != n or G.getSize()[1] < 1:
            raise ValueError('The discrete gradient must have one row per edge')
        matrices.append(G)
    if ops.get('curl') is not None:
        if ops['curl'].getSize()[1] != n:
            raise ValueError('The curl factor must have one column per edge')
        right = ops['curl'].createVecRight()
        try:
            matching = right.getOwnershipRange() == q.getOwnershipRange()
        finally:
            right.destroy()
        if not comm.allreduce(matching, op=MPI.LAND):
            raise ValueError('Curl columns must have the source vector ownership')
        matrices.append(ops['curl'])
    for matrix in matrices:
        kind = matrix.getType()
        if kind.endswith('aij'):
            values = matrix.getValuesCSR()[2]
        elif kind.endswith('dense'):
            values = matrix.getDenseArray()
        else:
            raise ValueError('MOR requires explicit real AIJ or dense matrices')
        valid = not np.iscomplexobj(values) and bool(np.isfinite(values).all())
        if not comm.allreduce(valid, op=MPI.LAND):
            raise ValueError('MOR matrices must contain finite real entries')
    diagonal = M.createVecLeft()
    try:
        M.getDiagonal(diagonal)
        values = diagonal.getArray(readonly=True)
        valid = bool(np.isfinite(values).all() and np.all(values > 0))
    finally:
        diagonal.destroy()
    if not comm.allreduce(valid, op=MPI.LAND):
        raise ValueError('The mass matrix must have a finite positive diagonal')


def _sum(comm, value):
    if np.ndim(value):
        result = np.empty_like(value)
        comm.Allreduce(np.ascontiguousarray(value), result, op=MPI.SUM)
        return result
    return comm.allreduce(value, op=MPI.SUM)


def _dense(comm, operation):
    """Root-only small linear algebra, with collective error propagation."""
    value = None
    if comm.rank == 0:
        try:
            value = (operation(), None)
        except Exception as error:
            value = (None, '%s: %s' % (type(error).__name__, error))
    result, error = comm.bcast(value, root=0)
    if error:
        raise MORFailure('Reduced solve failed: ' + error)
    return result


class _CurlQR:
    """Row-panel QR of C V, exchanging only locally needed ghost entries.

    C's local rows are split once into owned and ghost column CSR blocks.
    Each call returns a square local R, including zero modes, without forming
    the complete local C V or gathering a global basis.
    """

    def __init__(self, matrix, stats, block_rows=32768):
        started = perf_counter()
        self.stats = stats
        # Every rank reduces the same timing keys, including ranks with no rows.
        for key in ('curl_qr_setup', 'curl_action', 'local_qr'):
            stats['seconds'].setdefault(key, 0.0)
        self.block_rows = min(32768, int(block_rows))
        if self.block_rows <= 0:
            raise ValueError('Curl QR panel size must be positive')
        self.x = matrix.createVecRight()
        source, target, local_error = None, None, None
        try:
            first, last = self.x.getOwnershipRange()
            self.local_columns = last - first
            rows = matrix.getOwnershipRange()
            shape = (rows[1] - rows[0], matrix.getSize()[1])
            if matrix.getType().endswith('dense'):
                local = csr_matrix(matrix.getDenseArray())
                rowptr, columns, values = local.indptr, local.indices, local.data
            else:
                rowptr, columns, values = matrix.getValuesCSR()
            owned = (columns >= first) & (columns < last)
            prefix = np.empty(len(columns) + 1, dtype=PETSc.IntType)
            prefix[0] = 0
            np.cumsum(owned, dtype=PETSc.IntType, out=prefix[1:])
            owned_ptr = prefix[rowptr]
            ghost_ptr = rowptr - owned_ptr
            ghosts, ghost_indices = np.unique(columns[~owned], return_inverse=True)
            self.owned = csr_matrix((values[owned], columns[owned] - first, owned_ptr),
                                    shape=(shape[0], self.local_columns))
            self.ghost = csr_matrix((values[~owned], ghost_indices, ghost_ptr),
                                    shape=(shape[0], len(ghosts)))
            self.ghost_vector = PETSc.Vec().createSeq(len(ghosts), comm=PETSc.COMM_SELF)
            source = PETSc.IS().createGeneral(ghosts, comm=PETSc.COMM_SELF)
            target = PETSc.IS().createStride(len(ghosts), first=0, step=1,
                                           comm=PETSc.COMM_SELF)
        except Exception as error:
            local_error = '%s: %s' % (type(error).__name__, error)
        errors = matrix.comm.tompi4py().allgather(local_error)
        try:
            if any(error is not None for error in errors):
                raise MORFailure('Curl QR setup failed: ' + '; '.join(error for error in errors if error is not None))
            self.scatter = PETSc.Scatter().create(self.x, source,
                                                  self.ghost_vector, target)
        except Exception:
            self.destroy()
            raise
        finally:
            if source is not None:
                source.destroy()
            if target is not None:
                target.destroy()
        self._add_time('curl_qr_setup', perf_counter() - started)

    def _add_time(self, name, elapsed):
        seconds = self.stats['seconds']
        seconds[name] = seconds.get(name, 0.0) + elapsed

    def __call__(self, basis):
        if basis.ndim != 2 or basis.shape[0] != self.local_columns:
            raise ValueError('Curl QR basis must contain the locally owned rows')
        columns = basis.shape[1]
        local_rows = self.owned.shape[0]
        started = perf_counter()
        ghost_basis = np.empty((self.ghost.shape[1], columns), order='F')
        for column in range(columns):
            self.x.getArray()[:] = basis[:, column]
            self.scatter.scatter(self.x, self.ghost_vector,
                                 addv=PETSc.InsertMode.INSERT_VALUES,
                                 mode=PETSc.ScatterMode.FORWARD)
            ghost_basis[:, column] = self.ghost_vector.getArray(readonly=True)
        self._add_time('curl_action', perf_counter() - started)
        result = np.zeros((columns, columns), order='F')
        if not columns or not local_rows:
            return result
        panel_rows = min(self.block_rows, local_rows)
        work = np.zeros((columns + panel_rows, columns), order='F')
        # Query once, then pass the complete F-contiguous array on every fold.
        _, _, optimal, info = dgeqrf(work, lwork=-1, overwrite_a=True)
        if info:
            raise RuntimeError('Curl QR workspace query failed: %d' % info)
        lwork = int(optimal[0])
        for first in range(0, local_rows, panel_rows):
            last = min(first + panel_rows, local_rows)
            started = perf_counter()
            work[:columns] = result
            work[columns:] = 0.0
            panel = work[columns:columns + last - first]
            owned, ghost = self.owned[first:last], self.ghost[first:last]
            # CSR multivector multiplication ravels F-order V into a complete
            # C-order copy. Column actions keep both V and the panel in place.
            for column in range(columns):
                panel[:, column] = owned @ basis[:, column]
                if self.ghost.shape[1]:
                    panel[:, column] += ghost @ ghost_basis[:, column]
            self._add_time('curl_action', perf_counter() - started)
            started = perf_counter()
            qr, _, _, info = dgeqrf(work, lwork=lwork, overwrite_a=True)
            if info:
                raise RuntimeError('Curl QR factorization failed: %d' % info)
            if not np.shares_memory(qr, work):
                raise RuntimeError('Curl QR did not reuse its column-contiguous buffer')
            result[:] = np.triu(qr[:columns])
            self._add_time('local_qr', perf_counter() - started)
        return result

    def destroy(self):
        for name in ('scatter', 'ghost_vector', 'x'):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.destroy()


class _Action:
    """Reusable real PETSc input/output vectors for local array actions."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.x, self.y = matrix.createVecRight(), matrix.createVecLeft()

    def __call__(self, values, transpose=False):
        source, target = (self.y, self.x) if transpose else (self.x, self.y)
        action = self.matrix.multTranspose if transpose else self.matrix.mult
        source.getArray()[:] = np.real(values)
        action(source, target)
        result = target.getArray(readonly=True).copy()
        if np.iscomplexobj(values):
            source.getArray()[:] = np.imag(values)
            action(source, target)
            result = result + 1j * target.getArray(readonly=True)
        return result

    def destroy(self):
        self.x.destroy()
        self.y.destroy()


def _event_snapshot(events, stage):
    """Keep raw count/time evidence; a logging failure never changes a solve."""
    result = {}
    for label, event in events.items():
        try:
            info = event.getPerfInfo(stage=stage)
            value = float(info['time'])
            result[label] = dict(count=int(info['count']),
                                 time=value if np.isfinite(value) else repr(value))
        except Exception as error:
            result[label] = dict(count=None, time=None, error=str(error))
    return result


def _select_poisson_gauge(ops, matrix, diagonal):
    """Use a strong physical P1 gauge, retaining explicit-matrix/P2 behavior."""
    comm = matrix.comm.tompi4py()
    lo, hi = matrix.getOwnershipRange()
    if ops.get('fe_degree') == 2:
        local = int(np.argmax(diagonal)) if len(diagonal) else 0
        candidate = ((float(diagonal[local]), lo + local) if len(diagonal)
                     else (-np.inf, 0))
        value, pin = max(comm.allgather(candidate), key=lambda item: (item[0], -item[1]))
        return pin, dict(poisson_pinned_dof=int(pin), poisson_pinned_diagonal=float(value))
    spaces = comm.allgather(ops.get('S') is not None)
    if not any(spaces):
        return 0, {}
    error, owned_coordinates = None, None
    try:
        if not all(spaces):
            raise ValueError('Scalar space must be present on every rank')
        space = ops['S']
        dimension = space.mesh().geometry().dim()
        coordinates = np.asarray(space.tabulate_dof_coordinates()).reshape(-1, dimension)
        dofmap = space.dofmap()
        if tuple(dofmap.ownership_range()) != (lo, hi):
            raise ValueError('Scalar coordinates and Poisson matrix have different ownership')
        global_ids = np.asarray(dofmap.tabulate_local_to_global_dofs())
        owned_coordinates = np.empty((hi-lo, dimension))
        if len(coordinates) == hi-lo:
            owned_ids = global_ids[:hi-lo]
            if not np.array_equal(owned_ids, np.arange(lo, hi)):
                raise ValueError('Owned scalar DOFs must precede ghosts in global order')
            owned_coordinates[owned_ids-lo] = coordinates
        elif len(coordinates) == len(global_ids):
            owned = (global_ids >= lo) & (global_ids < hi)
            if not np.array_equal(np.sort(global_ids[owned]), np.arange(lo, hi)):
                raise ValueError('Scalar coordinate map must cover every owned DOF once')
            owned_coordinates[global_ids[owned]-lo] = coordinates[owned]
        else:
            raise ValueError('Scalar coordinate rows must match owned or ghost DOFs')
        if owned_coordinates.shape != (hi-lo, 3) or not np.isfinite(owned_coordinates).all():
            raise ValueError('Expected three finite scalar coordinate components')
    except Exception as caught:
        error = str(caught)
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise ValueError('Poisson gauge coordinates: ' + '; '.join(
            message for message in errors if message is not None))
    maximum = comm.allreduce(float(np.max(diagonal)) if len(diagonal) else -np.inf, op=MPI.MAX)
    candidates = np.flatnonzero(diagonal == maximum)
    local = min((tuple(float(value) for value in owned_coordinates[i]) + (int(lo+i),)
                 for i in candidates), default=(np.inf, np.inf, np.inf, -1))
    chosen = min(comm.allgather(local))
    pin = int(chosen[-1])
    return pin, dict(poisson_pinned_dof=pin, poisson_pinned_diagonal=float(maximum),
                     poisson_pinned_coordinates=list(chosen[:3]),
                     poisson_pinned_policy='max_diagonal_physical_coordinates')


class _LinearSolve:
    """Persistent solves; finite true-residual gaps may be diagnostic for a PC."""
    def __init__(self, matrix, stats, name, *, direct=False, gradient=None,
                 constants=None, pinned=False, ams_interpolation=None, pinned_dof=0,
                 require_true_residual=True):
        high_order = ams_interpolation is not None
        if high_order:
            _validate_ams_interpolation(matrix, gradient, ams_interpolation)
        prefix = 'custem_mor_%s_' % name
        if high_order and not direct:
            # Rank-local PC changes can block inside setFromOptions itself.
            # Compare before creating/binding any AMS state, not just after it.
            supplied = {key[len(prefix):]: value for key, value in PETSc.Options().getAll().items()
                        if key.startswith(prefix)}
            by_rank = matrix.comm.tompi4py().allgather(supplied)
            if any(values != by_rank[0] for values in by_rank):
                raise ValueError('P2 production requires identical CG/AMS options on every rank')
            required = dict(ksp_type='cg', pc_type='hypre', pc_hypre_type='ams',
                            ksp_norm_type='unpreconditioned', pc_hypre_ams_cycle_type='1')
            if any(key in supplied and str(supplied[key]).lower() != value
                   for key, value in required.items()):
                raise ValueError('P2 production requires CG with full-interpolation AMS cycle 1')
        if not direct and gradient is not None and not high_order and (constants is None or len(constants) != 3):
            raise ValueError('Production AMS requires three edge constant vectors')
        self.stats, self.name, self.matrix = stats, name, matrix
        self.direct, self.pinned = direct, pinned
        self.require_true_residual = bool(require_true_residual)
        self.pinned_dof = pinned_dof
        self.x, self.b = matrix.createVecRight(), matrix.createVecLeft()
        self.ksp = PETSc.KSP().create(matrix.comm)
        self.ksp.setOperators(matrix)
        pc = self.ksp.getPC()
        self.ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        keys = []

        def option(key, value):
            options[prefix + key] = value
            keys.append(prefix + key)

        if direct:
            self.ksp.setType('preonly')
            pc.setType('lu')
            pc.setFactorSolverType('mumps')
        else:
            self.ksp.setType('cg')
            self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            pc.setType('hypre')
            pc.setHYPREType('ams' if gradient is not None else 'boomeramg')
            if gradient is not None:
                pc.setHYPREDiscreteGradient(gradient)
                if high_order:
                    try:
                        # ND_Pi_Full avoids the broken ND_Pi array binding in
                        # petsc4py 3.20.5; do not pass the three components.
                        pc.setHYPRESetInterpolations(3, ND_Pi_Full=ams_interpolation)
                    except Exception:
                        self.destroy()
                        raise
                    option('pc_hypre_ams_cycle_type', 1)
                else:
                    pc.setHYPRESetEdgeConstantVectors(*constants)
                    option('pc_hypre_ams_cycle_type', 13)
                option('pc_hypre_ams_max_iter', 1)
                option('pc_hypre_ams_tol', 0.0)
                option('pc_hypre_ams_relax_type', 2)
                option('pc_hypre_ams_amg_alpha_options', '10,1,8,6,4')
                option('pc_hypre_ams_amg_beta_options', '10,1,8,6,4')
                option('pc_hypre_ams_amg_alpha_theta', 0.25)
                option('pc_hypre_ams_amg_beta_theta', 0.25)
            else:
                option('pc_hypre_boomeramg_max_iter', 1)
                option('pc_hypre_boomeramg_tol', 0.0)
                option('pc_hypre_boomeramg_relax_type_all', 'symmetric-SOR/Jacobi')
        events, stage, before, after = {}, None, {}, {}
        if direct:
            PETSc.Log.begin()  # PETSc guarantees repeated calls are idempotent.
            stage = PETSc.Log.Stage('custem_mor_' + name + '_setup')
            stage.push()
            events = {label: PETSc.Log.Event(event_name) for label, event_name in
                      (('analysis', 'MatLUFactorSym'), ('factor', 'MatLUFactorNum'))}
            before = _event_snapshot(events, int(stage))
        started = perf_counter()
        trace_cpc = (bool(os.environ.get('CUSTEM_CPC_TRACE_DIR')) and direct and
                     stats.get('route') == 'cpc' and name in ('P', 'M'))
        if trace_cpc:
            selected_options = {key: value for key, value in options.getAll().items()
                                if key.startswith(prefix) or 'mumps' in key.lower()}
            _cpc_trace(matrix.comm.tompi4py(), 'setup_begin', matrix=name,
                       petsc_options=selected_options, TMPDIR=os.environ.get('TMPDIR'),
                       MUMPS_OOC_TMPDIR=os.environ.get('MUMPS_OOC_TMPDIR'))
        try:
            if high_order and not direct:
                error = None
                try:
                    self.ksp.setFromOptions()
                except Exception as caught:
                    error = str(caught)
                errors = matrix.comm.tompi4py().allgather(error)
                if any(message is not None for message in errors):
                    raise ValueError('P2 CG/AMS options failed: ' + '; '.join(
                        message for message in errors if message is not None))
                valid = not (self.ksp.getType() != 'cg' or pc.getType() != 'hypre' or
                        pc.getHYPREType() != 'ams' or
                        self.ksp.getNormType() != PETSc.KSP.NormType.UNPRECONDITIONED or
                        options.getInt(prefix + 'pc_hypre_ams_cycle_type', 1) != 1)
                if not matrix.comm.tompi4py().allreduce(valid, op=MPI.LAND):
                    raise ValueError('P2 production requires CG with full-interpolation AMS cycle 1')
            else:
                self.ksp.setFromOptions()
            if trace_cpc:
                _cpc_trace(matrix.comm.tompi4py(), 'backend_setup_begin', matrix=name)
            self.ksp.setUp()
            setup_duration = perf_counter() - started
        except Exception as error:
            if trace_cpc:
                _cpc_trace(matrix.comm.tompi4py(), 'setup_failure', matrix=name,
                           error_type=type(error).__name__, error=str(error))
            self.destroy()
            raise
        finally:
            for key in keys:
                del options[key]
            if stage is not None:
                after = _event_snapshot(events, int(stage))
                stage.pop()
        if trace_cpc:
            try:
                disk_mode = dict(status='available', value=int(pc.getFactorMatrix().getMumpsIcntl(22)))
            except Exception as error:
                disk_mode = dict(status='unavailable', error=str(error))
            _cpc_trace(matrix.comm.tompi4py(), 'setup_end', matrix=name,
                       setup_s=setup_duration, petsc_events=after, mumps_icntl_22=disk_mode)
        key = name + '_setup'
        stats['seconds'][key] = stats['seconds'].get(key, 0.0) + setup_duration
        if direct:
            comm = matrix.comm.tompi4py()
            raw = dict(rank=comm.rank, stage=int(stage), before=before, after=after)
            by_rank = comm.allgather(raw)
            valid = True
            for label in events:
                deltas = []
                for row in by_rank:
                    a, b = row['before'][label], row['after'][label]
                    ok = (a['count'] is not None and b['count'] is not None and
                          b['count'] - a['count'] == 1 and
                          isinstance(a['time'], (int, float)) and
                          isinstance(b['time'], (int, float)) and
                          np.isfinite([a['time'], b['time']]).all() and
                          b['time'] >= a['time'])
                    if ok:
                        deltas.append(b['time'] - a['time'])
                    else:
                        valid = False
                # Invalid event times remain absent: neither setup time nor an
                # absolute value is a substitute for numeric factorization time.
                if len(deltas) == comm.size:
                    key = name + '_' + label
                    stats['seconds'][key] = stats['seconds'].get(key, 0.0) + float(max(deltas))
            timing = stats.setdefault('factorization_timing', {})
            previous = timing.get(name)
            record = dict(valid=bool(valid), stage=int(stage), events_by_rank=by_rank)
            try:
                factor = pc.getFactorMatrix()
                record['mumps_controls'] = dict(status='available',
                    icntl_10=int(factor.getMumpsIcntl(10)),
                    icntl_28=int(factor.getMumpsIcntl(28)),
                    cntl_1=float(factor.getMumpsCntl(1)))
            except Exception as error:
                record['mumps_controls'] = dict(status='not_available', error=str(error))
            records = [] if previous is None else previous['records']
            records.append(record)
            timing[name] = dict(record, records=records,
                                valid=bool(valid and (previous is None or previous['valid'])))
            if not timing[name]['valid']:
                for label in events:
                    stats['seconds'].pop(name + '_' + label, None)
            stats['factorization_timing_valid'] = bool(
                stats.get('factorization_timing_valid', True) and valid)
        key = name + ('_factorizations' if direct else '_setups')
        stats['counts'][key] = stats['counts'].get(key, 0) + 1

    def __call__(self, rhs, rtol=1e-6, max_it=300):
        real = self._real(np.real(rhs), rtol, max_it)
        if np.iscomplexobj(rhs):
            return real + 1j * self._real(np.imag(rhs), rtol, max_it)
        return real

    def _real(self, rhs, rtol, max_it):
        self.b.getArray()[:] = rhs
        if self.pinned:
            lo, hi = self.b.getOwnershipRange()
            if lo <= self.pinned_dof < hi:
                self.b.getArray()[self.pinned_dof-lo] = 0.0
        self.x.set(0.0)
        if not self.direct:
            self.ksp.setTolerances(rtol=rtol, atol=0.0, max_it=int(max_it))
        started = perf_counter()
        self.ksp.solve(self.b, self.x)
        duration = perf_counter() - started
        key = self.name + '_solve'
        self.stats['seconds'][key] = self.stats['seconds'].get(key, 0.0) + duration
        key = self.name + '_rhs'
        self.stats['counts'][key] = self.stats['counts'].get(key, 0) + 1
        self.last_iterations = int(self.ksp.getIterationNumber())
        key = self.name + '_iterations'
        self.stats['counts'][key] = self.stats['counts'].get(key, 0) + self.last_iterations
        solve_record = None
        if 'linear_solves' in self.stats:
            solve_record = dict(name=self.name, direct=self.direct,
                iterations=self.last_iterations, seconds=duration,
                rtol=float(rtol), max_it=int(max_it),
                require_true_residual=self.require_true_residual,
                true_residual_exceeds_rtol=None,
                reason=int(self.ksp.getConvergedReason()),
                basis_columns=self.stats['max_basis'],
                cycle=self.stats.get('current_cycle', 0),
                pole=self.stats.get('current_pole'))
            if self.direct:
                solve_record['rtol_applied'] = False
            self.stats['linear_solves'].append(solve_record)
        if self.ksp.getConvergedReason() <= 0:
            raise MORFailure('%s solve failed (PETSc reason %d, iterations %d)' %
                             (self.name, self.ksp.getConvergedReason(), self.last_iterations))
        if not self.direct or solve_record is not None:
            audit = self.b.duplicate()
            try:
                self.matrix.mult(self.x, audit)
                audit.aypx(-1.0, self.b)
                denominator = self.b.norm()
                rho = audit.norm() / denominator if denominator else audit.norm()
                if self.name == 'poisson' and not self.direct:
                    if solve_record is not None:
                        solve_record.update(initial_relative_residual=float(rho),
                                            recovery_count=0, residual_corrections=[])
                    if np.isfinite(rho) and rho > rtol:
                        # Correct the fresh residual without replacing the original
                        # right-hand side, its norm, or the accumulated solution.
                        delta = self.x.duplicate()
                        best_x, best_rho = self.x.copy(), rho
                        tolerances = self.ksp.getTolerances()
                        nonzero_guess = self.ksp.getInitialGuessNonzero()
                        correction_failed = False
                        try:
                            self.ksp.setInitialGuessNonzero(False)
                            for _ in range(2):
                                remaining = int(max_it) - self.last_iterations
                                if remaining <= 0:
                                    break
                                previous_rho = rho
                                if self.pinned:
                                    values = audit.getArray()
                                    if lo <= self.pinned_dof < hi:
                                        values[self.pinned_dof-lo] = 0.0
                                    del values
                                delta.set(0.0)
                                self.ksp.setTolerances(rtol=0.1, atol=0.0,
                                                       max_it=remaining)
                                started = perf_counter()
                                self.ksp.solve(audit, delta)
                                elapsed = perf_counter() - started
                                iterations = int(self.ksp.getIterationNumber())
                                reason = int(self.ksp.getConvergedReason())
                                self.last_iterations += iterations
                                key = self.name + '_iterations'
                                self.stats['counts'][key] += iterations
                                key = self.name + '_solve'
                                self.stats['seconds'][key] += elapsed
                                key = self.name + '_recoveries'
                                self.stats['counts'][key] = self.stats['counts'].get(key, 0) + 1
                                correction_failed = reason <= 0
                                if not correction_failed:
                                    self.x.axpy(1.0, delta)
                                    if self.pinned and lo <= self.pinned_dof < hi:
                                        self.x.getArray()[self.pinned_dof-lo] = 0.0
                                    self.matrix.mult(self.x, audit)
                                    audit.aypx(-1.0, self.b)
                                    rho = audit.norm() / denominator if denominator else audit.norm()
                                    if np.isfinite(rho) and rho < best_rho:
                                        self.x.copy(best_x)
                                        best_rho = rho
                                if solve_record is not None:
                                    solve_record['iterations'] = self.last_iterations
                                    solve_record['seconds'] += elapsed
                                    solve_record['recovery_count'] += 1
                                    solve_record['residual_corrections'].append(dict(
                                        iterations=iterations, seconds=elapsed,
                                        reason=reason, relative_residual=float(rho)))
                                if (correction_failed or not np.isfinite(rho) or
                                        rho >= previous_rho or rho <= rtol):
                                    break
                            best_x.copy(self.x)
                            self.matrix.mult(self.x, audit)
                            audit.aypx(-1.0, self.b)
                            rho = audit.norm() / denominator if denominator else audit.norm()
                        finally:
                            self.ksp.setTolerances(rtol=tolerances[0], atol=tolerances[1],
                                                   divtol=tolerances[2], max_it=tolerances[3])
                            self.ksp.setInitialGuessNonzero(nonzero_guess)
                            delta.destroy()
                            best_x.destroy()
                        if correction_failed:
                            if solve_record is not None:
                                solve_record['relative_residual'] = float(rho)
                            raise MORFailure('%s correction solve failed (PETSc reason %d, iterations %d)' %
                                             (self.name, reason, self.last_iterations))
            finally:
                audit.destroy()
            key = self.name + '_maximum_true_residual'
            self.stats[key] = max(self.stats.get(key, 0.0), float(rho))
            if solve_record is not None:
                solve_record['relative_residual'] = float(rho)
                solve_record['true_residual_exceeds_rtol'] = bool(rho > rtol)
            if not self.direct and (not np.isfinite(rho) or
                                    (self.require_true_residual and rho > rtol)):
                raise MORFailure('%s failed true residual: %.3e > %.3e' %
                                 (self.name, rho, rtol))
            if not self.direct and not self.require_true_residual and rho > rtol:
                key = self.name + '_diagnostic_true_residual_mismatches'
                self.stats['counts'][key] = self.stats['counts'].get(key, 0) + 1
        result = self.x.getArray(readonly=True).copy()
        if not self.matrix.comm.tompi4py().allreduce(bool(np.isfinite(result).all()), op=MPI.LAND):
            raise MORFailure(self.name + ' solve returned nonfinite values')
        return result

    def destroy(self):
        self.ksp.destroy()
        self.x.destroy()
        self.b.destroy()


def _stats(route, f, xi):
    return dict(route=route, frequencies=f.tolist(), xi0=float(xi), pole=float(xi),
                status='running', failure=None, residual_kind='assembled_matrix', counts={}, seconds={},
                per_frequency=[], directions=[], max_basis=0, gram_error=0.0)


def _norm(comm, values):
    quadratic = float(_sum(comm, np.vdot(values, values).real))
    if not np.isfinite(quadratic):
        raise MORFailure('Nonfinite global Euclidean norm')
    return float(np.sqrt(max(0.0, quadratic)))


def _mass_norm(comm, M, values):
    return _mass_norm_product(comm, values, M(values))


def _mass_norm_product(comm, values, mass_values):
    quadratic = float(_sum(comm, np.vdot(values, mass_values).real))
    if not np.isfinite(quadratic):
        raise MORFailure('Nonfinite global Hermitian mass norm')
    if quadratic < -1e-13 * max(1.0, _norm(comm, values)**2):
        raise MORFailure('Negative Hermitian mass norm')
    return np.sqrt(max(0.0, quadratic))


def _extend_gram(comm, previous, columns, candidate, mass_candidate):
    """Append a column of V.T M V, using the symmetric mass operator."""
    n = columns.shape[1]
    result = np.empty((n + 1, n + 1))
    result[:n, :n] = previous
    result[:n, n] = _sum(comm, columns.T @ mass_candidate)
    result[n, :n] = result[:n, n]
    result[n, n] = _sum(comm, np.dot(candidate, mass_candidate))
    return result


def _residual(comm, K, M, q, field, omega, normq, stats=None):
    """True residual of the assembled equation; the third result is an alias."""
    started = perf_counter()
    residual = 1j * omega * q - K(field) - 1j * omega * M(field)
    denominator = abs(omega) * normq
    rho = _norm(comm, residual) / denominator if denominator else _norm(comm, residual)
    if not np.isfinite(rho):
        raise MORFailure('Nonfinite original-equation matrix residual')
    if stats is not None:
        stats['counts']['true_residual_checks'] = stats['counts'].get('true_residual_checks', 0) + 1
        stats['seconds']['residual_validation'] = (stats['seconds'].get('residual_validation', 0.0)
                                                   + perf_counter() - started)
    return residual, float(rho), float(rho)


def _emit(ops, index, field, emit):
    if emit is not None:
        re, im = ops['M'].createVecRight(), ops['M'].createVecRight()
        try:
            re.getArray()[:] = np.real(field)
            im.getArray()[:] = np.imag(field)
            emit(index, re, im)
        finally:
            re.destroy()
            im.destroy()


def _source_basis(comm, M, shifted, seed, steps, stats):
    """Real M-Arnoldi relation Z V = [V, tail] Hbar, with two MGS passes."""
    started = perf_counter()
    stats['counts']['source_basis_builds'] = stats['counts'].get('source_basis_builds', 0) + 1
    capacity = min(int(steps), comm.allreduce(len(seed), op=MPI.SUM))
    eta = _mass_norm(comm, M, seed)
    if not eta:
        raise MORFailure('Nonzero source seed has zero M-norm')
    vectors = np.empty((len(seed), capacity + 1))
    mass_vectors = np.empty_like(vectors)
    Hbar = np.zeros((capacity + 1, capacity))
    vectors[:, 0] = seed / eta
    mass_vectors[:, 0] = M(vectors[:, 0])
    beta, breakdown, updates = 0.0, None, 0
    trace_cpc = bool(os.environ.get('CUSTEM_CPC_TRACE_DIR'))
    for column in range(capacity):
        if trace_cpc:
            column_started = perf_counter()
            _cpc_trace(comm, 'arnoldi_column_begin', cycle=stats['counts']['source_basis_builds'],
                       column=column + 1, projected_column_budget=capacity)
        remainder = shifted(mass_vectors[:, column])
        image_norm = _mass_norm(comm, M, remainder)
        mass_remainder = M(remainder)
        for repeat in range(2):
            if repeat:
                mass_remainder = M(remainder)
            for row in range(column + 1):
                coefficient = _sum(comm, np.dot(vectors[:, row], mass_remainder))
                Hbar[row, column] += coefficient
                remainder -= coefficient * vectors[:, row]
                mass_remainder -= coefficient * mass_vectors[:, row]
        updates += 1
        stats['counts']['source_arnoldi_updates'] = stats['counts'].get('source_arnoldi_updates', 0) + 1
        if not comm.allreduce(bool(np.isfinite(remainder).all()), op=MPI.LAND):
            raise MORFailure('Source Arnoldi produced a nonfinite direction')
        beta = _mass_norm(comm, M, remainder)
        if beta <= 64 * np.finfo(float).eps * image_norm:
            beta, breakdown = 0.0, 'invariant'
            if trace_cpc:
                _cpc_trace(comm, 'arnoldi_column_end', cycle=stats['counts']['source_basis_builds'],
                           column=column + 1, elapsed_s=perf_counter() - column_started,
                           beta=float(beta), breakdown=breakdown)
            break
        vectors[:, column + 1] = remainder / beta
        mass_vectors[:, column + 1] = M(vectors[:, column + 1])
        Hbar[column + 1, column] = beta
        if trace_cpc:
            _cpc_trace(comm, 'arnoldi_column_end', cycle=stats['counts']['source_basis_builds'],
                       column=column + 1, elapsed_s=perf_counter() - column_started,
                       beta=float(beta), breakdown=breakdown)
    V = vectors[:, :updates]
    tail = None if breakdown else vectors[:, updates]
    extended = updates + (tail is not None)
    gram = _sum(comm, vectors[:, :extended].T @ mass_vectors[:, :extended])
    gram_error = float(np.linalg.norm(gram - np.eye(extended), 2))
    stats['gram_error'] = max(stats['gram_error'], gram_error)
    stats['max_basis'] = max(stats['max_basis'], updates)
    stats['counts']['source_basis_columns'] = stats['counts'].get('source_basis_columns', 0) + updates
    stats['seconds']['basis'] = stats['seconds'].get('basis', 0.0) + perf_counter() - started
    if not np.isfinite(gram_error) or gram_error > 1e-10:
        raise MORFailure('Source extended M-Gram error %.3e exceeds 1e-10' % gram_error)
    return V, Hbar[:updates + 1, :updates], eta, tail, beta, updates, gram_error, breakdown


def solve_cpc(ops, frequencies, emit=None, steps=80, max_cycles=10, tol=1e-6):
    """Direct shared-source CPC forward, matching solve_restarted_source.

    Each cycle has at most ``steps`` real projected columns and one tail.
    Previous corrections retain real basis blocks and small coefficients;
    complete frequency fields are reconstructed individually, never stored
    as a full-space-by-frequency array. M and P each receive one MUMPS LU.
    """
    f = _frequencies(frequencies)
    _validate_ops(ops)
    if (isinstance(steps, (bool, np.bool_)) or isinstance(max_cycles, (bool, np.bool_))
            or int(steps) != steps or not 1 <= steps <= 80 or int(max_cycles) != max_cycles
            or not 1 <= max_cycles <= 10):
        raise ValueError('CPC requires 1..80 projected columns and 1..10 cycles')
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError('tol must be finite and between zero and one')
    xi = -2 * np.pi * (np.sqrt(f.min()) * np.sqrt(f.max()))
    if not np.isfinite(xi) or xi >= 0:
        raise ValueError('CPC pole is outside the finite negative range')
    stats = _stats('cpc', f, xi)
    stats['assembly_seconds'] = ops.get('assembly_seconds', {})
    stats['assembly_counts'] = ops.get('assembly_counts', {})
    stats.update(algorithm='shared_source_real_arnoldi', trace=[],
                 projected_column_budget=int(steps), cycle_budget=int(max_cycles),
                 retained_basis_columns=0)
    comm = ops['K'].comm.tompi4py()
    q = ops['q'].getArray(readonly=True).copy()
    normq = _norm(comm, q)
    if not normq and comm.allreduce(bool(np.any(q)), op=MPI.LOR):
        raise ValueError('Nonzero source norm is outside the representable range')
    shifts = 2j * np.pi * f
    if normq and (not np.isfinite(abs(shifts) * normq).all() or np.any(abs(shifts) * normq == 0)):
        raise ValueError('Nonzero frequency RHS norm is outside the representable range')
    K, M = _Action(ops['K']), _Action(ops['M'])
    resources = [K, M]
    P, blocks = None, []
    converged = np.zeros(len(f), dtype=bool)
    p = np.ones(len(f), dtype=complex)
    frequency_cycles = np.zeros(len(f), dtype=int)
    relative = np.ones(len(f)) if normq else np.zeros(len(f))
    physical_relative = relative.copy()
    pending = False
    started = perf_counter()

    def reconstruct(index):
        field = np.zeros_like(q, dtype=complex)
        for vectors, restart_tail, coefficients, tail_coefficients in blocks:
            field += vectors @ coefficients[:, index]
            if restart_tail is not None:
                field += restart_tail * tail_coefficients[index]
        return field

    try:
        if not normq:
            converged[:] = True
            stats['status'] = 'converged'
        else:
            P = ops['K'].copy()
            P.axpy(-xi, ops['M'], structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            P.assemble()
            shifted = _LinearSolve(P, stats, 'P', direct=True)
            resources.append(shifted)
            mass_solve = _LinearSolve(ops['M'], stats, 'M', direct=True)
            resources.append(mass_solve)
            seed = mass_solve(q)
            for cycle in range(1, int(max_cycles) + 1):
                _cpc_trace(comm, 'cycle_begin', cycle=cycle, projected_column_budget=int(steps),
                           retained_basis_columns=stats['retained_basis_columns'])
                active = ~converged
                previous_relative, previous_physical = relative.copy(), physical_relative.copy()
                previous_converged = converged.copy()
                V, Hbar, eta, tail, beta, updates, gram, breakdown = _source_basis(
                    comm, M, shifted, seed, steps, stats)
                H = Hbar[:updates, :]
                rhs = np.zeros(updates)
                rhs[0] = eta
                projection_started = perf_counter()

                def reduced():
                    Y = np.zeros((updates, len(f)), dtype=complex)
                    for index in np.flatnonzero(active):
                        Y[:, index] = np.linalg.solve(np.eye(updates) + (xi + shifts[index]) * H, rhs)
                    return Y

                Y = _dense(comm, reduced)
                d = beta * Y[-1]
                coefficients = (H @ Y) * (shifts * p)[None, :]
                tail_coefficients = d * shifts * p
                p_next = p.copy()
                p_next[active] = -p[active] * (xi + shifts[active]) * d[active]
                if not all(np.isfinite(value).all() for value in
                           (Y, coefficients, tail_coefficients, p_next)):
                    raise MORFailure('Shared source coefficients are nonfinite')
                blocks.append((V, tail, coefficients, tail_coefficients))
                pending = True
                stats['retained_basis_columns'] += updates + (tail is not None)
                stats['seconds']['projection'] = stats['seconds'].get('projection', 0.0) + perf_counter() - projection_started
                stats['counts']['source_reduced_solves'] = stats['counts'].get('source_reduced_solves', 0) + int(active.sum())
                frequency_cycles[active] += 1
                residual_gaps = [None] * len(f)
                mass_tail = M(tail) if tail is not None else None
                for index, frequency in enumerate(f):
                    field = reconstruct(index)
                    residual, relative[index], physical_relative[index] = _residual(
                        comm, K, M, q, field, 2 * np.pi * frequency, normq, stats=stats)
                    if active[index]:
                        theoretical = (shifts[index] * p_next[index] * mass_tail
                                       if tail is not None else np.zeros_like(q))
                        residual_gaps[index] = float(_norm(comm, residual - theoretical) /
                                                     (abs(shifts[index]) * normq))
                    converged[index] = max(relative[index], physical_relative[index]) <= tol
                stats['trace'].append(dict(cycle=cycle, basis_columns=updates, updates=updates,
                    gram_error=gram, basis_breakdown=breakdown,
                    active_frequency_count=int(active.sum()), relative_residuals=relative.tolist(),
                    physical_relative_residuals=physical_relative.tolist(), residual_gaps=residual_gaps,
                    max_relative_residual=float(max(relative.max(), physical_relative.max())),
                    converged_frequencies=converged.tolist(), remaining_frequency_count=int((~converged).sum()),
                    p_real=p_next.real.tolist(), p_imag=p_next.imag.tolist()))
                _cpc_trace(comm, 'cycle_residual', **stats['trace'][-1])
                pending = False
                if converged.all():
                    stats['status'] = 'converged'
                    break
                if tail is None:
                    raise MORFailure('Source has no restart tail before true-residual convergence')
                if np.any(p_next[~converged] == 0.):
                    raise MORFailure('Source residual coefficient underflowed before convergence')
                seed, p = tail.copy(), p_next
            if stats['status'] != 'converged':
                raise MORFailure('CPC exhausted %d shared source cycles' % max_cycles)
    except Exception as error:
        if pending:
            blocks.pop()
            stats['retained_basis_columns'] -= updates + (tail is not None)
            frequency_cycles[active] -= 1
            relative, physical_relative, converged = previous_relative, previous_physical, previous_converged
        stats.update(status='failed', failure='%s: %s' % (type(error).__name__, error))
    finally:
        try:
            for index, frequency in enumerate(f):
                field = reconstruct(index)
                if comm.allreduce(bool(np.isfinite(field).all()), op=MPI.LAND):
                    _emit(ops, index, field, emit)
            stats['per_frequency'] = [dict(index=index, frequency=float(frequency),
                cycles=int(frequency_cycles[index]), residual=float(relative[index]),
                physical_residual=float(physical_relative[index]), basis_columns=stats['max_basis'])
                for index, frequency in enumerate(f)]
        except Exception as error:
            stats.update(status='failed', failure='%s: %s' % (type(error).__name__, error))
        finally:
            blocks.clear()
            for resource in reversed(resources):
                resource.destroy()
            if P is not None:
                P.destroy()
            stats['seconds']['total'] = perf_counter() - started
    for key, duration in stats['seconds'].items():
        stats['seconds'][key] = comm.allreduce(duration, op=MPI.MAX)
    _finalize_stats(stats)
    return stats


def _split_field(g, basis, coefficients, omega, field, work):
    """Reconstruct into fixed vectors without promoting the whole real basis."""
    if basis.shape[1]:
        np.dot(basis, coefficients.imag, out=work)
        np.multiply(work, -omega, out=field.real)
        np.add(field.real, g, out=field.real)
        np.dot(basis, coefficients.real, out=work)
        np.multiply(work, omega, out=field.imag)
    else:
        field.real[:] = g
        field.imag.fill(0.0)
    return field


def _real_residual_direction(comm, residual):
    """Unit real combination with greatest energy in Re/Im of one residual."""
    finite = bool(np.isfinite(residual).all())
    if not comm.allreduce(finite, op=MPI.LAND):
        raise MORFailure('Nonfinite residual for real direction selection')
    local_scale = max(float(np.max(np.abs(residual.real), initial=0.0)),
                      float(np.max(np.abs(residual.imag), initial=0.0)))
    scale = comm.allreduce(local_scale, op=MPI.MAX)
    if not scale:
        return np.zeros(residual.size), dict(coefficients=[1.0, 0.0],
                                            eigenvalues=[0.0, 0.0], residual_scale=0.0)
    real, imag = residual.real / scale, residual.imag / scale
    products = _sum(comm, np.array([np.dot(real, real), np.dot(real, imag),
                                   np.dot(imag, imag)]))

    def select():
        gram = np.array([[products[0], products[1]], [products[1], products[2]]])
        values, vectors = np.linalg.eigh(gram)
        coefficients = vectors[:, -1]
        if coefficients[np.argmax(np.abs(coefficients))] < 0:
            coefficients = -coefficients
        return coefficients, values

    coefficients, eigenvalues = _dense(comm, select)
    direction = real
    direction *= coefficients[0]
    if direction.size:
        daxpy(imag, direction, a=coefficients[1])
    direction /= np.sqrt(eigenvalues[-1])
    return direction, dict(coefficients=coefficients.tolist(),
                           eigenvalues=eigenvalues.tolist(), residual_scale=float(scale))


class _ResidualQR:
    """Local append-only Householder QR of [h, Kv1, Mv1, ...].

    New image pairs are copied into the factor and can then be released.
    Each rank retains its own reflectors; only the padded small R is gathered.
    """

    def __init__(self, comm, local_rows, max_basis, stats):
        started = perf_counter()
        self.comm, self.stats = comm, stats
        self.local_rows, self.capacity = int(local_rows), 2 * int(max_basis) + 1
        self.width = 0
        self.factor = self.tau = self.scales = None
        for key in ('residual_qr_setup', 'residual_qr_apply', 'residual_qr_factor'):
            stats['seconds'].setdefault(key, 0.0)
        for key in ('residual_qr_updates', 'residual_qr_columns'):
            stats['counts'].setdefault(key, 0)
        error = None
        try:
            if self.local_rows < 0 or self.capacity < 1:
                raise ValueError('Invalid residual QR dimensions')
            self.factor = np.empty((self.local_rows, self.capacity), order='F')
            self.tau = np.empty(min(self.local_rows, self.capacity))
            self.scales = np.empty(self.capacity)
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        errors = comm.allgather(error)
        if any(errors):
            self.destroy()
            raise MORFailure('Residual QR setup failed: ' + '; '.join(e for e in errors if e))
        stats['seconds']['residual_qr_setup'] += perf_counter() - started

    def __call__(self, KV, MV, h):
        error, columns = None, -1
        try:
            columns = KV.shape[1]
            valid = (self.factor is not None and KV.ndim == 2 and
                     KV.shape[0] == self.local_rows and MV.shape == KV.shape and
                     h.shape == (self.local_rows,) and
                     not any(np.iscomplexobj(a) for a in (KV, MV, h)) and
                     self.width <= 2 * columns + 1 <= self.capacity)
            if not valid:
                raise ValueError('Invalid residual QR prefix or image dimensions')
            first = max(0, (self.width - 1) // 2)
            if not all(np.isfinite(a).all() for a in (KV[:, first:], MV[:, first:])):
                raise ValueError('Residual QR requires finite new image columns')
            if not self.width and not np.isfinite(h).all():
                raise ValueError('Residual QR requires finite h')
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        requests = self.comm.allgather((error, columns, self.width))
        errors = [request[0] for request in requests if request[0]]
        if errors or len({request[1:] for request in requests}) != 1:
            raise MORFailure('Invalid residual QR inputs: ' + '; '.join(errors or ['inconsistent prefixes']))

        if not self.width:
            self.append(h)
        for index in range((self.width - 1) // 2, columns):
            self.append(KV[:, index], MV[:, index])
        return self.factors()

    def append(self, *columns):
        """Commit the source column or one image pair collectively."""
        append_started = perf_counter()
        count = len(columns)
        error = None
        try:
            if (count not in (1, 2) or self.factor is None or
                    self.width + count > self.capacity or
                    any(a.shape != (self.local_rows,) or np.iscomplexobj(a) or
                        not np.isfinite(a).all() for a in columns)):
                raise ValueError('Invalid residual QR image columns')
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        requests = self.comm.allgather((error, self.width, count))
        errors = [request[0] for request in requests if request[0]]
        if errors or len({request[1:] for request in requests}) != 1:
            raise MORFailure('Invalid residual QR append: ' + '; '.join(errors or ['inconsistent prefixes']))
        old = self.width
        block = self.factor[:, old:old + count]
        for index, column in enumerate(columns):
            block[:, index] = column
        scales = np.sqrt(_sum(self.comm, np.einsum('ij,ij->j', block, block)))
        if not np.isfinite(scales).all():
            raise MORFailure('Nonfinite residual QR column norms')
        scales[scales == 0] = 1.0
        block /= scales
        error, new_tau = None, None
        try:
            k = min(self.local_rows, old)
            started = perf_counter()
            if k:
                reflectors = self.factor[:, :k]
                # At most two RHS: minimum workspace avoids rebuilding the
                # old blocked reflector products on every append.
                transformed, unused, info = dormqr(b'L', b'T', reflectors,
                    self.tau[:k], block, lwork=count, overwrite_c=1)
                if info or not np.shares_memory(transformed, block):
                    raise MORFailure('Residual QR reflector application/buffer reuse failed')
            self.stats['seconds']['residual_qr_apply'] += perf_counter() - started
            started = perf_counter()
            if self.local_rows > k:
                # A trailing row slice is not F-contiguous. Copy only the
                # new column block, never the old reflector matrix.
                tail = np.array(block[k:], order='F', copy=True)
                unused, unused_tau, optimal, info = dgeqrf(tail, lwork=-1, overwrite_a=1)
                if info:
                    raise MORFailure('Residual QR tail workspace query failed')
                factor, new_tau, unused, info = dgeqrf(
                    tail, lwork=int(optimal[0]), overwrite_a=1)
                if info or not np.shares_memory(factor, tail):
                    raise MORFailure('Residual QR tail factorization/buffer reuse failed')
                block[k:] = factor
            self.stats['seconds']['residual_qr_factor'] += perf_counter() - started
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        # A failed append leaves old reflectors untouched. All ranks
        # commit these columns together before entering another norm reduce.
        errors = self.comm.allgather(error)
        if any(errors):
            raise MORFailure('Residual QR append failed: ' + '; '.join(e for e in errors if e))
        if new_tau is not None:
            self.tau[k:k + len(new_tau)] = new_tau
        self.scales[old:old + count] = scales
        self.width += count
        self.stats['counts']['residual_qr_updates'] += 1
        self.stats['counts']['residual_qr_columns'] += count
        seconds = self.stats['seconds']
        seconds['residual_joint_QR'] = seconds.get('residual_joint_QR', 0.0) + perf_counter() - append_started

    def factors(self):
        """Extract only the local small R and original column scales."""
        error, local_R = None, None
        try:
            local_R = np.zeros((self.width, self.width), order='F')
            rows = min(self.local_rows, self.width)
            local_R[:rows] = np.triu(self.factor[:rows, :self.width])
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        errors = self.comm.allgather(error)
        if any(errors):
            raise MORFailure('Residual QR factor extraction failed: ' + '; '.join(e for e in errors if e))
        return local_R, self.scales[:self.width].copy()

    def combination(self, coefficients):
        """Recover W y from local Householder factors, including all modes."""
        started = perf_counter()
        result, error = None, None
        try:
            y = np.asarray(coefficients)
            if (self.factor is None or y.shape != (self.width,) or
                    not np.isfinite(y).all()):
                raise ValueError('Invalid residual QR combination coefficients')
            rows = min(self.local_rows, self.width)
            R = np.triu(self.factor[:rows, :self.width])
            reduced = R @ (self.scales[:self.width] * y)
            # Real LAPACK applies Q to the real and imaginary parts together.
            work = np.zeros((self.local_rows, 2), order='F')
            work[:rows, 0], work[:rows, 1] = reduced.real, reduced.imag
            if rows:
                transformed, unused, info = dormqr(b'L', b'N',
                    self.factor[:, :rows], self.tau[:rows], work,
                    lwork=2, overwrite_c=1)
                if info or not np.shares_memory(transformed, work):
                    raise MORFailure('Residual QR reconstruction/buffer reuse failed')
            if np.iscomplexobj(y):
                result = np.empty(self.local_rows, dtype=complex)
                result.real, result.imag = work[:, 0], work[:, 1]
            else:
                result = work[:, 0].copy()
            if not np.isfinite(result).all():
                raise MORFailure('Nonfinite residual QR reconstruction')
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        errors = self.comm.allgather(error)
        if any(errors):
            raise MORFailure('Residual QR combination failed: ' + '; '.join(e for e in errors if e))
        seconds = self.stats['seconds']
        seconds['residual_reconstruction'] = seconds.get('residual_reconstruction', 0.0) + perf_counter() - started
        return result

    def destroy(self):
        self.factor = self.tau = self.scales = None


def _residual_mr(comm, KV, MV, h, frequencies, stats, qr=None):
    """Distributed Euclidean MR via incremental QR and small, full-mode SVD.

    The target is the split-u equation with right-hand side h. Only small R
    factors and reduced coefficients are communicated; no normal equations.
    """
    columns = (qr.width - 1) // 2 if KV is None else KV.shape[1]
    seconds = stats.setdefault('seconds', {})
    qr_seconds = seconds.get('residual_joint_QR', 0.0)
    started = perf_counter()
    temporary = qr is None
    if temporary:
        qr = _ResidualQR(comm, KV.shape[0], columns, stats)
    try:
        local_R, scales = qr.factors() if KV is None else qr(KV, MV, h)
    finally:
        if temporary:
            qr.destroy()
    factors = comm.gather(local_R, root=0)
    # An offline full-image call may append inside __call__; count that time
    # once. Production appends before entering this small-system operation.
    seconds['residual_joint_QR'] = qr_seconds + perf_counter() - started
    started = perf_counter()

    def reduced():
        R = np.linalg.qr(np.vstack(factors), mode='r') * scales
        coefficients = np.empty((columns, len(frequencies)), dtype=complex)
        information = dict(condition_scaled_max=0.0, small_optimality_max=0.0,
                           retained_rank=columns, truncated=False, predicted_residuals=[])
        for index, frequency in enumerate(frequencies):
            s = 2j * np.pi * frequency
            A = R[:, 1::2] + s * R[:, 2::2]
            b = R[:, 0]
            if not columns:
                information['predicted_residuals'].append(float(np.linalg.norm(b)))
                continue
            column_scales = np.linalg.norm(A, axis=0)
            if np.any(column_scales == 0):
                raise MORFailure('Residual trial direction has zero equation image')
            scaled = A / column_scales
            U, singular, VH = np.linalg.svd(scaled, full_matrices=False)
            if not np.isfinite(singular).all() or singular[-1] <= 0:
                raise MORFailure('Singular residual trial system')
            a = (VH.conj().T @ ((U.conj().T @ b) / singular)) / column_scales
            coefficients[:, index] = a
            r = b - A @ a
            information['predicted_residuals'].append(float(np.linalg.norm(r)))
            information['condition_scaled_max'] = max(information['condition_scaled_max'],
                                                       float(singular[0] / singular[-1]))
            optimality = np.linalg.norm(scaled.conj().T @ r) / max(
                singular[0] * np.linalg.norm(b), np.finfo(float).tiny)
            information['small_optimality_max'] = max(information['small_optimality_max'],
                                                       float(optimality))
        return coefficients, information

    result = _dense(comm, reduced)
    seconds['reduced_solve'] = seconds.get('reduced_solve', 0.0) + perf_counter() - started
    return result


def _orthogonal(comm, M, vector, columns, gram):
    mass_remainder = M(vector)
    quadratic = float(_sum(comm, np.dot(vector, mass_remainder)))
    if not np.isfinite(quadratic) or quadratic < 0:
        raise MORFailure('Invalid initial MGS mass norm')
    initial = np.sqrt(quadratic)
    remainder = vector.copy()
    for unused in range(2):
        if unused:
            mass_remainder = M(remainder)
        coefficients = _sum(comm, columns @ mass_remainder)
        for j, v in enumerate(columns):
            coefficient = coefficients[j]
            # Ordered MGS updates in the small Gram space avoid storing M V.
            # Each pass begins with a fresh mass action, as does the norm audit.
            if remainder.size:
                daxpy(v, remainder, a=-coefficient)
            coefficients -= coefficient * gram[:, j]
    # Independently recompute M after both passes; the small Gram recurrence
    # is not used for the rank or normalization decision.
    mass_remainder = M(remainder)
    return remainder, initial, _mass_norm_product(comm, remainder, mass_remainder), mass_remainder


def solve_split_u(ops, frequencies, emit=None, max_basis=400, check_every=10,
                  tol=1e-6, direct=False, residual_correction=False,
                  pole_factor=None, poles=None, pole_every=20,
                  basis_strategy='residual_mr', pole_schedule='hold_first',
                  field_formulation='split_u', inner_rtol=1e-6):
    """Residual-driven split-u MR, using high/low/high-then-hold by default.

    ``direct=True`` is a matrix-reproduction test mode using MUMPS for P and
    the pinned Poisson matrix; production uses PCG/AMS and PCG/BoomerAMG.
    Each accepted direction is orthogonalized against the complete history.
    Explicit ``pole_factor`` retains the legacy geometric single-pole mode;
    explicit ``poles`` supplies one or two negative real shifts. Only the
    current AMS/factor is retained, and Poisson is set up once.
    Each pole block contains ``pole_every`` accepted columns (default 20).
    ``pole_schedule='hold_first'`` uses the first pole, then the second,
    then keeps the first for all remaining columns; ``'alternate'`` keeps
    alternating. A single pole stays fixed in either schedule.
    Candidate defect/eta checks and their corrections are disabled by default;
    ``residual_correction=True`` restores them. Matrix residual and
    orthogonality checks remain active; split-u also applies gradient cleanup.
    ``inner_rtol`` controls only the first P solve for each direction (default
    1e-6). Poisson solves retain 1e-10 and optional defect corrections retain
    1e-2, within the same shared 300-step PCG budget. Direct solves record the
    requested tolerance without applying an iterative acceptance threshold.

    ``field_formulation='total_field'`` retains the same MR engine and sets
    g=0 and h=q, so the basis represents E=s*V*a directly. It skips the
    explicit static Poisson solve and per-column gradient cleanup, while AMS
    still receives the discrete gradient and may perform internal nodal solves.
    This option requires ``basis_strategy='residual_mr'`` and
    ``residual_correction=False``.
    ``static_vector_count`` counts explicitly constructed static g vectors,
    including a zero projection of a nonzero source; the zero-source shortcut
    and total-field formulation construct none.

    The default ``basis_strategy='residual_mr'`` minimizes the dynamic equation
    residual after every accepted column, then selects one real combination
    of the latest worst-frequency u residual. QR residual norms screen all
    frequencies; complete fields and matrix residuals are checked when ready
    to converge and on exit. ``check_every`` only controls the original
    optional ``'chain'`` strategy. Poles and inner tolerances are shared.
    """
    f = _frequencies(frequencies)
    if (not isinstance(inner_rtol, Real) or isinstance(inner_rtol, (bool, np.bool_))
            or not np.isfinite(inner_rtol) or not 0 < inner_rtol < 1):
        raise ValueError('inner_rtol must be a finite real scalar between 0 and 1')
    inner_rtol = float(inner_rtol)
    if (isinstance(max_basis, (bool, np.bool_)) or isinstance(check_every, (bool, np.bool_))
            or isinstance(pole_every, (bool, np.bool_))
            or int(max_basis) != max_basis or max_basis < 1
            or int(check_every) != check_every or check_every < 1
            or int(pole_every) != pole_every or pole_every < 1
            or not np.isfinite(tol) or tol <= 0):
        raise ValueError('Invalid split-u basis, checkpoint or tolerance')
    if not isinstance(residual_correction, (bool, np.bool_)):
        raise ValueError('residual_correction must be a boolean')
    if basis_strategy not in ('chain', 'residual_mr'):
        raise ValueError("basis_strategy must be 'chain' or 'residual_mr'")
    if field_formulation not in ('split_u', 'total_field'):
        raise ValueError("field_formulation must be 'split_u' or 'total_field'")
    static_split = field_formulation == 'split_u'
    if not static_split and basis_strategy != 'residual_mr':
        raise ValueError("total_field requires basis_strategy='residual_mr'")
    if not static_split and residual_correction:
        raise ValueError('total_field requires residual_correction=False')
    if pole_schedule not in ('hold_first', 'alternate'):
        raise ValueError("pole_schedule must be 'hold_first' or 'alternate'")
    adaptive = basis_strategy == 'residual_mr'
    if ops.get('fe_degree') == 2 and (not adaptive or residual_correction):
        raise ValueError('P2 currently supports residual_mr without residual_correction only')
    if ((not adaptive or residual_correction) and ops.get('curl') is None
            and ops.get('compute_curl') is not None):
        ops['compute_curl']()
    _validate_ops(ops, need_gradient=True)
    base_pole = float(-2 * np.pi * (np.sqrt(f.min()) * np.sqrt(f.max())))
    if not np.isfinite(base_pole) or base_pole >= 0:
        raise ValueError('split-u pole is outside the finite negative range')
    if poles is not None and pole_factor is not None:
        raise ValueError('Specify either poles or pole_factor, not both')
    if pole_factor is not None:
        if (not isinstance(pole_factor, Real) or isinstance(pole_factor, (bool, np.bool_))
                or not np.isfinite(pole_factor) or pole_factor <= 0):
            raise ValueError('pole_factor must be a finite positive real scalar')
        pole_factor = float(pole_factor)
        poles, pole_mode = (base_pole * pole_factor,), 'factor_single'
    elif poles is None:
        middle = np.sqrt(f.min()) * np.sqrt(f.max())
        low, high = f[f <= middle], f[f > middle]
        poles = tuple(float(-2 * np.pi * np.sqrt(part.min()) * np.sqrt(part.max()))
                      for part in (high, low) if len(part))
        pole_mode = 'auto_two_band' if len(poles) == 2 else 'auto_single'
    else:
        pole_mode = 'explicit'
    try:
        poles = tuple(poles)
    except TypeError:
        raise ValueError('split-u poles require one or two finite negative real values') from None
    if (len(poles) not in (1, 2) or any(
            not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value >= 0 for value in poles)):
        raise ValueError('split-u poles require one or two finite negative real values')
    poles = tuple(float(value) for value in poles)
    if any(not np.isfinite(value) or value >= 0 for value in poles):
        raise ValueError('split-u poles must be finite negative float64 values')
    xi = poles[0]
    stats = _stats('split_u', f, xi)
    stats.update(base_pole=base_pole, pole_factor=pole_factor, poles=list(poles),
                 pole_mode=pole_mode, pole_every=int(pole_every), basis_budget=int(max_basis),
                 pole_schedule=pole_schedule,
                 trace=[], linear_solves=[], current_cycle=0, current_pole=xi)
    stats['assembly_seconds'] = ops.get('assembly_seconds', {})
    stats['assembly_counts'] = ops.get('assembly_counts', {})
    stats['test_direct_inverse'] = bool(direct)
    stats['inner_rtol_requested'] = inner_rtol
    stats['pcg_inner_rtol_applied'] = not bool(direct)
    stats['residual_correction'] = bool(residual_correction)
    stats['basis_strategy'] = basis_strategy
    stats.update(field_formulation=field_formulation, static_split=static_split,
                 gradient_cleanup=static_split, static_vector_count=0)
    stats['projection'] = 'minimum_residual' if adaptive else 'galerkin'
    stats['check_every'] = 1 if adaptive else int(check_every)
    if ops.get('fe_degree') == 2:
        stats.update(fe_degree=2, ams_interface='full_interpolation', ams_cycle_type=1,
                     auxiliary_setup=ops.get('auxiliary_setup', {}))
    stats['counts']['direction_defect_checks'] = 0
    if adaptive:
        stats.update(residual_verified=False, verified_basis_columns=None,
                     verification_policy='convergence_or_exit')
        stats['counts'].update(predicted_residual_checks=0, full_residual_checkpoints=0)
    comm = ops['K'].comm.tompi4py()
    q = ops['q'].getArray(readonly=True).copy()
    normq = _norm(comm, q)
    K, M, G = _Action(ops['K']), _Action(ops['M']), _Action(ops['G'])
    C = (_Action(ops['curl']) if residual_correction and ops.get('curl') is not None
         else None)
    resources = [K, M, G] + ([C] if C is not None else [])
    matrices = []
    vectors = np.empty((len(q), int(max_basis)), order='F')
    gram_matrix = np.empty((int(max_basis), int(max_basis)))
    basis = vectors[:, :0]
    coefficients, curl_qr, residual_qr = None, None, None
    P, shifted, P_action, current_pole = None, None, None, None
    last_checked_columns = None
    g = np.zeros_like(q)
    field = np.empty(len(q), dtype=complex)
    field_work = np.empty_like(q)
    next_residual, target_frequency = None, None
    mr_information = None
    started = perf_counter()

    def release_build_resources():
        nonlocal P_action, shifted, P
        if P_action is not None:
            P_action.destroy()
            P_action = None
        if shifted is not None:
            shifted.destroy()
            shifted = None
        if P is not None:
            P.destroy()
            P = None
        while resources:
            resources.pop().destroy()
        while matrices:
            matrices.pop().destroy()

    def verify_fields(V, candidate_coefficients):
        rows = []
        for fi, frequency in enumerate(f):
            omega = 2 * np.pi * frequency
            _split_field(g, V, candidate_coefficients[:, fi], omega, field, field_work)
            unused, rho, physical = _residual(comm, K, M, q, field, omega, normq, stats=stats)
            rows.append(dict(index=fi, frequency=float(frequency), residual=rho,
                             physical_residual=physical, basis_columns=V.shape[1]))
        if adaptive:
            stats['counts']['full_residual_checkpoints'] += 1
        return rows

    def check_basis(force_verify=False):
        nonlocal coefficients, curl_qr, residual_qr, last_checked_columns, next_residual
        nonlocal target_frequency, mr_information
        cached = last_checked_columns == basis.shape[1]
        if cached and (not adaptive or not force_verify or stats['residual_verified']):
            return (not adaptive or stats['residual_verified']) and max(
                row['residual'] for row in stats['per_frequency']) <= tol
        projection_started = perf_counter()
        if cached:
            candidate_coefficients = coefficients
        elif basis.shape[1]:
            V = basis
            gram_started = perf_counter()
            Mr = gram_matrix[:basis.shape[1], :basis.shape[1]]
            gram = float(np.linalg.norm(Mr - np.eye(basis.shape[1]), 2))
            stats['seconds']['gram'] = stats['seconds'].get('gram', 0.0) + perf_counter() - gram_started
            stats['gram_error'] = max(stats['gram_error'], gram)
            if not np.isfinite(gram) or gram > 1e-10:
                raise MORFailure('split-u Gram error %.3e exceeds 1e-10' % gram)
            if adaptive:
                if residual_qr.width != 1 + 2 * V.shape[1]:
                    raise MORFailure('Residual QR does not cover the accepted basis')
                candidate_coefficients, mr_information = _residual_mr(
                    comm, None, None, h, f, stats,
                    qr=residual_qr)
                stats['mr_information'] = mr_information
            elif ops.get('curl') is not None:
                reduced_rhs = _sum(comm, V.T @ h)
                if curl_qr is None:
                    curl_qr = _CurlQR(ops['curl'], stats)
                    resources.append(curl_qr)
                local_R, local_error = None, None
                try:
                    local_R = curl_qr(V)
                except Exception as error:
                    local_error = 'rank %d: %s: %s' % (comm.rank, type(error).__name__, error)
                errors = comm.allgather(local_error)
                if any(error is not None for error in errors):
                    raise MORFailure('Curl QR failed: ' + '; '.join(error for error in errors if error is not None))
                factors = comm.gather(local_R, root=0)

                def reduced():
                    # M = LL^T; transform whitens mass, then diagonalizes curl.
                    mass_factor = np.linalg.cholesky(0.5 * (Mr + Mr.T))
                    R = np.vstack(factors)
                    whitened = np.linalg.solve(mass_factor, R.T).T
                    unused, singular, VH = np.linalg.svd(whitened, full_matrices=False)
                    transform = np.linalg.solve(mass_factor.T, VH.T)
                    return transform @ ((transform.T @ reduced_rhs)[:, None] /
                                        (singular[:, None]**2 + 1j * 2 * np.pi * f[None, :]))
            else:
                reduced_rhs = _sum(comm, V.T @ h)
                Kr = _sum(comm, V.T @ np.column_stack([K(v) for v in basis.T]))

                def reduced():
                    return np.column_stack([np.linalg.solve(Kr + 1j * 2 * np.pi * frequency * Mr,
                                                             reduced_rhs) for frequency in f])
            if not adaptive:
                reduced_started = perf_counter()
                candidate_coefficients = _dense(comm, reduced)
                stats['seconds']['reduced_solve'] = stats['seconds'].get('reduced_solve', 0.0) + perf_counter() - reduced_started
        else:
            candidate_coefficients = np.empty((0, len(f)), dtype=complex)
        stats['seconds']['projection'] = stats['seconds'].get('projection', 0.0) + perf_counter() - projection_started
        predicted = None
        if adaptive:
            if cached:
                predicted = np.asarray(stats['predicted_relative_residuals'])
            elif basis.shape[1]:
                predicted = np.asarray(mr_information['predicted_residuals'], dtype=float) / normq
            elif normq:
                predicted = np.full(f.shape, _norm(comm, h) / normq)
            else:
                predicted = np.zeros(len(f))
            valid = (predicted.shape == f.shape and np.isfinite(predicted).all()
                     and np.all(predicted >= 0))
            if not comm.allreduce(bool(valid), op=MPI.LAND):
                raise MORFailure('Invalid predicted screening residuals')
            if not cached:
                stats['counts']['predicted_residual_checks'] += len(f)
        verified = (not adaptive or force_verify or np.all(predicted <= tol))
        if verified:
            rows = verify_fields(basis, candidate_coefficients)
        else:
            rows = [dict(index=fi, frequency=float(frequency), residual=float(predicted[fi]),
                         physical_residual=float(predicted[fi]), basis_columns=basis.shape[1])
                    for fi, frequency in enumerate(f)]
        coefficients = candidate_coefficients
        if adaptive:
            stats.update(residual_verified=bool(verified),
                         residual_kind='assembled_matrix' if verified else 'reduced_qr',
                         predicted_relative_residuals=predicted.tolist())
            if verified:
                stats['verified_basis_columns'] = basis.shape[1]
            for row in rows:
                row.update(residual_verified=bool(verified),
                           predicted_residual=float(predicted[row['index']]))
            worst = int(np.argmax([row['residual'] for row in rows]))
            target_frequency = float(f[worst])
            if basis.shape[1]:
                a = coefficients[:, worst]
                s = 2j * np.pi * target_frequency
                weights = np.empty(1 + 2 * len(a), dtype=complex)
                weights[0] = 1.
                weights[1::2], weights[2::2] = -a, -s * a
                next_residual = residual_qr.combination(weights)
            else:
                next_residual = h.astype(complex)
        stats['per_frequency'] = rows
        if not cached:
            stats['counts']['checkpoints'] = stats['counts'].get('checkpoints', 0) + 1
        last_checked_columns = basis.shape[1]
        entry = dict(cycle=stats['current_cycle'], pole=xi,
            basis_columns=basis.shape[1], gram_error=stats['gram_error'],
            relative_residuals=[row['residual'] for row in rows],
            max_relative_residual=max(row['residual'] for row in rows),
            remaining_frequency_count=sum(row['residual'] > tol for row in rows),
            counts=stats['counts'].copy())
        if adaptive:
            entry.update(residual_verified=bool(verified), residual_kind=stats['residual_kind'],
                         predicted_relative_residuals=predicted.tolist())
        if cached:
            stats['trace'][-1] = entry
        else:
            stats['trace'].append(entry)
        return verified and max(max(row['residual'], row['physical_residual']) for row in rows) <= tol

    def reject_extension(message):
        # A checkpoint interval can request a direction after the retained basis
        # already solves every frequency. Prove that before reporting failure;
        # never accept the rejected direction or alter the final catch status.
        if basis.shape[1] and check_basis(force_verify=adaptive):
            diagnostic.update(status='extension_not_needed', rejected_reason=message)
            raise _ConvergedBeforeExtension()
        raise MORFailure(message)

    try:
        if not normq:
            h = q.copy()
            check_basis()
            stats['status'] = 'converged'
        else:
            static_started = perf_counter()
            if static_split:
                MG = ops['M'].matMult(ops['G'])
                L = ops['G'].transposeMatMult(MG)
                MG.destroy()
                matrices.append(L)
                diagonal_vec = L.createVecLeft()
                L.getDiagonal(diagonal_vec)
                ldiag = diagonal_vec.getArray(readonly=True).copy()
                diagonal_vec.destroy()
                if not comm.allreduce(bool(np.all(ldiag > 0)), op=MPI.LAND):
                    raise MORFailure('Nodal Poisson diagonal is not strictly positive')
                lo, hi = L.getOwnershipRange()
                # A weak-air gauge amplifies roundoff in tiny cleanup RHSs.
                pinned_dof, gauge_metadata = _select_poisson_gauge(ops, L, ldiag)
                stats.update(gauge_metadata)
                # Collective call; only the rank owning the gauge lists it.
                pinned = np.array([pinned_dof] if lo <= pinned_dof < hi else [], dtype=PETSc.IntType)
                L.zeroRowsColumns(pinned, diag=1.0)
                poisson = _LinearSolve(L, stats, 'poisson', direct=direct, pinned=True,
                                       pinned_dof=pinned_dof)
                resources.append(poisson)

                def gradient(rhs):
                    return G(poisson(G(rhs, transpose=True), rtol=1e-10, max_it=300))

                g = gradient(q)
                stats['static_vector_count'] = 1
                h = q - M(g)
            else:
                h = q.copy()
            if adaptive:
                residual_qr = _ResidualQR(comm, len(q), max_basis, stats)
                resources.append(residual_qr)
                residual_qr.append(h)
            stats['static_h_norm'] = _norm(comm, h)
            stats['seconds']['static_split'] = perf_counter() - static_started
            if check_basis():
                stats['status'] = 'converged'
            else:
                if residual_correction:
                    diagonal_vec = ops['M'].createVecRight()
                    ops['M'].getDiagonal(diagonal_vec)
                    diagonal = diagonal_vec.getArray(readonly=True).copy()
                    diagonal_vec.destroy()
                    lower = ops.get('mass_lower_bound')
                    if lower is None:
                        lower = ops['compute_mass_lower_bound']()
                    lower = float(lower)
                    if (not np.isfinite(lower) or lower <= 0 or
                            not comm.allreduce(bool(np.all(diagonal > 0)), op=MPI.LAND)):
                        raise MORFailure('Invalid full-mass diagonal lower bound')
                rhs = h.copy()
                for column in range(int(max_basis)):
                    cycle = basis.shape[1] // int(pole_every) + 1
                    xi = poles[(cycle - 1) % len(poles)]
                    if pole_schedule == 'hold_first' and cycle >= 3:
                        xi = poles[0]
                    stats.update(current_cycle=cycle, current_pole=xi)
                    if xi != current_pole:
                        if P_action is not None:
                            P_action.destroy()
                            P_action = None
                        if shifted is not None:
                            shifted.destroy()
                            shifted = None
                        if P is not None:
                            P.destroy()
                            P = None
                        P = ops['K'].copy()
                        P.axpy(-xi, ops['M'], structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
                        P.assemble()
                        high_order_options = (dict(ams_interpolation=ops['ams_interpolation'])
                                              if ops.get('fe_degree') == 2 and not direct else {})
                        shifted = _LinearSolve(P, stats, 'P', direct=direct,
                            gradient=None if direct else ops['G'], constants=ops.get('constants'),
                            **high_order_options)
                        if residual_correction:
                            P_action = _Action(P)
                        current_pole = xi
                    direction_started = perf_counter()
                    diagnostic = dict(column=column, cycle=cycle, pole=xi,
                                      iterations=0, refinements=0,
                                      candidates=[], status='building')
                    stats['directions'].append(diagnostic)
                    if adaptive:
                        rhs, selection = _real_residual_direction(comm, next_residual)
                        diagnostic.update(target_frequency=target_frequency,
                                          real_selection=selection,
                                          target_residual=max(row['residual'] for row in stats['per_frequency']))

                    def request(vector, rtol):
                        remaining = 300 - diagnostic['iterations']
                        if not direct and remaining <= 0:
                            reject_extension('Shared 300-step PCG budget exhausted')
                        try:
                            solution = shifted(vector, rtol=rtol, max_it=max(1, remaining))
                        except MORFailure as error:
                            diagnostic['iterations'] += 0 if direct else shifted.last_iterations
                            reject_extension(str(error))
                        diagnostic['iterations'] += 0 if direct else shifted.last_iterations
                        return solution

                    vector = request(rhs, inner_rtol)
                    accepted, deflated = None, False
                    for refinement in range(3 if residual_correction else 1):
                        projected = vector.copy() if residual_correction else None
                        remainder, initial, post, mass_remainder = _orthogonal(
                            comm, M, vector, basis.T, gram_matrix[:basis.shape[1], :basis.shape[1]])
                        nodal = projection_error = 0.0 if static_split else None
                        gram = 0.0
                        for projection_pass in range(4):
                            if not initial or post <= 1e-12 * initial:
                                break
                            candidate = remainder / post
                            mass_candidate = M(candidate)
                            if static_split:
                                node_defect = G(mass_candidate, transpose=True)
                                nodal = _norm(comm, node_defect / np.sqrt(ldiag))
                            gram_started = perf_counter()
                            extended_gram = _extend_gram(comm, gram_matrix[:basis.shape[1], :basis.shape[1]],
                                                         basis, candidate, mass_candidate)
                            gram = float(np.linalg.norm(extended_gram - np.eye(basis.shape[1] + 1), 2))
                            stats['seconds']['gram'] = stats['seconds'].get('gram', 0.0) + perf_counter() - gram_started
                            if gram > 1e-10:
                                cleaned, unused, post, mass_remainder = _orthogonal(
                                    comm, M, remainder, basis.T, gram_matrix[:basis.shape[1], :basis.shape[1]])
                                if residual_correction:
                                    projected += cleaned - remainder
                                remainder = cleaned
                                if post <= 1e-12 * initial:
                                    break
                                candidate = remainder / post
                                mass_candidate = M(candidate)
                                if static_split:
                                    node_defect = G(mass_candidate, transpose=True)
                                    nodal = _norm(comm, node_defect / np.sqrt(ldiag))
                                gram_started = perf_counter()
                                extended_gram = _extend_gram(comm, gram_matrix[:basis.shape[1], :basis.shape[1]],
                                                             basis, candidate, mass_candidate)
                                gram = float(np.linalg.norm(extended_gram - np.eye(basis.shape[1] + 1), 2))
                                stats['seconds']['gram'] = stats['seconds'].get('gram', 0.0) + perf_counter() - gram_started
                                if gram > 1e-10:
                                    reject_extension('split-u direction Gram acceptance failed')
                            if not static_split or nodal <= 1e-12:
                                break
                            if projection_pass:
                                projection_error = _mass_norm(comm, M, gradient(mass_candidate))
                                if projection_error <= 1e-10:
                                    break
                            if projection_pass == 3:
                                reject_extension('split-u normalized gradient constraint failed')
                            cleaned = remainder - gradient(mass_remainder)
                            cleaned, unused, post, mass_remainder = _orthogonal(
                                comm, M, cleaned, basis.T, gram_matrix[:basis.shape[1], :basis.shape[1]])
                            if residual_correction:
                                projected += cleaned - remainder
                            remainder = cleaned
                        row = dict(refinement=refinement, post_m_norm=float(post),
                                   scaled_nodal_defect=None if nodal is None else float(nodal),
                                   gram_error=gram,
                                   projection_error=(None if projection_error is None
                                                     else float(projection_error)),
                                   gradient_constraint_checked=static_split)
                        if residual_correction:
                            # Preserve old-span coefficients while auditing the
                            # full candidate after projection and cleanup.
                            defect = rhs - P_action(projected)
                            physical_defect = (defect if C is None else
                                               rhs - C(C(projected), transpose=True) + xi * M(projected))
                            denominator = max(_norm(comm, rhs), np.finfo(float).tiny)
                            rho = max(_norm(comm, defect), _norm(comm, physical_defect)) / denominator
                            bound = (np.sqrt(_sum(comm, np.dot(physical_defect / diagonal,
                                                              physical_defect)) / lower) /
                                     ((-xi) * max(post, np.finfo(float).tiny)))
                            row.update(residual=rho, direction_bound=float(bound))
                            stats['counts']['direction_defect_checks'] += 1
                        diagnostic['candidates'].append(row)
                        diagnostic.update(row)
                        if ((not initial or post <= 1e-12 * initial)
                                and (not residual_correction or rho <= 1e-6)):
                            deflated = True
                            diagnostic['status'] = 'rank_deflated'
                            break
                        if (post > 1e-12 * initial
                                and (not residual_correction or (rho <= 1e-6 and bound <= 1e-3))):
                            accepted = candidate
                            diagnostic['status'] = 'accepted'
                            break
                        if refinement == 2:
                            reject_extension('split-u direction failed actual-defect acceptance '
                                             '(rho %.3e, bound %.3e)' % (rho, bound))
                        # Correct the complete candidate whose defect was
                        # measured, then reapply the same acceptance procedure.
                        vector = projected + request(defect, 1e-2)
                        diagnostic['refinements'] += 1
                    stats['seconds']['directions'] = stats['seconds'].get('directions', 0.0) + perf_counter() - direction_started
                    if accepted is not None:
                        n = basis.shape[1]
                        vectors[:, n] = accepted
                        gram_matrix[:n + 1, :n + 1] = extended_gram
                        basis = vectors[:, :n + 1]
                        stats['max_basis'] = basis.shape[1]
                        if adaptive:
                            residual_qr.append(K(accepted), mass_candidate)
                        else:
                            rhs = mass_candidate.copy()
                    # Accepted columns now live in V/QR. These work vectors
                    # are not needed during projection or full-field checks.
                    vector = remainder = mass_remainder = candidate = None
                    mass_candidate = accepted = cleaned = projected = None
                    if adaptive:
                        rhs = None
                    if (adaptive or deflated or basis.shape[1] % int(check_every) == 0
                            or basis.shape[1] % int(pole_every) == 0 or column + 1 == max_basis):
                        if check_basis(force_verify=adaptive and (
                                deflated or column + 1 == max_basis)):
                            stats['status'] = 'converged'
                            break
                        if deflated:
                            raise MORFailure('No independent split-u direction remains before convergence')
                if stats['status'] != 'converged':
                    raise MORFailure('split-u exhausted %d basis columns' % max_basis)
    except _ConvergedBeforeExtension:
        stats['status'] = 'converged'
    except Exception as error:
        stats.update(status='failed', failure='%s: %s' % (type(error).__name__, error))
        # A final partial basis check is required even between checkpoints.
        try:
            check_basis(force_verify=adaptive)
        except Exception as secondary:
            stats['failure'] += '; final check: %s' % secondary
    finally:
        try:
            if adaptive and coefficients is not None and not stats['residual_verified']:
                # A failed later QR may leave an older finite coefficient
                # prefix. Verify exactly that prefix before emitting it.
                try:
                    retained = basis[:, :coefficients.shape[0]]
                    rows = verify_fields(retained, coefficients)
                    predicted = stats['predicted_relative_residuals']
                    for row in rows:
                        row.update(residual_verified=True,
                                   predicted_residual=predicted[row['index']])
                    stats.update(per_frequency=rows, residual_verified=True,
                                 residual_kind='assembled_matrix',
                                 verified_basis_columns=retained.shape[1])
                    stats['trace'][-1].update(residual_verified=True,
                        residual_kind='assembled_matrix',
                        relative_residuals=[row['residual'] for row in rows],
                        max_relative_residual=max(row['residual'] for row in rows),
                        remaining_frequency_count=sum(row['residual'] > tol for row in rows),
                        counts=stats['counts'].copy())
                except Exception as error:
                    failure = 'retained-field verification: %s: %s' % (type(error).__name__, error)
                    stats.update(status='failed', failure=(stats['failure'] + '; ' + failure
                                 if stats['failure'] else failure))
            # Sampling needs only V and the reduced coefficients. Drop QR,
            # AMG/AMS, shifted matrices and direction work before output.
            release_build_resources()
            gram_matrix = next_residual = rhs = vector = remainder = None
            mass_remainder = mass_candidate = accepted = cleaned = projected = None
            # Reconstruct one frequency at a time, including on numerical failure.
            if coefficients is not None and coefficients.shape[0] <= basis.shape[1]:
                emitted_basis = basis[:, :coefficients.shape[0]]
                stats['emitted_basis_columns'] = coefficients.shape[0]
                for fi, frequency in enumerate(f):
                    _split_field(g, emitted_basis, coefficients[:, fi], 2 * np.pi * frequency,
                                 field, field_work)
                    if comm.allreduce(bool(np.isfinite(field).all()), op=MPI.LAND):
                        _emit(ops, fi, field, emit)
            else:
                stats['emitted_basis_columns'] = 0
                for fi in range(len(f)):
                    _emit(ops, fi, g.astype(complex), emit)
        except Exception as error:
            stats.update(status='failed', failure='%s: %s' % (type(error).__name__, error))
        finally:
            release_build_resources()
            stats['seconds']['total'] = perf_counter() - started
    for key, duration in stats['seconds'].items():
        stats['seconds'][key] = comm.allreduce(duration, op=MPI.MAX)
    _finalize_stats(stats)
    stats['dynamic_basis_columns'] = stats['basis_columns']
    return stats


def _finalize_stats(stats):
    """Small public statistics, without retaining full frequency fields."""
    stats['converged'] = stats['status'] == 'converged'
    stats['relative_residuals'] = [row['residual'] for row in stats['per_frequency']]
    stats['physical_relative_residuals'] = [row['physical_residual'] for row in stats['per_frequency']]
    stats['basis_columns'] = stats['max_basis']
    counts = stats['counts']
    counts['shift_factorizations'] = counts.get('P_factorizations', 0)
    counts['mass_factorizations'] = counts.get('M_factorizations', 0)
    counts['ams_setups'] = counts.get('P_setups', 0)
    counts.setdefault('poisson_setups', 0)

