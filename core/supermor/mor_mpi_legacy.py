"""P1 SUPERMOR kernel used by the paper's layered-sphere MPI experiment.

The three historical definitions below are retained verbatim from
custEM/core/mor.py at 86c71eafa94204a6fc69a7a6b0bd829e009ac7fa
(SHA-256 b5a67400f0711ccd0a543eba2500a7cfc0d3e63b3fa27c9d3952ac0d70597917).
They retain the Kg image and -Kg/s right-hand-side term. Common numerical
helpers come from the independently migrated release core; every shared
helper is AST-identical to that snapshot except _LinearSolve's opt-in CPC
tracing, which does not execute for the split-u route used here.

This compatibility module supports P1 only. The current SUPERMOR method
remains the normal public core; selecting this module is explicit at the
MPI experiment entry. No historical measured timings are reused.
"""
from .mor import (
    np, MPI, PETSc, Real, perf_counter, dgeqrf, dormqr, MORFailure,
    _Action, _ConvergedBeforeExtension, _CurlQR, _LinearSolve,
    _dense, _emit, _extend_gram, _finalize_stats, _frequencies,
    _mass_norm, _norm, _orthogonal, _real_residual_direction,
    _residual, _select_poisson_gauge, _split_field, _stats, _sum,
    _validate_ops,
)

SOURCE_COMMIT = "86c71eafa94204a6fc69a7a6b0bd829e009ac7fa"
SOURCE_MOR_SHA256 = "b5a67400f0711ccd0a543eba2500a7cfc0d3e63b3fa27c9d3952ac0d70597917"

class _ResidualQR:
    """Local append-only Householder QR of [h, Kg, Kv1, Mv1, ...].

    New image pairs are copied into the factor and can then be released.
    Each rank retains its own reflectors; only the padded small R is gathered.
    """

    def __init__(self, comm, local_rows, max_basis, stats):
        started = perf_counter()
        self.comm, self.stats = comm, stats
        self.local_rows, self.capacity = int(local_rows), 2 * int(max_basis) + 2
        self.width = 0
        self.factor = self.tau = self.scales = None
        for key in ('residual_qr_setup', 'residual_qr_apply', 'residual_qr_factor'):
            stats['seconds'].setdefault(key, 0.0)
        for key in ('residual_qr_updates', 'residual_qr_columns'):
            stats['counts'].setdefault(key, 0)
        error = None
        try:
            if self.local_rows < 0 or self.capacity < 2:
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

    def __call__(self, KV, MV, h, kg):
        error, columns = None, -1
        try:
            columns = KV.shape[1]
            valid = (self.factor is not None and KV.ndim == 2 and
                     KV.shape[0] == self.local_rows and MV.shape == KV.shape and
                     h.shape == kg.shape == (self.local_rows,) and
                     not any(np.iscomplexobj(a) for a in (KV, MV, h, kg)) and
                     self.width <= 2 * columns + 2 <= self.capacity)
            if not valid:
                raise ValueError('Invalid residual QR prefix or image dimensions')
            first = max(0, (self.width - 2) // 2)
            if not all(np.isfinite(a).all() for a in (KV[:, first:], MV[:, first:])):
                raise ValueError('Residual QR requires finite new image columns')
            if not self.width and not all(np.isfinite(a).all() for a in (h, kg)):
                raise ValueError('Residual QR requires finite h and Kg')
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        requests = self.comm.allgather((error, columns, self.width))
        errors = [request[0] for request in requests if request[0]]
        if errors or len({request[1:] for request in requests}) != 1:
            raise MORFailure('Invalid residual QR inputs: ' + '; '.join(errors or ['inconsistent prefixes']))

        if not self.width:
            self.append(h, kg)
        for index in range((self.width - 2) // 2, columns):
            self.append(KV[:, index], MV[:, index])
        return self.factors()

    def append(self, first, second):
        """Commit one image pair collectively, without retaining its inputs."""
        append_started = perf_counter()
        error = None
        try:
            if (self.factor is None or self.width + 2 > self.capacity or
                    any(a.shape != (self.local_rows,) or np.iscomplexobj(a) or
                        not np.isfinite(a).all() for a in (first, second))):
                raise ValueError('Invalid residual QR image pair')
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        requests = self.comm.allgather((error, self.width))
        errors = [request[0] for request in requests if request[0]]
        if errors or len({request[1] for request in requests}) != 1:
            raise MORFailure('Invalid residual QR append: ' + '; '.join(errors or ['inconsistent prefixes']))
        old = self.width
        pair = self.factor[:, old:old + 2]
        pair[:, 0], pair[:, 1] = first, second
        scales = np.sqrt(_sum(self.comm, np.einsum('ij,ij->j', pair, pair)))
        if not np.isfinite(scales).all():
            raise MORFailure('Nonfinite residual QR column norms')
        scales[scales == 0] = 1.0
        pair /= scales
        error, new_tau = None, None
        try:
            k = min(self.local_rows, old)
            started = perf_counter()
            if k:
                reflectors = self.factor[:, :k]
                # Only two RHS: minimum workspace avoids rebuilding the
                # old blocked reflector products on every append.
                transformed, unused, info = dormqr(b'L', b'T', reflectors,
                    self.tau[:k], pair, lwork=2, overwrite_c=1)
                if info or not np.shares_memory(transformed, pair):
                    raise MORFailure('Residual QR reflector application/buffer reuse failed')
            self.stats['seconds']['residual_qr_apply'] += perf_counter() - started
            started = perf_counter()
            if self.local_rows > k:
                # A trailing row slice is not F-contiguous. Copy only the
                # new two-column tail, never the old reflector matrix.
                tail = np.array(pair[k:], order='F', copy=True)
                unused, unused_tau, optimal, info = dgeqrf(tail, lwork=-1, overwrite_a=1)
                if info:
                    raise MORFailure('Residual QR tail workspace query failed')
                factor, new_tau, unused, info = dgeqrf(
                    tail, lwork=int(optimal[0]), overwrite_a=1)
                if info or not np.shares_memory(factor, tail):
                    raise MORFailure('Residual QR tail factorization/buffer reuse failed')
                pair[k:] = factor
            self.stats['seconds']['residual_qr_factor'] += perf_counter() - started
        except Exception as exception:
            error = '%s: %s' % (type(exception).__name__, exception)
        # A failed append leaves old reflectors untouched. All ranks
        # commit this pair together before entering another norm reduce.
        errors = self.comm.allgather(error)
        if any(errors):
            raise MORFailure('Residual QR append failed: ' + '; '.join(e for e in errors if e))
        if new_tau is not None:
            self.tau[k:k + len(new_tau)] = new_tau
        self.scales[old:old + 2] = scales
        self.width += 2
        self.stats['counts']['residual_qr_updates'] += 1
        self.stats['counts']['residual_qr_columns'] += 2
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


def _residual_mr(comm, KV, MV, h, kg, frequencies, stats, qr=None):
    """Distributed Euclidean MR via incremental QR and small, full-mode SVD.

    The target is the original E equation, including -Kg/s. Only small R
    factors and reduced coefficients are communicated; no normal equations.
    """
    columns = (qr.width - 2) // 2 if KV is None else KV.shape[1]
    seconds = stats.setdefault('seconds', {})
    qr_seconds = seconds.get('residual_joint_QR', 0.0)
    started = perf_counter()
    temporary = qr is None
    if temporary:
        qr = _ResidualQR(comm, KV.shape[0], columns, stats)
    try:
        local_R, scales = qr.factors() if KV is None else qr(KV, MV, h, kg)
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
            A = R[:, 2::2] + s * R[:, 3::2]
            b = R[:, 0] - R[:, 1] / s
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
    g=0, h=q and Kg=0, so the basis represents E=s*V*a directly. It skips the
    explicit static Poisson solve and per-column gradient cleanup, while AMS
    still receives the discrete gradient and may perform internal nodal solves.
    This option requires ``basis_strategy='residual_mr'`` and
    ``residual_correction=False``.
    ``static_vector_count`` counts explicitly constructed static g vectors,
    including a zero projection of a nonzero source; the zero-source shortcut
    and total-field formulation construct none.

    The default ``basis_strategy='residual_mr'`` minimizes the complete E matrix
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
    kg, next_residual, target_frequency = None, None, None
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
                if residual_qr.width != 2 + 2 * V.shape[1]:
                    raise MORFailure('Residual QR does not cover the accepted basis')
                candidate_coefficients, mr_information = _residual_mr(
                    comm, None, None, h, kg, f, stats,
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
                # h and Kg are real, so h-Kg/(i*omega) has orthogonal real and
                # imaginary parts. No full frequency fields are needed here.
                predicted = np.hypot(_norm(comm, h), _norm(comm, kg) / (2 * np.pi * f)) / normq
            else:
                predicted = np.zeros(len(f))
            valid = (predicted.shape == f.shape and np.isfinite(predicted).all()
                     and np.all(predicted >= 0))
            if not comm.allreduce(bool(valid), op=MPI.LAND):
                raise MORFailure('Invalid predicted original-equation residuals')
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
                weights = np.empty(2 + 2 * len(a), dtype=complex)
                weights[:2] = 1., 0.  # u residual; MR's E target also includes -Kg/s.
                weights[2::2], weights[3::2] = -a, -s * a
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
                kg = K(g) if static_split else np.zeros_like(q)
                residual_qr = _ResidualQR(comm, len(q), max_basis, stats)
                resources.append(residual_qr)
                residual_qr.append(h, kg)
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


_frozen_solve_split_u = solve_split_u


def solve_split_u(ops, frequencies, *args, **kwargs):
    """Run the frozen MPI split-u definition with a P1-only boundary."""
    degree = ops.get("fe_degree", 1)
    if isinstance(degree, bool) or degree != 1:
        raise NotImplementedError("The historical sphere MPI kernel supports P1 only")
    stats = _frozen_solve_split_u(ops, frequencies, *args, **kwargs)
    stats["historical_kernel"] = dict(commit=SOURCE_COMMIT,
        original_mor_sha256=SOURCE_MOR_SHA256, retained_Kg_MR_term=True,
        common_helpers="release core; P1 numerical definitions retained")
    return stats
