"""Production MPI DFOM and IFOM numerical kernels.

The functions below are preserved from the paper's terrain/sphere worker.
They contain no experiment launcher, model paths, or output workflow.
"""
import math
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


class BlockLayout:
    """Scatter between original edge ownership and saved MatNest global ISs.

    No assumption about real/imaginary ordering in the converted AIJ vector.
    Each index set contains this rank's entries, in its edge-vector order.
    """
    def __init__(self, component, block, index_sets):
        from petsc4py import PETSc
        self.PETSc = PETSc
        self.parts = [component.duplicate(), component.duplicate()]
        lo, hi = component.getOwnershipRange()
        self.component_is = PETSc.IS().createStride(hi-lo, first=lo, step=1, comm=component.comm)
        require(len(index_sets) == 2, 'Expected two MatNest component index sets')
        self.scatters = []
        for part, indices in zip(self.parts, index_sets):
            require(indices.getLocalSize() == hi-lo, 'MatNest IS/local edge ownership mismatch')
            self.scatters.append(PETSc.Scatter().create(part, self.component_is, block, indices))

    def pack(self, arrays, block):
        block.set(0.)
        for part, scatter, values in zip(self.parts, self.scatters, arrays):
            require(len(values) == part.getLocalSize(), 'Wrong local component size')
            part.getArray()[:] = values
            scatter.scatter(part, block, addv=self.PETSc.InsertMode.INSERT_VALUES,
                            mode=self.PETSc.ScatterMode.FORWARD)

    def unpack(self, block):
        for part, scatter in zip(self.parts, self.scatters):
            scatter.scatter(block, part, addv=self.PETSc.InsertMode.INSERT_VALUES,
                            mode=self.PETSc.ScatterMode.REVERSE)
        return tuple(part.getArray(readonly=True).copy() for part in self.parts)

    def destroy(self):
        for item in self.scatters+self.parts+[self.component_is]:
            item.destroy()


def full_order(ops, frequencies, emit, direct):
    """Paper DFOM or FGMRES/PRESB IFOM, with original-matrix global audits."""
    if not direct and ops.get('fe_degree', 1) == 2:
        raise NotImplementedError('P2 IFOM is not supported by the paper MPI solver')
    import numpy as np
    from mpi4py import MPI
    from petsc4py import PETSc
    from . import mor
    K, M, q = ops['K'], ops['M'], ops['q'].getArray(readonly=True).copy()
    comm = K.comm.tompi4py()
    actions = [mor._Action(K), mor._Action(M)]
    normq = float(ops['q'].norm())
    stats = dict(route='DFOM' if direct else 'IFOM', status='running', failure=None,
                 residual_kind='assembled_matrix', per_frequency=[], counts={}, seconds={})
    started = time.perf_counter()
    try:
        for index, frequency in enumerate(frequencies):
            omega = 2*np.pi*frequency
            local = dict(counts={}, seconds={}, linear_solves=[], max_basis=0)
            row = dict(index=index, frequency=float(frequency), status='failed')
            resources, direct_solver, inner, outer = [], None, None, None
            bridges, saved_is = [], []
            try:
                B = M.copy(); B.scale(omega)
                negative_B = B.copy(); negative_B.scale(-1.)
                resources.extend([B, negative_B])
                block = PETSc.Mat().createNest([[K, negative_B], [B, K]], comm=K.comm)
                resources.append(block)
                block.assemble()
                row_is, col_is = block.getNestISs()
                row_is = [indices.copy() for indices in row_is]
                col_is = [indices.copy() for indices in col_is]
                saved_is.extend(row_is+col_is)
                block.convert('aij')
                rhs, solution = block.createVecLeft(), block.createVecRight()
                resources.extend([rhs, solution])
                left = BlockLayout(ops['q'], rhs, row_is)
                right = BlockLayout(ops['q'], solution, col_is)
                bridges.extend([left, right])
                left.pack((np.zeros_like(q), omega*q), rhs)
                solution.set(0.)
                setup_tick = time.perf_counter()
                if direct:
                    direct_solver = mor._LinearSolve(block, local, 'DFOM', direct=True)
                else:
                    H = K.copy()
                    resources.append(H)
                    H.axpy(omega, M, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
                    H.assemble()
                    inner = mor._LinearSolve(H, local, 'IFOM_inner', gradient=ops['G'],
                                             constants=ops['constants'], require_true_residual=False)
                    require(inner.ksp.getType() == 'cg', 'IFOM inner method must be PCG')

                    class PRESB:
                        def apply(self, pc, x, y):
                            r1, r2 = left.unpack(x)
                            t = inner(r1+r2, rtol=1e-3, max_it=300)
                            h = inner(r2-omega*actions[1](t), rtol=1e-3, max_it=300)
                            right.pack((t+h, h), y)

                    outer = PETSc.KSP().create(K.comm)
                    outer.setOperators(block)
                    outer.setType('fgmres')
                    outer.setGMRESRestart(100)
                    outer.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                    outer.setTolerances(rtol=1e-6, atol=0., max_it=2000)
                    outer.setInitialGuessNonzero(False)
                    outer.getPC().setType('python')
                    outer.getPC().setPythonContext(PRESB())
                    outer.setUp()
                row['setup_s'] = comm.allreduce(time.perf_counter()-setup_tick, op=MPI.MAX)
                solve_tick = time.perf_counter()
                if direct:
                    solution.getArray()[:] = direct_solver(rhs.getArray(readonly=True))
                else:
                    outer.solve(rhs, solution)

                def original_residual():
                    re, im = right.unpack(solution)
                    _, rho, _ = mor._residual(comm, *actions, q, re+1j*im, omega, normq, stats=stats)
                    return re, im, rho

                real_values, imag_values, rho = original_residual()
                row['initial_residual'] = rho
                corrections = []
                if direct:
                    # Same LU only; keep the lowest finite original-matrix residual.
                    best = solution.copy()
                    residual = rhs.duplicate()
                    try:
                        best_rho = rho
                        for attempt in range(3):
                            if best_rho <= 1e-10:
                                break
                            best.copy(solution)
                            block.mult(solution, residual)
                            residual.aypx(-1., rhs)  # b-Ax
                            correction = direct_solver(residual.getArray(readonly=True))
                            solution.getArray()[:] += correction
                            _, _, candidate = original_residual()
                            improved = bool(np.isfinite(candidate) and candidate < best_rho)
                            corrections.append(dict(attempt=attempt+1, residual=candidate, retained=improved))
                            if improved:
                                solution.copy(best)
                                best_rho = candidate
                        best.copy(solution)
                        real_values, imag_values, rho = original_residual()
                    finally:
                        best.destroy(); residual.destroy()
                row.update(residual=rho, correction_attempts=corrections,
                           solve_s=comm.allreduce(time.perf_counter()-solve_tick, op=MPI.MAX),
                           outer_iterations=0 if direct else int(outer.getIterationNumber()),
                           outer_reason=int(direct_solver.ksp.getConvergedReason() if direct else outer.getConvergedReason()))
                # Equation success and strict reference qualification are separate.
                # Keep every finite <=1e-6 solution, even if three same-LU
                # corrections did not attain the 1e-10 reference target.
                threshold = 1e-6
                row['equation_tolerance'] = threshold
                if direct:
                    row.update(reference_target=1e-10,
                               reference_accepted=bool(math.isfinite(rho) and rho <= 1e-10))
                require(row['outer_reason'] > 0 and math.isfinite(rho) and rho <= threshold,
                        'Full-order original-matrix residual failed: '+str(rho))
                real, imag = K.createVecRight(), K.createVecRight()
                try:
                    real.getArray()[:] = real_values
                    imag.getArray()[:] = imag_values
                    emit(index, real, imag)
                finally:
                    real.destroy(); imag.destroy()
                row['status'] = 'converged'
            except Exception as caught:
                row['failure'] = type(caught).__name__+': '+str(caught)
                row['traceback'] = traceback.format_exc()
                stats['failure'] = row['failure']
            finally:
                row['linear_solver_stats'] = local
                stats['per_frequency'].append(row)
                if outer is not None:
                    outer.destroy()
                for solver in (direct_solver, inner):
                    if solver is not None:
                        solver.destroy()
                for bridge in bridges:
                    bridge.destroy()
                for item in reversed(resources):
                    item.destroy()
                for indices in saved_is:
                    indices.destroy()
            if row['status'] != 'converged':
                break
        stats['status'] = ('converged' if len(stats['per_frequency']) == len(frequencies) and
                           all(row['status'] == 'converged' for row in stats['per_frequency']) else 'failed')
        if direct:
            qualified = [row['index'] for row in stats['per_frequency'] if row.get('reference_accepted') is True]
            unverified = [row['index'] for row in stats['per_frequency'] if row.get('reference_accepted') is not True]
            stats.update(reference_target=1e-10, reference_accepted_indices=qualified,
                         unverified_reference_indices=unverified,
                         reference_status='qualified' if len(qualified) == len(frequencies) else 'unverified')
        stats['counts'].update(factorizations=sum(row['linear_solver_stats']['counts'].get('DFOM_factorizations', 0)
                                                 for row in stats['per_frequency']),
                               ams_setups=sum(row['linear_solver_stats']['counts'].get('IFOM_inner_setups', 0)
                                              for row in stats['per_frequency']),
                               inner_rhs=sum(row['linear_solver_stats']['counts'].get('IFOM_inner_rhs', 0)
                                             for row in stats['per_frequency']),
                               inner_iterations=sum(row['linear_solver_stats']['counts'].get('IFOM_inner_iterations', 0)
                                                    for row in stats['per_frequency']),
                               outer_iterations=sum(row.get('outer_iterations', 0) for row in stats['per_frequency']))
        stats['seconds']['total'] = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
        return stats
    finally:
        for action in actions:
            action.destroy()
