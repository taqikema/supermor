"""P1 exact sequence, consistent material mass and fitted source contracts."""
import numpy as np

from supermor.operators import assemble_mor

from .fe_helpers import full_matrix, independent_matrices, small_model


def test_p1_operators_match_independent_ufl_and_exact_sequence():
    model = small_model()
    ops = assemble_mor(model)
    product = None
    try:
        expected_k, expected_m = independent_matrices(model)
        np.testing.assert_allclose(full_matrix(ops["K"]), expected_k,
            rtol=2e-13, atol=2e-14)
        mass = full_matrix(ops["M"])
        np.testing.assert_allclose(mass, expected_m, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(mass, mass.T, rtol=0., atol=2e-14)
        np.linalg.cholesky(mass)
        assert np.linalg.norm(mass - np.diag(np.diag(mass))) > .01
        product = ops["K"].matMult(ops["G"])
        assert product.norm() / ops["K"].norm() < 1e-13
        assert ops["K"].getSize()[0] == model.FS.V.dim()
    finally:
        if product is not None:
            product.destroy()
        ops["destroy"]()


def test_fitted_line_source_has_negative_current_moment_and_reverses():
    model = small_model()
    ops = assemble_mor(model)
    reverse = None
    try:
        moments = [ops["q"].dot(vector) for vector in ops["constants"]]
        np.testing.assert_allclose(moments, [-1., 0., 0.], rtol=0., atol=1e-13)
        model.FE.tx[0] = list(reversed(model.FE.tx[0]))
        reverse = assemble_mor(model)
        np.testing.assert_allclose(reverse["q"].getArray(readonly=True),
            -ops["q"].getArray(readonly=True), rtol=0., atol=1e-13)
    finally:
        if reverse is not None:
            reverse["destroy"]()
        ops["destroy"]()
