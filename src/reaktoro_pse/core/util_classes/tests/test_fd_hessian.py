#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/reaktoro-pse/"
#################################################################################
import numpy as np
import pytest
import cyipopt
from scipy.sparse import coo_matrix

from reaktoro_pse.core.util_classes.fd_hessian import (
    JacobianFiniteDifference,
    compute_fd_steps,
)
from reaktoro_pse.core.util_classes.hessian_functions import (
    HessianApproximation,
    HessTypes,
    HESSIAN_DEFAULTS,
)
from reaktoro_pse.core.tests.test_reaktoro_state import build_rkt_state_with_species
from reaktoro_pse.core.tests.test_reaktoro_solver import build_standard_state

# quadratic outputs y_i = 0.5 x'A_i x + b_i'x, jacobian row i = (A_i x + b_i)'
A = [
    np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 3.0]]),
    np.array([[1.0, -2.0, 0.0], [-2.0, -1.0, 0.0], [0.0, 0.0, 0.5]]),  # indefinite
]
B = [np.array([1.0, 0.0, -1.0]), np.array([0.0, 2.0, 0.0])]
X0 = np.array([1.0, 2.0, 0.5])
LAM = np.array([0.7, -1.3])


def quad_jac(x):
    return np.array([Ai @ x + bi for Ai, bi in zip(A, B)])


class MockSolver:
    """counts solves, optionally fails at given perturbed points"""

    def __init__(self, fail=None):
        self.solves = 0
        self.prepared = 0
        self.fail = fail or (lambda x: False)

    def solve(self, x):
        self.solves += 1
        if self.fail(x):
            raise cyipopt.CyIpoptEvaluationError
        return quad_jac(x)

    def prepare(self):
        self.prepared += 1


def make_fd(mock, central=False):
    return JacobianFiniteDifference(mock.solve, mock.prepare, central=central)


def exact_hessian(lam):
    return sum(l * Ai for l, Ai in zip(lam, A))


@pytest.mark.parametrize("central", [False, True])
def test_linear_jacobian_exact(central):
    mock = MockSolver()
    fd = make_fd(mock, central)
    steps = compute_fd_steps(X0, 1e-4)
    H, ok = fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM, steps)
    assert ok.all()
    np.testing.assert_allclose(H, exact_hessian(LAM), rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(H, H.T)
    assert fd.last_asymmetry < 1e-6
    assert mock.solves == (6 if central else 3)
    # shadow solver is prepared (warm started) before every perturbed solve
    assert mock.prepared == mock.solves
    # per output hessians
    Hi = fd.output_hessians()
    for i in range(2):
        np.testing.assert_allclose(Hi[i], A[i], rtol=1e-6, atol=1e-8)


def test_new_multipliers_reuse_cache():
    mock = MockSolver()
    fd = make_fd(mock)
    steps = compute_fd_steps(X0, 1e-4)
    fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM, steps)
    solves = mock.solves
    lam2 = np.array([-3.0, 0.25])
    H, ok = fd.evaluate_fd_hessian(X0, quad_jac(X0), lam2, steps)
    assert mock.solves == solves
    np.testing.assert_allclose(H, exact_hessian(lam2), rtol=1e-6, atol=1e-8)
    # new point re-solves
    x1 = X0 + 0.1
    fd.evaluate_fd_hessian(x1, quad_jac(x1), lam2, steps)
    assert mock.solves == solves + 3


def test_failed_forward_step_uses_backward():
    # fail any point where input 1 is increased
    mock = MockSolver(fail=lambda x: x[1] > X0[1])
    fd = make_fd(mock)
    steps = compute_fd_steps(X0, 1e-4)
    H, ok = fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM, steps)
    assert ok.all()
    np.testing.assert_allclose(H, exact_hessian(LAM), rtol=1e-6, atol=1e-8)
    assert mock.solves == 4
    assert fd.number_of_failed_solves == 1


def test_failed_both_directions_reports_not_ok():
    mock = MockSolver(fail=lambda x: x[2] != X0[2])
    fd = make_fd(mock)
    steps = compute_fd_steps(X0, 1e-4)
    H, ok = fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM, steps)
    assert not ok.any()
    # failure is cached too, new multipliers do not retry
    solves = mock.solves
    fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM * 2, steps)
    assert mock.solves == solves


def test_bounds_select_direction():
    # input 0 at upper bound -> only backward step, input 2 at lower bound -> forward
    mock = MockSolver()
    fd = make_fd(mock, central=True)
    lower = np.array([-np.inf, -np.inf, X0[2]])
    upper = np.array([X0[0], np.inf, np.inf])
    steps = compute_fd_steps(X0, 1e-4, lower=lower, upper=upper)
    seen = []
    orig = mock.solve

    def solve(x):
        seen.append(x.copy())
        return orig(x)

    fd.solve_function = solve
    H, ok = fd.evaluate_fd_hessian(X0, quad_jac(X0), LAM, steps, lower, upper)
    assert ok.all()
    assert all(x[0] <= upper[0] and x[2] >= lower[2] for x in seen)
    np.testing.assert_allclose(H, exact_hessian(LAM), rtol=1e-6, atol=1e-8)


def test_compute_fd_steps():
    x = np.array([300.0, 1e-5, 0.0, 2.0])
    typ = np.array([np.nan, 1e-3, np.nan, np.nan])
    h = compute_fd_steps(x, 1e-4, typ)
    np.testing.assert_allclose(h, [3e-2, 1e-7, 1e-4, 2e-4])
    # box narrower than step: step shrinks to fit larger side
    h = compute_fd_steps(
        np.array([1.0]), 0.1, lower=np.array([0.99]), upper=np.array([1.02])
    )
    np.testing.assert_allclose(h, [0.02])


def test_fd_type_and_defaults():
    assert HessTypes.FD_jacobian == "FD_jacobian"
    for key in ("fd_hessian_step", "fd_hessian_central", "fd_hessian_fallback_type"):
        assert key in HESSIAN_DEFAULTS
    from reaktoro_pse.reaktoro_block_config.hessian_options import HessianOptions

    cfg = HessianOptions().get_dict()()
    for key in ("fd_hessian_step", "fd_hessian_central", "fd_hessian_fallback_type"):
        assert cfg[key] == HESSIAN_DEFAULTS[key]


def test_overlay_keeps_pattern():
    """gray box overlays fd values on fallback hessian pattern"""
    from reaktoro_pse.core.reaktoro_gray_box import ReaktoroGrayBox

    support = [np.array([0, 1]), np.array([2])]
    qn = HessianApproximation(hessian_type=HessTypes.SR1, hessian_support=support)
    jac = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    fallback = qn.get_hessian(X0, jac, LAM)
    gb = ReaktoroGrayBox.__new__(ReaktoroGrayBox)
    H = np.arange(9, dtype=float).reshape(3, 3)
    H = H + H.T

    class Provider:
        def __init__(self, ok):
            self.ok = ok

        def evaluate_fd_hessian(self, *args):
            return H, self.ok

    gb._input_values = list(X0)
    gb._input_vars = None
    gb.inputs = ["a", "b", "c"]
    gb.fd_hessian_step = 1e-4
    gb.jacobian_matrix = jac
    gb._outputs_dual_multipliers = LAM
    gb.fd_provider = Provider(np.array([True, True, False]))
    out = gb._fd_hessian(fallback)
    np.testing.assert_array_equal(out.row, fallback.row)
    np.testing.assert_array_equal(out.col, fallback.col)
    for r, c, v, v_qn in zip(out.row, out.col, out.data, fallback.data):
        if r < 2 and c < 2:
            assert v == H[r, c]
        else:
            assert v == v_qn
    # no cross block entries in pattern
    assert not any((r < 2) != (c < 2) for r, c in zip(out.row, out.col))


###############################################################################
# real reaktoro


def _solver_and_point(solver):
    keys = solver.input_specs.rkt_inputs.rkt_input_list
    solver.solve_reaktoro_block()
    x0 = np.array(
        [solver.input_specs.rkt_inputs[k].get_temp_value() for k in keys], dtype=float
    )
    jac, y = solver.solve_reaktoro_block(params=dict(zip(keys, x0)))
    return keys, x0, np.array(jac).copy(), np.array(y).copy()


def test_reaktoro_fd_hessian_vs_output_second_differences(build_standard_state):
    solver = build_standard_state
    keys, x0, J0, y0 = _solver_and_point(solver)
    names = list(solver.output_specs.rkt_outputs.keys())
    # outputs with curvature (pH and H2O are linear in inputs for this system)
    out_idx = [names.index(("speciesAmount", "CO2")), names.index(("speciesAmount", "Ca+2"))]
    lam = np.zeros(len(y0))
    lam[out_idx] = [1.0, -0.5]
    steps = compute_fd_steps(x0, 1e-4)
    for central, tol in [(False, 1e-3), (True, 1e-5)]:
        fd = JacobianFiniteDifference.from_solver(solver, keys, central=central)
        H, ok = fd.evaluate_fd_hessian(x0, J0, lam, steps)
        assert ok.all()
        assert fd.last_asymmetry < tol

    # second differences of the outputs (no jacobian involved)
    def out(x):
        _, y = solver.solve_reaktoro_block(params=dict(zip(keys, x)))
        return np.array(y)[out_idx] @ lam[out_idx]

    for k in [keys.index("HCO3-"), keys.index("CO2"), keys.index("Ca+2")]:
        h = 1e-3 * abs(x0[k])
        xp, xm = x0.copy(), x0.copy()
        xp[k] += h
        xm[k] -= h
        d2 = (out(xp) - 2 * out(x0) + out(xm)) / h**2
        assert H[k, k] == pytest.approx(d2, rel=2e-2)


def test_reaktoro_fd_does_not_touch_main_solver(build_standard_state):
    """perturbed solves run on a shadow solver, main solver and its reaktoro warm
    start stay bit identical"""
    solver = build_standard_state
    keys, x0, J0, y0 = _solver_and_point(solver)
    objs = [solver.input_specs.rkt_inputs[k] for k in keys]
    temps = [o.temp_value for o in objs]
    amounts = np.array(solver.state.state.speciesAmounts()).copy()
    fd = JacobianFiniteDifference.from_solver(solver, keys)
    H, ok = fd.evaluate_fd_hessian(
        x0, J0, np.ones(len(y0)), compute_fd_steps(x0, 1e-4)
    )
    assert ok.all()
    assert fd.shadow["solver"] is not solver
    np.testing.assert_array_equal(np.array(solver.state.state.speciesAmounts()), amounts)
    np.testing.assert_array_equal(np.array(solver.jacobian_matrix), J0)
    np.testing.assert_array_equal(np.array(solver.outputs), y0)
    assert [o.temp_value for o in objs] == temps
    # shadow jacobian at the base point agrees with main jacobian
    J_shadow = fd.solve_function(x0)
    np.testing.assert_allclose(J_shadow, J0, rtol=1e-5, atol=1e-10)
