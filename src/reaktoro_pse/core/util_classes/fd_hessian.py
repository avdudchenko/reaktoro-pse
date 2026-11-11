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
"""Hessian of the gray box Lagrangian from finite differences of the Jacobian.

Reaktoro provides an (analytic) Jacobian J(x) = dy/dx of the gray box outputs.
Perturbing each input once gives the third order tensor

    T[k] = (J(x + h_k e_k) - J(x)) / h_k  ~  dJ/dx_k,   T[k][i, j] ~ d2y_i/dx_j dx_k

and the Hessian of sum_i lambda_i y_i is H = sym(G), with column k of G = T[k]^T lambda.

The tensor only depends on x, so it is cached: a call with new multipliers at the
same point only re-contracts and needs no new solves.

Perturbed solves run on a separate "shadow" solver built from the exported
configuration of the main solver (same as parallel workers), so the main solver and
its Reaktoro warm start are never touched. This matters: the converged Reaktoro state
depends (within tolerance) on the warm start, and for poorly determined outputs
(e.g. pE of a system without redox buffer) running perturbed solves on the main
solver visibly changes the solution path. The main state can not be restored either,
ChemicalState.assign() is not safe (the next solve can stop after one iteration with
values of the assigned state) and restoring species amounts does not restore the
equilibrium variables. Before each perturbed solve the shadow state is warm started
from the species amounts, temperature and pressure of the main state.
"""
import numpy as np
import cyipopt
import idaes.logger as idaeslog

__author__ = "Alexander V. Dudchenko"
_log = idaeslog.getLogger(__name__)

def compute_fd_steps(x, relative_step, typical_values=None, lower=None, upper=None):
    """step size per input, h_k = relative_step * max(|x_k|, typical_k)

    Args:
        x: current inputs
        relative_step: relative step size
        typical_values: typical magnitude of each input (e.g. 1/scaling factor),
            None or nan entries use |x_k|
        lower, upper: bounds (None/nan/inf for none), used to make sure a step in at
            least one direction stays inside the bounds
    Returns:
        array of positive step sizes
    """
    x = np.asarray(x, dtype=float)
    mag = np.abs(x)
    if typical_values is not None:
        typ = np.asarray(typical_values, dtype=float)
        typ = np.where(np.isfinite(typ), np.abs(typ), 0.0)
        mag = np.maximum(mag, typ)
    h = relative_step * mag
    h[h == 0] = relative_step
    lb, ub = _bounds(len(x), lower, upper)
    # if box is narrower than a step in both directions, shrink step to fit
    room = np.maximum(ub - x, x - lb)
    shrink = np.isfinite(room) & (room < h) & (room > 0)
    h[shrink] = room[shrink]
    return h


def _bounds(n, lower, upper):
    lb = np.full(n, -np.inf) if lower is None else np.array(lower, dtype=float)
    ub = np.full(n, np.inf) if upper is None else np.array(upper, dtype=float)
    lb[~np.isfinite(lb)] = -np.inf
    ub[~np.isfinite(ub)] = np.inf
    return lb, ub


class JacobianFiniteDifference:
    """finite difference hessian from a jacobian function

    Args:
        solve_function: f(x) -> jacobian (n_outputs x n_inputs) at x, raises on failure
        prepare_function: called before each perturbed solve (e.g. warm start of the
            shadow solver from the main solver, reset of failure counters)
        central: use central differences where both steps are inside the bounds
    """

    def __init__(
        self,
        solve_function,
        prepare_function=None,
        central=False,
    ):
        self.solve_function = solve_function
        self.prepare_function = prepare_function
        self.central = central
        # solve at x and use it as base for one sided differences (instead of the
        # jacobian passed in), set for shadow solvers
        self.solve_base_point = False
        self._base = None
        self._cache_key = None
        self._tensor = None
        self._ok = None
        self.number_of_solves = 0
        self.number_of_failed_solves = 0
        self.number_of_evaluations = 0
        self.last_asymmetry = None

    @classmethod
    def from_solver(cls, solver, input_keys, central=False, config_data=None):
        """build for a ReaktoroSolver or ReaktoroCoupledSolver, perturbed solves run on
        a shadow solver built (on first use) from config_data, or from configs exported
        from solver if not provided"""
        # import here, reaktoro_model_data imports solver modules
        from reaktoro_pse.core.reaktoro_model_data import (
            build_solver_from_configs,
            export_solver_configs,
            solver_states,
            reset_failure_counters,
        )

        keys = list(input_keys)
        shadow = {}

        def get_shadow():
            if not shadow:
                configs = config_data
                if configs is None:
                    configs = export_solver_configs(solver)
                shadow["solver"], shadow["models"] = build_solver_from_configs(configs)
                # same as worker initialization, also required by coupled solver
                shadow["solver"].equilibrate_state()
                shadow["pairs"] = list(
                    zip(solver_states(solver), solver_states(shadow["solver"]))
                )
                _log.info("built shadow reaktoro solver for fd hessian")
            return shadow

        def prepare():
            sh = get_shadow()
            for main_state, shadow_state in sh["pairs"]:
                ms, ss = main_state.state, shadow_state.state
                ss.setTemperature(ms.temperature())
                ss.setPressure(ms.pressure())
                ss.setSpeciesAmounts(ms.speciesAmounts())
            reset_failure_counters(sh["solver"])

        def solve(x):
            jac, _ = get_shadow()["solver"].solve_reaktoro_block(
                params=dict(zip(keys, x))
            )
            return np.array(jac, dtype=float, copy=True)

        fd = cls(solve, prepare_function=prepare, central=central)
        fd.shadow = shadow
        fd.solve_base_point = True
        return fd

    def _try_solve(self, x):
        if self.prepare_function is not None:
            self.prepare_function()
        self.number_of_solves += 1
        try:
            jac = self.solve_function(x)
        except (cyipopt.CyIpoptEvaluationError, RuntimeError) as e:
            self.number_of_failed_solves += 1
            _log.debug(f"fd hessian perturbed solve failed: {e!r}")
            return None
        if not np.all(np.isfinite(jac)):
            self.number_of_failed_solves += 1
            return None
        return jac

    def _base_jacobian(self, x, jac0):
        """jacobian one sided differences are taken against. With a shadow solver
        this is the shadow jacobian at x: the shadow and main solver converge to
        slightly different states (warm start), and their difference divided by
        the step would swamp the curvature"""
        if not self.solve_base_point:
            return jac0
        if self._base is None:
            self._base = self._try_solve(x)
            if self._base is None:
                self._base = False
        return None if self._base is False else self._base

    def _column(self, x, jac0, k, h, lb, ub):
        """dJ/dx_k, None if all perturbed solves failed"""
        fwd_ok = x[k] + h <= ub[k]
        bwd_ok = x[k] - h >= lb[k]
        if self.central and fwd_ok and bwd_ok:
            jp = self._try_solve(_perturb(x, k, h))
            jm = self._try_solve(_perturb(x, k, -h))
            if jp is not None and jm is not None:
                return (jp - jm) / (2 * h)
            base = self._base_jacobian(x, jac0)
            if base is None:
                return None
            if jp is not None:
                return (jp - base) / h
            if jm is not None:
                return (base - jm) / h
            return None
        # forward difference, opposite direction if outside bounds or failed
        directions = [h, -h] if fwd_ok else [-h, h]
        for d in directions:
            if (d > 0 and not fwd_ok) or (d < 0 and not bwd_ok):
                continue
            base = self._base_jacobian(x, jac0)
            if base is None:
                return None
            jp = self._try_solve(_perturb(x, k, d))
            if jp is not None:
                return (jp - base) / d
        return None

    def compute_tensor(self, x, jac0, steps, lower=None, upper=None):
        """compute (or reuse cached) dJ/dx_k for all k at x

        Returns:
            True if every column succeeded
        """
        x = np.array(x, dtype=float)
        jac0 = np.array(jac0, dtype=float)
        key = (x.tobytes(), jac0.tobytes())
        if key == self._cache_key:
            return self._ok
        n_out, n_in = jac0.shape
        lb, ub = _bounds(n_in, lower, upper)
        steps = np.asarray(steps, dtype=float)
        tensor = np.zeros((n_in, n_out, n_in))
        ok = True
        self._base = None
        for k in range(n_in):
            col = self._column(x, jac0, k, steps[k], lb, ub)
            if col is None:
                ok = False
                _log.warning(
                    f"fd hessian: perturbed solves failed for input {k}, "
                    "using fallback hessian for this point"
                )
                break
            tensor[k] = col
        self._cache_key = key
        self._steps = steps
        self._tensor = tensor if ok else None
        self._ok = ok
        return ok

    def hessian(self, multipliers):
        """symmetric hessian of sum_i lambda_i y_i from cached tensor (None if failed)"""
        if not self._ok:
            return None
        lam = np.asarray(multipliers, dtype=float)
        # G[j, k] = sum_i lambda_i T[k][i, j]
        G = np.einsum("kij,i->jk", self._tensor, lam)
        # asymmetry in inputs scaled by step size (~ typical magnitude), otherwise
        # inputs with large units (e.g. pressure) dominate the norm
        Gs = G * np.outer(self._steps, self._steps)
        norm = np.linalg.norm(Gs)
        self.last_asymmetry = np.linalg.norm(Gs - Gs.T) / norm if norm > 0 else 0.0
        return 0.5 * (G + G.T)

    def output_hessians(self):
        """symmetric hessian of each output, shape (n_out, n_in, n_in), None if failed"""
        if not self._ok:
            return None
        Hi = np.transpose(self._tensor, (1, 2, 0))  # [i, j, k]
        return 0.5 * (Hi + np.transpose(Hi, (0, 2, 1)))

    def evaluate_fd_hessian(self, x, jac0, multipliers, steps, lower=None, upper=None):
        """hessian of sum_i lambda_i y_i at x

        Returns:
            (H, ok) where H is dense (n_in x n_in) and ok is a boolean array over inputs,
            False where the fallback hessian has to be used
        """
        self.number_of_evaluations += 1
        n_in = np.shape(jac0)[1]
        if not self.compute_tensor(x, jac0, steps, lower, upper):
            return np.zeros((n_in, n_in)), np.zeros(n_in, dtype=bool)
        H = self.hessian(multipliers)
        if self.last_asymmetry > 1e-2:
            _log.debug(
                f"fd hessian asymmetry {self.last_asymmetry:.2e}, "
                "jacobian may be noisy or a phase boundary was crossed"
            )
        return H, np.ones(n_in, dtype=bool)


def _perturb(x, k, h):
    xp = x.copy()
    xp[k] += h
    return xp
