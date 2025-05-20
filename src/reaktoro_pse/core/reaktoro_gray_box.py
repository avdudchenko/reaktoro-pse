#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/reaktoro-pse/"
#################################################################################
from cycler import V
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
)
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, tril

import copy
import idaes.logger as idaeslog
from sympy import N
from pyomo.contrib.pynumero.interfaces.utils import (
    make_lower_triangular_full,
    CondensedSparseSummation,
)

__author__ = "Ilayda Akkor, Alexander V. Dudchenko, Paul Vecchiarelli, Ben Knueven"
_log = idaeslog.getLogger(__name__)


class HessTypes:
    GaussNewton = "GaussNewton"
    BFGS = "BFGS"
    BFGS_mod = "BFGS_mod"
    BFGS_damp = "BFGS_damp"
    BFGS_ipopt = "BFGS_ipopt"
    no_hessian = "no_hessian"
    NoHessian = "NoHessian"
    sparse_16 = "sparse_16"
    diag_inv = "diag_inv"


import idaes.core.util.scaling as iscale


class ReaktoroGrayBox(ExternalGreyBoxModel):
    ########################################################################################
    # custom Grey Box functions
    def configure(
        self,
        reaktoro_solver=None,
        inputs=None,
        input_dict=None,
        outputs=None,
        hessian_type=None,
    ):
        # assign a Reaktoro state object to instance
        self.reaktoro_solver = reaktoro_solver
        if hessian_type is None:
            self.hess_type = reaktoro_solver.hessian_type
        else:
            self.hess_type = hessian_type
        if inputs is None:
            self.inputs = reaktoro_solver.input_specs.rkt_inputs.rkt_input_list
        else:
            self.inputs = inputs
        if input_dict is None:
            self.input_dict = reaktoro_solver.input_specs.rkt_inputs
        else:
            self.input_dict = input_dict
        if outputs is None:
            self.outputs = list(reaktoro_solver.output_specs.rkt_outputs.keys())
        else:
            self.outputs = outputs
        self._input_scale = np.ones(len(self.inputs))
        self._outputs_dual_multipliers = np.ones(len(self.outputs))
        self.header_saved = False
        self.step = 0
        self.old_params = None

        _log.info(f"RKT gray box using {self.hess_type} hessian type")
        if self.hess_type != HessTypes.no_hessian:
            setattr(self, "evaluate_hessian_outputs", self._evaluate_hessian_outputs)

    ########################################################################################
    # standard Grey Box functions
    def input_names(self):
        # get input names (required by Grey Box)
        return self.inputs

    def output_names(self):
        # get output names (not required, but helpful)
        return self.outputs

    def set_input_values(self, input_values):
        self._input_scale = self.reaktoro_solver.get_input_scaling()
        # set input values from Pyomo as inputs to External Model (required by Grey Box)
        # self._scaled_input_values = list(np.array(input_values) * self._input_scale)
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        # initialize Pyomo block for External Model
        block_components = [obj for obj in pyomo_block.component_objects(pyo.Var)]
        for block in block_components:
            # 1e-16 is Reaktoro's epsilon value
            if "inputs" in block.name:
                for var in self.inputs:
                    block[var].value = 1  # self.input_dict[var].get_pyomo_var_value()
                    block[var].setlb(self.input_dict[var].get_lower_bound())
                    block[var].setub(None)
            elif "outputs" in block.name:
                for prop in self.outputs:
                    block[prop].setlb(None)
                    block[prop].setub(None)
                    block[prop].value = 0.1

    def evaluate_outputs(self):
        # update Reaktoro state with current inputs (this function runs repeatedly)
        self.params = dict(zip(self.inputs, self._input_values))

        self.get_last_output(self.params)

        return np.array(self.rkt_result, dtype=np.float64)

    def get_last_output(self, new_params):
        """only eval reaktoro if params changed!"""
        if self.old_params is None or any(
            new_params[key] != self.old_params[key] for key in new_params
        ):
            self.jacobian_matrix, self.rkt_result = (
                self.reaktoro_solver.solve_reaktoro_block(params=new_params)
            )
            self.step += 1
        self.old_params = copy.deepcopy(new_params)

    def evaluate_jacobian_outputs(self):
        self.evaluate_outputs()
        jm = np.array(self.jacobian_matrix)
        i = np.array([i for i in range(jm.shape[0]) for j in range(jm.shape[1])])
        j = np.array([j for i in range(jm.shape[0]) for j in range(jm.shape[1])])

        cm = coo_matrix((jm.flatten(), (i, j)))
        return cm

    def set_output_constraint_multipliers(self, _outputs_dual_multipliers):
        np.copyto(self._outputs_dual_multipliers, _outputs_dual_multipliers)

    def get_output_constraint_scaling_factors(self):
        return self.reaktoro_solver.get_jacobian_scaling()

    def hessian_gauss_newton_version(self, sparse_jac, threshold=7):

        s = np.zeros((len(self.inputs), len(self.inputs)))
        for i in range(self.jacobian_matrix.shape[0]):
            row = self.jacobian_matrix[i, :]
            if sparse_jac:
                row = np.round(row, decimals=threshold)
            s += self._outputs_dual_multipliers[i] * np.outer(row.T, row)
        hess = s
        return hess

    def update_bfgs_matrix(self):
        h_sum = np.zeros((len(self.inputs), len(self.inputs)))
        for i in range(self.jacobian_matrix.shape[0]):
            h_sum += self.H[i] * self._outputs_dual_multipliers[i]

        self.x = self._input_values.copy()
        self.del_f = self.jacobian_matrix.copy()
        return h_sum

    def create_bfgs_matrix(self):
        if not hasattr(self, "x"):
            self.x = None
        if not hasattr(self, "H"):
            self.H = []
            for i in range(self.jacobian_matrix.shape[0]):
                self.H.append(np.identity(len(self.inputs)))

    def hessian_bfgs(self):
        # Cautious BFGS update implementation (Li and Fukushima)
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self._input_values]) - self.x).T
            alpha = 1
            eps = 1e-6
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.H[i] @ s_k
                w = (np.linalg.norm(s_k) ** 2) * (
                    np.linalg.norm(self.del_f[i, :]) ** alpha
                )

                if y_s > eps * w:
                    self.H[i] = (
                        self.H[i]
                        + (y_k @ y_k.T) / (y_s)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )

        return self.update_bfgs_matrix()

    def hessian_modified_bfgs(self):
        # Modified BFGS update implementation (Li and Fukushima)
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self._input_values]) - self.x).T
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.H[i] @ s_k
                if s_k.any():
                    t_k = 1 + max(0, -y_s / (np.linalg.norm(s_k) ** 2))
                    z_k = y_k + t_k * np.linalg.norm(self.del_f[i, :]) * s_k
                    self.H[i] = (
                        self.H[i]
                        + (z_k @ z_k.T) / (z_k.T @ s_k)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )

        return self.update_bfgs_matrix()

    def hessian_damped_bfgs(self):
        # apply Powell's damping on the BFGS update
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self._input_values]) - self.x).T
            phi = 0.9
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.H[i] @ s_k

                # new
                s_H_s = s_k.T @ H_s
                if y_s >= phi * s_H_s:
                    delta_k = 1
                else:
                    delta_k = (1 - phi) * s_H_s / (s_H_s - y_s)
                z_k = delta_k * y_k + (1 - delta_k) * H_s
                z_s = z_k.T @ s_k
                if z_k.shape != y_k.shape:
                    raise RuntimeError()
                if s_k.any():  # extra
                    self.H[i] = (
                        self.H[i] + (z_k @ z_k.T) / (z_s) - (H_s @ H_s.T) / (s_H_s)
                    )
                ###########################################
        return self.update_bfgs_matrix()

    def hessian_ipopt_bfgs_modification(self):
        # BFGS update is only done on certain conditions (taken from IPOPT's implementation)
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self._input_values]) - self.x).T
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.H[i] @ s_k
                mach_eps = np.finfo(float).eps
                if (
                    y_s.T
                    > np.sqrt(mach_eps) * np.linalg.norm(s_k) * np.linalg.norm(y_k)
                ) and (np.linalg.norm(s_k, np.inf) >= 100 * mach_eps):
                    self.H[i] = (
                        self.H[i]
                        + (y_k @ y_k.T) / (y_s)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )
        return self.update_bfgs_matrix()

    def hessian_diag_inv_value(self):
        hessian = np.zeros((len(self.inputs), len(self.inputs)))
        for idx, v in enumerate(self._input_values):
            hessian[idx, idx] = 1.0 / v
        h_sum = np.zeros((len(self.inputs), len(self.inputs)))
        for i in range(self.jacobian_matrix.shape[0]):
            h_sum += self._outputs_dual_multipliers[i] * hessian[i]
        return h_sum

    def _evaluate_hessian_outputs(self):
        if self.hess_type == HessTypes.NoHessian:
            hessian_matrix = _sparse_diagonal(len(self.inputs), 0)
        elif self.hess_type == HessTypes.sparse_16:
            hessian_matrix = _sparse_diagonal(len(self.inputs), 1e-16)
        elif self.hess_type == HessTypes.GaussNewton:
            hessian_matrix = self.hessian_gauss_newton_version(sparse_jac=False)
        elif self.hess_type == HessTypes.BFGS:
            hessian_matrix = self.hessian_bfgs()
        elif self.hess_type == HessTypes.BFGS_mod:
            hessian_matrix = self.hessian_modified_bfgs()
        elif self.hess_type == HessTypes.BFGS_damp:
            hessian_matrix = self.hessian_damped_bfgs()
        elif self.hess_type == HessTypes.BFGS_ipopt:
            hessian_matrix = self.hessian_ipopt_bfgs_modification()
        elif self.hess_type == HessTypes.diag_inv:
            hessian_matrix = self.hessian_diag_inv_value()
        else:
            raise RuntimeError(
                f"Hessian type {self.hess_type} not implemented in ReaktoroGrayBox"
            )
        if isinstance(hessian_matrix, coo_matrix):
            return hessian_matrix
        else:
            low_triangular_hessian = _hand_tril(np.array(hessian_matrix))
            return low_triangular_hessian


def _hand_tril(jm):
    assert jm.shape[0] == jm.shape[1]
    shape = jm.shape[0]
    row = []
    col = []
    val = []

    for i in range(shape):
        for j in range(i + 1):
            row.append(i)
            col.append(j)
            v = jm[i, j]

            val.append(v)

    return coo_matrix((val, (row, col)), shape=(shape, shape))


def _sparse_diagonal(shape, value=1e-16):
    rows = []
    cols = []
    vals = []
    for i in range(shape):
        rows.append(i)
        cols.append(i)
        vals.append(value)

    return coo_matrix((vals, (rows, cols)), shape=(shape, shape))
