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


import numpy as np

from scipy.sparse import coo_matrix

__author__ = "Ilayda Akkor, Alexander V. Dudchenko, Paul Vecchiarelli, Ben Knueven"


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


class HessianApproximation:
    def __init__(self, hessian_type=None):
        if hessian_type is None:
            self.hessian_matrix_type = HessTypes.NoHessian
        else:
            self.hessian_matrix_type = hessian_type

    def hessian_gauss_newton_version(self, sparse_jac, threshold=7):
        """standard gauss newton hessian approximation"""
        hess = np.zeros((len(self.inputs), len(self.inputs)))
        for i in range(self.jacobian_matrix.shape[0]):
            row = self.jacobian_matrix[i, :]
            if sparse_jac:
                row = np.round(row, decimals=threshold)
            hess += np.outer(row.T, row)

        self.hessian_matrix = hess

    def create_bfgs_matrix(self):
        if not hasattr(self, "x"):
            self.x = None
        if not hasattr(self, "hessian_matrix"):
            self.hessian_matrix = []
            for i in range(self.jacobian_matrix.shape[0]):
                self.hessian_matrix.append(np.identity(len(self.inputs)))

    def update_bfgs_matrix(self):
        self.x = self.inputs.copy()
        self.del_f = self.jacobian_matrix.copy()

    def hessian_bfgs(self):
        """Cautious BFGS update implementation (Li and Fukushima)"""
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self.inputs]) - self.x).T
            alpha = 1
            eps = 1e-6
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.hessian_matrix[i] @ s_k
                w = (np.linalg.norm(s_k) ** 2) * (
                    np.linalg.norm(self.del_f[i, :]) ** alpha
                )

                if y_s > eps * w:
                    self.hessian_matrix[i] = (
                        self.hessian_matrix[i]
                        + (y_k @ y_k.T) / (y_s)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )

        self.update_bfgs_matrix()

    def hessian_modified_bfgs(self):
        """Modified BFGS update implementation (Li and Fukushima)"""
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self.inputs]) - self.x).T
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.hessian_matrix[i] @ s_k
                if s_k.any():
                    t_k = 1 + max(0, -y_s / (np.linalg.norm(s_k) ** 2))
                    z_k = y_k + t_k * np.linalg.norm(self.del_f[i, :]) * s_k
                    self.hessian_matrix[i] = (
                        self.hessian_matrix[i]
                        + (z_k @ z_k.T) / (z_k.T @ s_k)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )
        self.update_bfgs_matrix()

    def hessian_damped_bfgs(self):
        """apply Powell's damping on the BFGS update"""
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self.inputs]) - self.x).T
            phi = 0.9
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.hessian_matrix[i] @ s_k

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
                    self.hessian_matrix[i] = (
                        self.hessian_matrix[i]
                        + (z_k @ z_k.T) / (z_s)
                        - (H_s @ H_s.T) / (s_H_s)
                    )
                ###########################################
        self.update_bfgs_matrix()

    def hessian_ipopt_bfgs_modification(self):
        """BFGS update is only done on certain conditions (taken from IPOPT's implementation)"""
        self.create_bfgs_matrix()
        if self.x is not None:
            s_k = (np.array([self.inputs]) - self.x).T
            for i in range(self.jacobian_matrix.shape[0]):
                y_k = np.array([self.jacobian_matrix[i, :] - self.del_f[i, :]]).T
                y_s = y_k.T @ s_k
                H_s = self.hessian_matrix[i] @ s_k
                mach_eps = np.finfo(float).eps
                if (
                    y_s.T
                    > np.sqrt(mach_eps) * np.linalg.norm(s_k) * np.linalg.norm(y_k)
                ) and (np.linalg.norm(s_k, np.inf) >= 100 * mach_eps):
                    self.hessian_matrix[i] = (
                        self.hessian_matrix[i]
                        + (y_k @ y_k.T) / (y_s)
                        - (H_s @ H_s.T) / (s_k.T @ H_s)
                    )
        self.update_bfgs_matrix()

    def hessian_diag_inv_value(self):
        self.hessian_matrix = np.zeros((len(self.inputs), len(self.inputs)))
        for idx, v in enumerate(self.inputs):
            self.hessian_matrix[idx, idx] = 1.0 / v

    def sparse_diagonal(self, shape, value=1e-16):
        rows = []
        cols = []
        vals = []
        for i in range(shape):
            rows.append(i)
            cols.append(i)
            vals.append(value)

        self.hessian_matrix = coo_matrix((vals, (rows, cols)), shape=(shape, shape))

    def get_hessian(self, input_values, output_values, jacobian):
        self.inputs = input_values
        self.outputs = output_values
        self.jacobian_matrix = jacobian
        if self.hessian_matrix_type == HessTypes.NoHessian:
            self.sparse_diagonal(len(self.inputs), 0)
        elif self.hessian_matrix_type == HessTypes.sparse_16:
            self.sparse_diagonal(len(self.inputs), 1e-16)
        elif self.hessian_matrix_type == HessTypes.GaussNewton:
            self.hessian_gauss_newton_version(sparse_jac=False)
        elif self.hessian_matrix_type == HessTypes.BFGS:
            self.hessian_bfgs()
        elif self.hessian_matrix_type == HessTypes.BFGS_mod:
            self.hessian_modified_bfgs()
        elif self.hessian_matrix_type == HessTypes.BFGS_damp:
            self.hessian_damped_bfgs()
        elif self.hessian_matrix_type == HessTypes.BFGS_ipopt:
            self.hessian_ipopt_bfgs_modification()
        elif self.hessian_matrix_type == HessTypes.diag_inv:
            self.hessian_diag_inv_value()
        else:
            raise NotImplementedError(
                f"Hessian type {self.hessian_matrix_type} not implemented"
            )
        return self.hessian_matrix
