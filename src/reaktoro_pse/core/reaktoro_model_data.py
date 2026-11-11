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
"""Build independent Reaktoro solvers from exported configurations.

Used by parallel workers and to build "shadow" solvers (e.g. for finite difference
hessians), so perturbed solves never touch the warm start of the main solver.
"""
from reaktoro_pse.core.reaktoro_state import ReaktoroState
from reaktoro_pse.core.reaktoro_coupled_solver import ReaktoroCoupledSolver
from reaktoro_pse.core.reaktoro_jacobian import ReaktoroJacobianSpec
from reaktoro_pse.core.reaktoro_outputs import ReaktoroOutputSpec
from reaktoro_pse.core.reaktoro_inputs import ReaktoroInputSpec
from reaktoro_pse.core.reaktoro_solver import ReaktoroSolver

__author__ = "Alexander V. Dudchenko"

PROPERTY_BLOCK_KEY = "property_block"


class RktModelData:
    def __init__(self, config_data):
        """Initialize the Reaktoro state with the provided configuration data."""

        self.build_state(config_data)

    def build_state(self, config_data):
        state_config, input_config, output_config, jacobian_config, solver_config = (
            config_data
        )
        self.state = ReaktoroState()
        self.state.load_from_export_object(state_config)
        self.state.build_state()
        self.inputs = ReaktoroInputSpec(self.state)
        self.inputs.load_from_export_object(input_config)
        self.inputs.build_input_specs()
        self.outputs = ReaktoroOutputSpec(self.state)
        self.outputs.load_from_export_object(output_config)
        self.jacobian = ReaktoroJacobianSpec(self.state, self.outputs)
        self.jacobian.load_from_export_object(jacobian_config)
        self.solver = ReaktoroSolver(
            self.state, self.inputs, self.outputs, self.jacobian
        )
        self.solver.load_from_export_object(solver_config)


def export_solver_configs(solver):
    """configuration data for a ReaktoroSolver (list) or ReaktoroCoupledSolver (dict),
    in the format used by build_solver_from_configs"""
    if isinstance(solver, ReaktoroCoupledSolver):
        configs = {
            f"spc_blk_{i}": export_solver_configs(spc)
            for i, spc in enumerate(solver.speciation_solvers)
        }
        configs[PROPERTY_BLOCK_KEY] = export_solver_configs(solver.property_solver)
        return configs
    return [
        solver.state.export_config(),
        solver.input_specs.export_config(),
        solver.output_specs.export_config(),
        solver.jacobian_specs.export_config(),
        solver.export_config(),
    ]


def build_solver_from_configs(config_data):
    """build solver from export data

    Returns:
        (solver, model_data) where model_data is a RktModelData (list config) or a dict
        of RktModelData (coupled config, keyed as in config_data)
    """
    if isinstance(config_data, list):
        model = RktModelData(config_data)
        return model.solver, model
    elif isinstance(config_data, dict):
        models = {}
        spc_solvers = []
        for key in config_data:
            models[key] = RktModelData(config_data[key])
            if key != PROPERTY_BLOCK_KEY:
                spc_solvers.append(models[key].solver)
        solver = ReaktoroCoupledSolver(spc_solvers)
        solver.register_property_solver(models[PROPERTY_BLOCK_KEY].solver)
        for obj in models[PROPERTY_BLOCK_KEY].solver.input_specs.rkt_inputs.values():
            if obj.dummy_var_key is not None:
                obj.dummy_var = solver.outputs[obj.dummy_var_key]
        return solver, models
    raise TypeError(
        "config_data must be a list or a dictionary containing the configuration data"
    )


def solver_states(solver):
    """ReaktoroState objects of a solver, in a fixed order"""
    if isinstance(solver, ReaktoroCoupledSolver):
        states = []
        for spc in solver.speciation_solvers:
            states.extend(solver_states(spc))
        states.extend(solver_states(solver.property_solver))
        return states
    return [solver.state]


def reset_failure_counters(solver):
    if isinstance(solver, ReaktoroCoupledSolver):
        for spc in solver.speciation_solvers:
            reset_failure_counters(spc)
        reset_failure_counters(solver.property_solver)
    else:
        solver._sequential_fails = 0
