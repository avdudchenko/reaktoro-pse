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

from calendar import c
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Pipe
from reaktoro_pse.core.reaktoro_state import (
    ReaktoroState,
)
from reaktoro_pse.core.reaktoro_jacobian import (
    ReaktoroJacobianSpec,
)
from reaktoro_pse.core.reaktoro_outputs import (
    ReaktoroOutputSpec,
)

from reaktoro_pse.core.reaktoro_inputs import (
    ReaktoroInputSpec,
)
from reaktoro_pse.core.reaktoro_solver import (
    ReaktoroSolver,
)
import numpy as np
import time

import cyipopt
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


class RemoteWorker:
    def __init__(
        self,
        config_data,
        input_reference,
        output_reference,
        jacobian_reference,
    ):
        """build remote instance of reaktoro state and solver for execution in
        its own process"""
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
        self.param_dict = {}
        self.input_reference = shared_memory.SharedMemory(name=input_reference)
        self.output_reference = shared_memory.SharedMemory(name=output_reference)
        self.jacobian_reference = shared_memory.SharedMemory(name=jacobian_reference)
        self.input_matrix = np.ndarray(
            (3, len(self.inputs.rkt_inputs.keys())),
            dtype=float,
            buffer=self.input_reference.buf,
        )
        self.jacobian_matrix = np.ndarray(
            (
                len(self.outputs.rkt_outputs.keys()),
                len(self.inputs.rkt_inputs.keys()),
            ),
            dtype=float,
            buffer=self.jacobian_reference.buf,
        )

        self.output_matrix = np.ndarray(
            len(self.outputs.rkt_outputs.keys()),
            dtype=float,
            buffer=self.output_reference.buf,
        )
        self.params = {}

    def initialize(self, presolve=False):
        _log.info("Initialized in remote worker")

        self.update_inputs()
        self.state.equilibrate_state()
        jacobian, outputs = self.solver.solve_reaktoro_block(presolve=presolve)
        self.update_output_matrix(outputs, jacobian)

    def solve(self):
        self.get_params()
        try:
            jacobian, outputs = self.solver.solve_reaktoro_block(self.params)
            self.update_output_matrix(outputs, jacobian)
            return WorkerMessages.success
        except cyipopt.CyIpoptEvaluationError:
            return WorkerMessages.CyIpoptEvaluationError

    def update_output_matrix(self, outputs, jacobian):
        np.copyto(self.output_matrix, outputs)
        np.copyto(self.jacobian_matrix, jacobian)

    def get_params(self):
        for i, key in enumerate(self.inputs.rkt_inputs.keys()):
            self.params[key] = self.input_matrix[2][i]

    def update_inputs(self):
        for i, key in enumerate(self.inputs.rkt_inputs.keys()):
            self.inputs.rkt_inputs[key].value = self.input_matrix[0][i]
            self.inputs.rkt_inputs[key].converted_value = self.input_matrix[1][i]


class WorkerMessages:
    initialize = "initialize"
    update_values = "update_values"
    solve = "solve"
    success = "success"
    failed_solve = "failed_solve"
    CyIpoptEvaluationError = "CyIpoptEvaluationError"
    terminate = "terminate"


class LocalWorker:
    def __init__(self, worker_data):
        """defines local instance of worker to provide
        direct access to function execution"""
        self.worker_data = worker_data
        self.local_pipe, self.remote_pipe = Pipe()
        self.get_input_and_output_sizes()

    def get_input_and_output_sizes(self):
        # for storing raw and converted values
        # index 0 for values for init
        # index 1 for converted values for init
        # index 3 for ipopt solver calls
        self.input_keys = self.worker_data.inputs.rkt_inputs.keys()
        input_matrix = np.zeros(
            (3, len(self.worker_data.inputs.rkt_inputs.keys())), dtype=float
        )
        self.input_reference = shared_memory.SharedMemory(
            create=True, size=input_matrix.nbytes
        )
        self.input_matrix = np.ndarray(
            (3, len(self.worker_data.inputs.rkt_inputs.keys())),
            dtype=float,
            buffer=self.input_reference.buf,
        )
        # for storing output matrix and jacobian
        jacobian_matrix = np.zeros(
            (
                len(self.worker_data.outputs.rkt_outputs.keys()),
                len(self.input_keys),
            ),
            dtype=float,
        )
        output_matrix = np.zeros(len(self.worker_data.outputs.rkt_outputs.keys()))
        self.jacobian_reference = shared_memory.SharedMemory(
            create=True, size=jacobian_matrix.nbytes
        )
        self.jacobian_matrix = np.ndarray(
            (
                len(self.worker_data.outputs.rkt_outputs.keys()),
                len(self.input_keys),
            ),
            dtype=float,
            buffer=self.jacobian_reference.buf,
        )
        self.output_reference = shared_memory.SharedMemory(
            create=True, size=output_matrix.nbytes
        )
        self.output_matrix = np.ndarray(
            len(self.worker_data.outputs.rkt_outputs.keys()),
            dtype=float,
            buffer=self.output_reference.buf,
        )

    def initialize(self, presolve):

        for i, key in enumerate(self.input_keys):
            self.input_matrix[0][i] = self.worker_data.inputs.rkt_inputs[
                key
            ].get_value()
            self.input_matrix[1][i] = self.worker_data.inputs.rkt_inputs[key].get_value(
                apply_conversion=True
            )
        self.local_pipe.send((WorkerMessages.initialize, presolve))

        result = self.local_pipe.recv()
        # we want to block here.

        if result == WorkerMessages.success:
            self.update_outputs()
            _log.warning("Worker initialized")
        else:
            raise RuntimeError("Worker failed to initialize")

    def solve(self, params):
        self.update_params(params)
        self.local_pipe.send(WorkerMessages.solve)

    def update_outputs(self):
        for i, key in enumerate(self.worker_data.outputs.rkt_outputs):
            self.worker_data.outputs.rkt_outputs[key].value = self.output_matrix[i]

    def update_params(self, params):
        for i, key in enumerate(self.input_keys):
            self.input_matrix[2][i] = params[key]

    def get_solution(self):
        if self.local_pipe.poll:
            result = self.local_pipe.recv()
            if result == WorkerMessages.success:
                return self.jacobian_matrix.copy(), self.output_matrix.copy()
            elif result == WorkerMessages.CyIpoptEvaluationError:
                raise cyipopt.CyIpoptEvaluationError
            else:
                raise ValueError("The worker failed and did not return a solution")

    def terminate(self):
        self.local_pipe.send(WorkerMessages.terminate)


class ReaktoroParallelManager:
    def __init__(self):
        self.registered_workers = {}
        self.processes = {}

    def register_block(self, block_idx, block_data):
        self.registered_workers[block_idx] = LocalWorker(block_data)

    def get_solve_and_get_function(self, block_idx):
        return (
            self.registered_workers[block_idx].solve,
            self.registered_workers[block_idx].get_solution,
        )

    def get_initialize_function(self, block_idx):
        return self.registered_workers[block_idx].initialize

    def start_workers(self):
        for idx, local_worker in self.registered_workers.items():
            worker_config = local_worker.worker_data.get_configs()
            process = mp.Process(
                target=ReaktoroActor,
                args=(
                    local_worker.remote_pipe,
                    worker_config,
                    local_worker.input_reference.name,
                    local_worker.output_reference.name,
                    local_worker.jacobian_reference.name,
                ),
            )
            process.start()
            _log.info(f"Started parallel worker {idx}")
            self.processes[idx] = process

    def terminate_workers(self):
        for idx, local_worker in self.registered_workers.items():
            local_worker.terminate()
            self.processes[idx].join()


def ReaktoroActor(
    pipe, reaktoro_block_data, input_matrix, output_matrix, jacobian_matrix
):
    reaktoro_worker = RemoteWorker(
        reaktoro_block_data, input_matrix, output_matrix, jacobian_matrix
    )
    dog_watch = time.time()
    max_time = (
        300  # five min time out should be enough for waiting, this will kill worker
    )
    while True:
        if pipe.poll():
            msg = pipe.recv()
            if isinstance(msg, tuple):
                command = msg[0]
                option = msg[1]
            else:
                command = msg

            if command == WorkerMessages.update_values:
                reaktoro_worker.update_inputs()
                result = WorkerMessages.success
            if command == WorkerMessages.initialize:
                reaktoro_worker.initialize(presolve=option)
                result = WorkerMessages.success
            if command == WorkerMessages.solve:
                reaktoro_worker.solve()
                result = WorkerMessages.success
            if command == WorkerMessages.terminate:
                return
            pipe.send(result)
            dog_watch = time.time()
        if abs(time.time() - dog_watch) > max_time:
            return
        time.sleep(1e-3)  # 1ms sleep time to reduce load