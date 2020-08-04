# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments
"""
This module provides the ``QSimDevice`` from Cirq.
"""
import cirq
import qsimcirq

from .simulator_device import SimulatorDevice


class QSimDevice(SimulatorDevice):
    r"""QSim device for PennyLane.
    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """
    name = "QSim device for PennyLane"
    short_name = "cirq.qsim"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):
        super().__init__(wires, shots, analytic, qubits)
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())
        self._simulator = qsimcirq.QSimSimulator()

        self._initial_state = None
        self._result = None
        self._state = None
