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
This module provides the ``QFlex`` from Cirq.
"""
import cirq

try:
    import qflexcirq
except ImportError:
    raise ImportError(
        "QFlex Cirq is needed for the QFlex device to work."
        "\nInstallation instructions can be found at https://github.com/ngnrsaa/qflex"
    )

from .simulator_device import SimulatorDevice
from .cirq_device import CirqDevice


# TODO: Not fully working yet; need to update cirq.Circuit to use GridQubit.
class QFlexDevice(SimulatorDevice):
    r"""QFlex device for PennyLane.
    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """
    name = "QFlex device for PennyLane"
    short_name = "cirq.qflex"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):

        if qubits is None:
            qubits = [cirq.GridQubit(0, wire) for wire in range(wires)]

        super().__init__(wires, shots, analytic, qubits)
        grid = qflexcirq.QFlexGrid().create_rectangular(1, len(self.wires))
        self.qflex_device = qflexcirq.QFlexVirtualDevice(qflex_grid=grid)

        self.circuit = cirq.Circuit()
        self._simulator = qflexcirq.QFlexSimulator()

        self._initial_state = None
        self._result = None
        self._state = None

    def apply(self, operations, **kwargs):
        CirqDevice.apply(self, operations, **kwargs)

        # We apply an identity gate to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        for qb in self.qubits:
            self.circuit.append(cirq.IdentityGate(1)(qb))
        if self.analytic:
            qflex_circuit = qflexcirq.QFlexCircuit(
                cirq_circuit=self.circuit,
                device=self.qflex_device,
                allow_decomposition=True
            )

            self._state = self._simulator.compute_amplitudes(
                program=qflex_circuit,
                bitstrings=list(range(2 ** len(self.wires)))
            )
