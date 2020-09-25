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
This module provides the ``QSimDevice`` and ``QSimhDevice`` from Cirq.
"""
import cirq
import numpy as np
import pennylane as qml

try:
    import qsimcirq
except ImportError:
    raise ImportError(
        "QSim Cirq is needed for the QSim devices to work."
        "\nIt can be installed using pip:"
        "\n\npip install qsimcirq"
    )

from .simulator_device import SimulatorDevice
from .cirq_device import CirqDevice


class QSimDevice(SimulatorDevice):
    r"""QSim device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
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
    name = "QSim device for PennyLane"
    short_name = "cirq.qsim"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):
        super().__init__(wires, shots, analytic, qubits)
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())
        self._simulator = qsimcirq.QSimSimulator()

    def reset(self):
        # pylint: disable=missing-function-docstring
        super().reset()
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())

    def _apply_basis_state(self, basis_state_operation):
        # pylint: disable=missing-function-docstring
        if not self.analytic:
            raise qml.DeviceError("The operation BasisState is only supported in analytic mode.")

        self.reset()
        basis_state_array = np.array(basis_state_operation.parameters[0])

        if len(basis_state_array) != len(self.qubits):
            raise qml.DeviceError(
                "For BasisState, the state has to be specified for the correct number of qubits. Got a state for {} qubits, expected {}.".format(
                    len(basis_state_array), len(self.qubits)
                )
            )

        if not np.all(np.isin(basis_state_array, np.array([0, 1]))):
            raise qml.DeviceError(
                "Argument for BasisState can only contain 0 and 1. Got {}".format(
                    basis_state_operation.parameters[0]
                )
            )

        n_idx = np.where(basis_state_array)

        for q in np.array(self.qubits)[n_idx]:
            self.circuit.append(cirq.X(q))

    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        # pylint: disable=missing-function-docstring
        raise NotImplementedError("QSim does not support arbitrary state preparations.")


class QSimhDevice(SimulatorDevice):
    r"""QSimh device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
        qsimh_options (dict): A dictionary with options for the QSimh simulator. See the `QSim
            usage documentation <https://github.com/quantumlib/qsim/blob/master/docs/usage.md>`__
            for further details.
    """
    name = "QSimh device for PennyLane"
    short_name = "cirq.qsimh"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None, qsimh_options=None):
        super().__init__(wires, shots, analytic, qubits)
        if qsimh_options is None:
            qsimh_options = {"k": [0], "w": 0, "p": 0, "r": 1}
        self.circuit = None
        self.qsimh_options = qsimh_options
        self._simulator = qsimcirq.QSimhSimulator(qsimh_options)

    def _apply_basis_state(self, basis_state_operation):
        # pylint: disable=missing-function-docstring
        if not self.analytic:
            raise qml.DeviceError("The operation BasisState is only supported in analytic mode.")

        self.reset()
        basis_state_array = np.array(basis_state_operation.parameters[0])

        if len(basis_state_array) != len(self.qubits):
            raise qml.DeviceError(
                "For BasisState, the state has to be specified for the correct number of qubits. Got a state for {} qubits, expected {}.".format(
                    len(basis_state_array), len(self.qubits)
                )
            )

        if not np.all(np.isin(basis_state_array, np.array([0, 1]))):
            raise qml.DeviceError(
                "Argument for BasisState can only contain 0 and 1. Got {}".format(
                    basis_state_operation.parameters[0]
                )
            )

        n_idx = np.where(basis_state_array)

        for q in np.array(self.qubits)[n_idx]:
            self.circuit.append(cirq.X(q))

    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        # pylint: disable=missing-function-docstring
        raise NotImplementedError("QSimh does not support arbitrary state preparations.")

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        CirqDevice.apply(self, operations, **kwargs)

        # TODO: remove the need for this hack by keeping better track of unused wires
        # We apply identity gates to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        for qb in self.qubits:
            self.circuit.append(cirq.IdentityGate(1)(qb))

        if self.analytic:
            state = self._simulator.compute_amplitudes(
                program=self.circuit, bitstrings=list(range(2 ** len(self.wires)))
            )

            self._state = np.array(state)
