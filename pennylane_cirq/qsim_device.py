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
import numpy as np
import cirq
import pennylane as qml

try:
    import qsimcirq
except ImportError as e:
    raise ImportError(
        "qsim Cirq is needed for the qsim devices to work."
        "\nIt can be installed using pip:"
        "\n\npip install qsimcirq"
    ) from e

from .simulator_device import SimulatorDevice
from .cirq_device import CirqDevice


class QSimDevice(SimulatorDevice):
    r"""qsim device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. If ``None``, expectation values are calculated analytically.
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
        qsim_options: (Dict[str, Any]): A dictionary with options for the qsimh simulator. See the `qsim
            usage documentation <https://github.com/quantumlib/qsim/blob/master/docs/usage.md>`__
            for further details.
    """

    name = "QSim device for PennyLane"
    short_name = "cirq.qsim"

    def __init__(self, wires, shots=None, qubits=None, qsim_options=None):
        super().__init__(wires, shots)
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())
        self._simulator = qsimcirq.QSimSimulator(qsim_options=qsim_options or {})

    def reset(self):
        # pylint: disable=missing-function-docstring
        super().reset()
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())

    @property
    def operations(self):
        # pylint: disable=missing-function-docstring
        return set(self._base_operation_map) - {
            "StatePrep",
            "BasisState",
            "CRX",
            "CRY",
            "CRZ",
            "CRot",
        }

    @property
    def observables(self):
        # pylint: disable=missing-function-docstring
        return set(self._base_observable_map)

    def expval(self, observable, shot_range=None, bin_size=None):

        if isinstance(observable.simplify(), qml.Identity):
            return 1

        return super().expval(observable, shot_range, bin_size)


class QSimhDevice(SimulatorDevice):
    r"""qsimh device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        qsimh_options (dict): A dictionary with options for the qsimh simulator. See the `qsim
            usage documentation <https://github.com/quantumlib/qsim/blob/master/docs/usage.md>`__
            for further details.
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. If ``None``, expectation values are calculated analytically.
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """

    name = "qsimh device for PennyLane"
    short_name = "cirq.qsimh"

    def __init__(self, wires, qsimh_options, shots=None, qubits=None):
        super().__init__(wires, shots, qubits)

        self.circuit = None
        self.qsimh_options = qsimh_options
        self._simulator = qsimcirq.QSimhSimulator(qsimh_options)

    @property
    def operations(self):
        # pylint: disable=missing-function-docstring
        return set(self._base_operation_map) - {
            "StatePrep",
            "BasisState",
            "CRX",
            "CRY",
            "CRZ",
            "CRot",
        }

    @property
    def observables(self):
        # pylint: disable=missing-function-docstring
        return set(self._base_observable_map)

    def capabilities(self):  # pylint: disable=missing-function-docstring
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=(self.shots is None),  # State information is only set if obtaining shots
        )
        return capabilities

    def expval(self, observable, shot_range=None, bin_size=None):
        return qml.devices.QubitDevice.expval(self, observable, shot_range, bin_size)

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        CirqDevice.apply(self, operations, **kwargs)

        # TODO: remove the need for this hack by keeping better track of unused wires
        # We apply identity gates to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        for qb in self.qubits:
            self.circuit.append(cirq.IdentityGate(1)(qb))

        state = self._simulator.compute_amplitudes(
            program=self.circuit, bitstrings=list(range(2 ** len(self.wires)))
        )

        self._state = np.array(state)

    def generate_samples(self):
        # pylint: disable=missing-function-docstring
        number_of_states = 2**self.num_wires

        rotated_prob = self.analytic_probability()
        if rotated_prob is not None:
            rotated_prob /= np.sum(rotated_prob)

        samples = self.sample_basis_states(number_of_states, rotated_prob)
        return self.states_to_binary(samples, self.num_wires)
