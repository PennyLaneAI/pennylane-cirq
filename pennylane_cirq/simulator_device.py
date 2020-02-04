# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Cirq Simulator Device
========

**Module name:** :mod:`pennylane_cirq.simulator_device`

.. currentmodule:: pennylane_cirq.simulator_device

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Cirq simulator as a PennyLane device.

Classes
-------

.. autosummary::
   SimulatorDevice

----
"""
import math
import cirq
import numpy as np
import pennylane as qml

from .cirq_device import CirqDevice

class SimulatorDevice(CirqDevice):
    r"""Cirq simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need 
            to >= 1. In analytic mode, shots indicates the number of entries
            that are returned by device.sample.
        analytic (bool): Indicates that expectation values and variances should
            be calculated analytically. Defaults to `True`. 
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used 
            as wires. The wire number corresponds to the index in the list.
            By default, an array of `cirq.LineQubit` instances is created.
    """
    name = "Cirq Simulator device for PennyLane"
    short_name = "cirq.simulator"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):
        super().__init__(wires, shots, analytic, qubits)

        self._simulator = cirq.Simulator()

        self._initial_state = None
        self._result = None
        self._state = None

    def reset(self):
        # pylint: disable=missing-function-docstring
        super().reset()

        self._initial_state = None
        self._result = None
        self._state = None

    def _apply_basis_state(self, basis_state_operation):
        # pylint: disable=missing-function-docstring
        if not self.analytic:
            raise qml.DeviceError(
                "The operation BasisState is only supported in analytic mode."
            )

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

        self._initial_state = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
        basis_state_idx = np.sum(2 ** np.argwhere(np.flip(basis_state_array) == 1))
        self._initial_state[basis_state_idx] = 1.0

    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        # pylint: disable=missing-function-docstring
        if not self.analytic:
            raise qml.DeviceError(
                "The operation QubitStateVector is only supported in analytic mode."
            )

        state_vector = np.array(
            qubit_state_vector_operation.parameters[0], dtype=np.complex64
        )

        if len(state_vector) != 2 ** len(self.qubits):
            raise qml.DeviceError(
                "For QubitStateVector, the state has to be specified for the correct number of qubits. Got a state of length {}, expected {}.".format(
                    len(state_vector), 2 ** len(self.qubits)
                )
            )

        norm_squared = np.sum(np.abs(state_vector) ** 2)
        if not np.isclose(norm_squared, 1.0, atol=1e-3, rtol=0):
            raise qml.DeviceError(
                "The given state for QubitStateVector is not properly normalized to 1.0. Got norm {}".format(
                    math.sqrt(norm_squared)
                )
            )

        self._initial_state = state_vector

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        super().apply(operations, **kwargs)

        # We apply an identity gate to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        self.circuit.append(cirq.IdentityGate(len(self.qubits))(*self.qubits))

        if self.analytic:
            self._result = self._simulator.simulate(
                self.circuit, initial_state=self._initial_state
            )

            self._state = np.array(self._result.state_vector())

    def probability(self, wires=None):
        # pylint: disable=missing-function-docstring
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        probs = np.abs(self._state) ** 2

        return self.marginal_prob(probs, wires)

    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            The state includes possible basis rotations for non-diagonal 
            observables. Note that this behaviour is differs from PennyLane's
            default.qubit plugin.
        """
        return self._state

    def generate_samples(self):
        # pylint: disable=missing-function-docstring
        if self.analytic:
            return super().generate_samples()

        for wire in range(self.num_wires):
            self.circuit.append(cirq.measure(self.qubits[wire], key=str(wire)))

        self._result = self._simulator.run(self.circuit, repetitions=self.shots)

        # Bring measurements to a more managable form, but keep True/False as values for now
        # They will be changed in the measurement routines where the observable is available
        return np.array(
            [
                self._result.measurements[str(wire)].flatten()
                for wire in range(self.num_wires)
            ]
        ).T.astype(int)
