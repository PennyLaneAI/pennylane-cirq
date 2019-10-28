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
import itertools

from collections import OrderedDict
import functools

import cirq
import math
import numpy as np
import pennylane as qml

from .cirq_device import CirqDevice


@functools.lru_cache()
def z_eigs(n):
    """Return the eigenvalues of an n-fold tensor product of Pauli Z operators."""
    if n == 1:
        return np.array([1, -1])

    return np.concatenate([z_eigs(n - 1), -z_eigs(n - 1)])


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
    _capabilities = {"model": "qubit", "tensor_observables": False}

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):
        super().__init__(wires, shots, qubits)

        self.initial_state = None
        self.simulator = cirq.Simulator()
        self.result = None
        self.measurements = None
        self.state = None
        self.analytic = analytic

    def reset(self):
        super().reset()

        self.initial_state = None

    def pre_apply(self):
        super().pre_apply()

        self._first_apply = True

    def apply(self, operation, wires, par):
        super().apply(operation, wires, par)

        if operation == "BasisState":
            if not self._first_apply:
                raise qml.DeviceError(
                    "The operation BasisState is only supported at the beginning of a circuit."
                )

            if not self.analytic:
                raise qml.DeviceError(
                    "The operation BasisState is only supported in analytic mode."
                )

            basis_state_array = np.array(par[0])

            if len(basis_state_array) != len(self.qubits):
                raise qml.DeviceError(
                    "For BasisState, the state has to be specified for the correct number of qubits. Got a state for {} qubits, expected {}.".format(
                        len(basis_state_array), len(self.qubits)
                    )
                )

            if not np.all(np.isin(basis_state_array, np.array([0, 1]))):
                raise qml.DeviceError(
                    "Argument for BasisState can only contain 0 and 1. Got {}".format(par[0])
                )

            self.initial_state = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
            basis_state_idx = np.sum(2 ** np.argwhere(np.flip(basis_state_array) == 1))
            self.initial_state[basis_state_idx] = 1.0

        elif operation == "QubitStateVector":
            if not self._first_apply:
                raise qml.DeviceError(
                    "The operation QubitStateVector is only supported at the beginning of a circuit."
                )

            if not self.analytic:
                raise qml.DeviceError(
                    "The operation QubitStateVector is only supported in analytic mode."
                )

            state_vector = np.array(par[0], dtype=np.complex64)

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

            self.initial_state = state_vector

        if self._first_apply:
            self._first_apply = False

    def pre_measure(self):
        super().pre_measure()

        # We apply an identity gate to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        self.circuit.append(cirq.IdentityGate(len(self.qubits))(*self.qubits))

        if self.analytic:
            if self.initial_state is None:
                self.result = self.simulator.simulate(self.circuit)
            else:
                self.result = self.simulator.simulate(
                    self.circuit, initial_state=self.initial_state
                )

            self.state = np.array(self.result.state_vector())

        elif self.obs_queue:
            for wire in range(self.num_wires):
                self.circuit.append(cirq.measure(self.qubits[wire], key=str(wire)))

            self.result = self.simulator.run(self.circuit, repetitions=self.shots)

            # Bring measurements to a more managable form, but keep True/False as values for now
            # They will be changed in the measurement routines where the observable is available

            self.measurements = np.array(
                [self.result.measurements[str(wire)].flatten() for wire in range(self.num_wires)]
            )

    def probability(self):
        if self.state is None:
            raise qml.DeviceError("Probability can not be computed because the internal state is None.")

        states = itertools.product(range(2), repeat=self.num_wires)
        probs = np.abs(self.state) ** 2

        return OrderedDict(zip(states, probs))

    def marginal_probability(self, wires):
        """The marginal probability over the given wires.

        Args:
            wires (Array[int]): the wires with respect to which the marginal probabilities 
                are calculated
        """
        num_wires = len(wires)
        probabilities = self.probability()
        marginal_states = itertools.product(range(2), repeat=num_wires)

        marginal_probabilities = OrderedDict()

        for marginal_state in marginal_states:
            marginal_probabilities[marginal_state] = 0.0

        for state in probabilities:
            marginal_state = tuple(state[wire] for wire in wires)
            marginal_probabilities[marginal_state] += probabilities[state]

        return marginal_probabilities

    def _get_eigenvalues(self, observable, wires, par):
        """Return the eigenvalues of the given observable.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (List[int] or List[List[int]]): subsystems the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            array[float]: eigenvalues of the observable
        """
        num_wires = len(wires)

        # All ones corresponds to the Identity observable
        eigenvalues = np.ones(2 ** num_wires)

        if observable == "Hermitian":
            # Take the eigenvalues from the stored values
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())

            eigenvalues = self._eigs[Hkey]["eigval"]
        elif observable != "Identity":
            # If we don't have an Hermitian observable we use
            # a diagonalization to tensors of Z observables
            eigenvalues = z_eigs(num_wires)

        return eigenvalues

    def expval(self, observable, wires, par):
        eigenvalues = self._get_eigenvalues(observable, wires, par)

        if self.analytic:
            # We have to use the state of the simulation to find the expectation value
            marginal_probability = np.fromiter(
                self.marginal_probability(wires).values(), dtype=np.float
            )

            return np.dot(eigenvalues, marginal_probability)
        else:
            return self.sample(observable, wires, par).mean()

    def var(self, observable, wires, par):
        eigenvalues = self._get_eigenvalues(observable, wires, par)

        if self.analytic:
            # We have to use the state of the simulation to find the expectation value
            marginal_probability = np.fromiter(
                self.marginal_probability(wires).values(), dtype=np.float
            )

            return (
                np.dot(eigenvalues ** 2, marginal_probability)
                - np.dot(eigenvalues, marginal_probability) ** 2
            )
        else:
            return self.sample(observable, wires, par).var()

    def sample(self, observable, wires, par):
        eigenvalues = self._get_eigenvalues(observable, wires, par)

        if self.analytic:
            # We have to use the state of the simulation to find the expectation value
            marginal_probabilities = np.fromiter(
                self.marginal_probability(wires).values(), dtype=np.float64
            )

            # np.random.choice does not even tolerate small deviations
            # from 1, so we have to adjust the probabilities here
            probability_sum = np.sum(marginal_probabilities)
            marginal_probabilities /= probability_sum

            return np.random.choice(eigenvalues, size=self.shots, p=marginal_probabilities)
        else:
            return CirqDevice._convert_measurements(self.measurements[wires], eigenvalues)
