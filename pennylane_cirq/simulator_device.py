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
SimulatorDevice
========

**Module name:** :mod:`pennylane_cirq.simulator_device`

.. currentmodule:: pennylane_cirq.simulator_device

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Target Framework device/simulator as a PennyLane device.

It can inherit from the abstract FrameworkDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   SimulatorDevice

----
"""
import itertools

from collections import OrderedDict

import cirq
import math
import numpy as np
import pennylane as qml

from .cirq_device import CirqDevice


class SimulatorDevice(CirqDevice):
    r"""Cirq simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
    """
    name = "Cirq Simulator device for PennyLane"
    short_name = "cirq.simulator"

    def __init__(self, wires, shots=0, qubits=None):
        # Todo: docstring
        super().__init__(wires, shots, qubits)

        self.initial_state = None
        self.simulator = cirq.Simulator()
        self.result = None
        self.measurements = None
        self.state = None

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

            if self.shots > 0:
                raise qml.DeviceError(
                    "The operation BasisState is only supported in analytic mode (shots=0)."
                )

            basis_state_array = np.array(par[0])

            if len(basis_state_array) != len(self.qubits):
                raise qml.DeviceError(
                    "For BasisState, the state has to be specified for the correct number of qubits. Got a state for {} qubits, expected {}.".format(len(basis_state_array), len(self.qubits))
                    )

            if not np.all(np.isin(basis_state_array, np.array([0, 1]))):
                raise qml.DeviceError("Argument for BasisState can only contain 0 and 1. Got {}".format(par[0]))

            self.initial_state = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
            basis_state_idx = np.sum(2 ** np.argwhere(np.flip(basis_state_array) == 1))
            self.initial_state[basis_state_idx] = 1.0

        elif operation == "QubitStateVector":
            if not self._first_apply:
                raise qml.DeviceError(
                    "The operation QubitStateVector is only supported at the beginning of a circuit."
                )

            if self.shots > 0:
                raise qml.DeviceError(
                    "The operation QubitStateVector is only supported in analytic mode (shots=0)."
                )

            state_vector = np.array(par[0], dtype=np.complex64)

            if len(state_vector) != 2**len(self.qubits):
                raise qml.DeviceError(
                    "For QubitStateVector, the state has to be specified for the correct number of qubits. Got a state of length {}, expected {}.".format(len(state_vector), 2**len(self.qubits))
                    )

            norm_squared = np.sum(np.abs(state_vector)**2)
            if not np.isclose(norm_squared, 1.0, atol=1e-3, rtol=0):
                raise qml.DeviceError(
                    "The given state for QubitStateVector is not properly normalized to 1.0. Got norm {}".format(math.sqrt(norm_squared))
                )

            self.initial_state = state_vector

        if self._first_apply:
            self._first_apply = False

    def pre_measure(self):
        super().pre_measure()

        # We apply an identity gate to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        self.circuit.append(cirq.IdentityGate(len(self.qubits))(*self.qubits))

        if self.shots == 0:
            if self.initial_state is None:
                self.result = self.simulator.simulate(self.circuit)
            else:
                self.result = self.simulator.simulate(
                    self.circuit, initial_state=self.initial_state
                )

            self.state = np.array(self.result.state_vector())
        # Do nothing if there is nothing to measure
        elif self.obs_queue:
            for e in self.obs_queue:
                wire = e.wires[0]

                self.circuit.append(cirq.measure(self.qubits[wire], key=str(wire)))

            num_shots = max(
                [self.shots] + [e.num_samples for e in self.obs_queue if e.return_type == "sample"]
            )

            self.result = self.simulator.run(self.circuit, repetitions=num_shots)

            # Bring measurements to a more managable form, but keep True/False as values for now
            # They will be changed in the measurement routines where the observable is available
            self.measurements = np.array(
                [self.result.measurements[str(wire)].flatten() for wire in range(self.num_wires)]
            )

    def probability(self):
        if self.state is None:
            return None

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

        for marginal_state in enumerate(marginal_states):
            marginal_probabilities[marginal_state] = np.sum(
                [probabilities[state] for state in probabilities if tuple(state[wire] for wire in wires) == marginal_state]
            )

        return marginal_probabilities


    def expval(self, observable, wires, par):
        num_wires = len(wires)

        eigenvalues = np.ones(2**num_wires)

        if observable == "Hermitian":
            # Take the eigenvalues from the stored values
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())
            eigenvalues = self._eigs[Hkey]["eigval"]
        elif observable != "Identity":
            # TODO: Add support for Tensor observables after it is merged in PL
            eigenvalues[1] = -1

        if self.shots == 0:
            # We have to use the state of the simulation to find the expectation value
            marginal_probability = np.fromiter(self.marginal_probability(wires).values(), dtype=np.float)

            return np.dot(eigenvalues, marginal_probability)
        else:
            return self.sample(observable, wires, par).mean()

    def var(self, observable, wires, par):
        wire = wires[0]

        zero_value = 1
        one_value = -1

        if observable == "Hermitian":
            # Take the eigenvalues from the stored values
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())
            zero_value = self._eigs[Hkey]["eigvec"][0]
            one_value = self._eigs[Hkey]["eigvec"][1]

        elif observable == "Identity":
            one_value = 1

        if self.shots == 0:
            # We have to use the state of the simulation to find the expectation value
            probabilities = self.probability()

            zero_marginal_prob = np.sum(
                [probabilities[state] for state in probabilities if state[wire] == 0]
            )
            one_marginal_prob = 1 - zero_marginal_prob

            # Var = <A^2> - <A>^2
            return (
                zero_marginal_prob * zero_value ** 2
                + one_marginal_prob * one_value ** 2
                - (zero_marginal_prob * zero_value + one_marginal_prob * one_value) ** 2
            )
        else:
            return self.sample(observable, wires, par).var()

    def sample(self, observable, wires, par, n=None):
        if not n:
            n = self.shots

        wire = wires[0]

        zero_value = 1
        one_value = -1

        if observable == "Hermitian":
            # Take the eigenvalues from the stored values
            Hmat = par[0]
            Hkey = tuple(Hmat.flatten().tolist())
            zero_value = self._eigs[Hkey]["eigvec"][0]
            one_value = self._eigs[Hkey]["eigvec"][1]

        elif observable == "Identity":
            one_value = 1

        if self.shots == 0:
            # We have to use the state of the simulation to find the expectation value
            probabilities = self.probability()

            zero_marginal_prob = np.sum(
                [probabilities[state] for state in probabilities if state[wire] == 0]
            )
            one_marginal_prob = 1 - zero_marginal_prob

            return np.random.choice(
                [zero_value, one_value], size=n, p=[zero_marginal_prob, one_marginal_prob]
            )
        else:
            return CirqDevice._convert_measurements(
                self.measurements[wires[0]], zero_value, one_value
            )[:n]


