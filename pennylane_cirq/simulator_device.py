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
Cirq Simulator Devices
======================

**Module name:** :mod:`pennylane_cirq.simulator_device`

.. currentmodule:: pennylane_cirq.simulator_device

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Cirq simulators as PennyLane device.

Classes
-------

.. autosummary::
   SimulatorDevice
   MixedStateSimulatorDevice

----
"""
import math
import itertools as it

import cirq
import numpy as np
import pennylane as qml

from .cirq_device import CirqDevice
from .cirq_operation import CirqOperation


class SimulatorDevice(CirqDevice):
    r"""Cirq simulator device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to >= 1. If ``None``, expectation values are calculated analytically.
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
        simulator (Optional[cirq.Simulator]): Optional custom simulator object to use. If
            None, the default ``cirq.Simulator()`` will be used instead.
    """
    name = "Cirq Simulator device for PennyLane"
    short_name = "cirq.simulator"
    # pylint: disable=too-many-arguments
    def __init__(self, wires, shots=None, qubits=None, simulator=None):
        super().__init__(wires, shots, qubits)

        self._simulator = simulator or cirq.Simulator()

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
        if not self.shots is None:
            raise qml.DeviceError("The operation BasisState is only supported in analytic mode.")

        wires = basis_state_operation.wires

        if len(basis_state_operation.parameters[0]) != len(wires):
            raise qml.DeviceError(
                "For BasisState, the state has to be specified for the correct number of qubits. Got a state for {} qubits, expected {}.".format(
                    len(basis_state_operation.parameters[0]), len(self.qubits)
                )
            )

        if not np.all(np.isin(basis_state_operation.parameters[0], np.array([0, 1]))):
            raise qml.DeviceError(
                "Argument for BasisState can only contain 0 and 1. Got {}".format(
                    basis_state_operation.parameters[0]
                )
            )

        # expand basis state to device wires
        basis_state_array = np.zeros(self.num_wires, dtype=int)
        basis_state_array[wires] = basis_state_operation.parameters[0]

        self._initial_state = np.zeros(2 ** len(self.qubits), dtype=np.complex64)
        basis_state_idx = np.sum(2 ** np.argwhere(np.flip(basis_state_array) == 1))
        self._initial_state[basis_state_idx] = 1.0

    def _expand_state(self, state_vector, wires):
        """Expands state vector to more wires"""
        basis_states = np.array(list(it.product([0, 1], repeat=len(wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
        unravelled_indices[:, wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        state = np.zeros([2 ** self.num_wires], dtype=np.complex64)
        state[ravelled_indices] = state_vector
        state_vector = state.reshape([2] * self.num_wires)

        return state_vector.flatten()

    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        # pylint: disable=missing-function-docstring
        if not self.shots is None:
            raise qml.DeviceError(
                "The operation QubitStateVector is only supported in analytic mode."
            )

        state_vector = np.array(qubit_state_vector_operation.parameters[0], dtype=np.complex64)
        wires = self.map_wires(qubit_state_vector_operation.wires)

        if len(wires) != self.num_wires or sorted(wires) != wires.tolist():
            state_vector = self._expand_state(state_vector, wires)

        if len(state_vector) != 2 ** len(self.qubits):
            raise qml.DeviceError(
                "For QubitStateVector, the state has to be specified for the correct number of qubits. Got a state of length {}, expected {}.".format(
                    len(state_vector), 2 ** len(wires)
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

        # TODO: remove the need for this hack by keeping better track of unused wires
        # We apply identity gates to all wires, otherwise Cirq would ignore
        # wires that are not acted upon

        for q in self.qubits:
            self.pre_rotated_circuit.append(cirq.IdentityGate(1)(q))

        for q in self.qubits:
            self.circuit.append(cirq.IdentityGate(1)(q))

        if self.shots is None:
            if (
                isinstance(self._simulator, cirq.DensityMatrixSimulator)
                and self._initial_state is not None
            ):
                if np.shape(self._initial_state) == (2 ** self.num_wires,):
                    self._initial_state = self._convert_to_density_matrix(self._initial_state)

            self._result = self._simulator.simulate(self.circuit, initial_state=self._initial_state)
            self._state = self._get_state_from_cirq(self._result)

    def analytic_probability(self, wires=None):
        # pylint: disable=missing-function-docstring
        if self._state is None:
            return None

        probs = self._get_computational_basis_probs()
        return self.marginal_prob(probs, wires)

    def _convert_to_density_matrix(self, state_vec):
        """Convert ``state_vec`` into a density matrix."""
        dim = 2 ** self.num_wires
        return np.kron(state_vec, state_vec.conj()).reshape((dim, dim))

    @staticmethod
    def _get_state_from_cirq(result):
        """Extract the state array from a Cirq TrialResult ``result``"""
        return np.array(result.state_vector())

    def _get_computational_basis_probs(self):
        """Extract the probabilities of all computational basis measurements."""
        return np.abs(self._state) ** 2

    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            The state includes possible basis rotations for non-diagonal
            observables. Note that this behaviour differs from PennyLane's
            default.qubit plugin.
        """
        return self._state

    def generate_samples(self):
        # pylint: disable=missing-function-docstring
        if self.shots is None:
            return super().generate_samples()

        for wire in range(self.num_wires):
            self.circuit.append(cirq.measure(self.qubits[wire], key=str(wire)))

        self._result = self._simulator.run(self.circuit, repetitions=self.shots)

        return np.array(
            [self._result.measurements[str(wire)].flatten() for wire in range(self.num_wires)]
        ).T.astype(int)

    def expval(self, observable, shot_range=None, bin_size=None):
        # pylint: disable=missing-function-docstring
        # Analytic mode
        if self.shots is None:
            if not isinstance(observable, qml.operation.Tensor):
                # Observable on a single wire
                # Projector, Hermitian
                if self._observable_map[observable.name] is None or observable.name == "Projector":
                    return super().expval(observable, shot_range, bin_size)

                if observable.name == "Hadamard":
                    circuit = self.circuit
                    obs = cirq.PauliSum() + self.to_paulistring(qml.PauliZ(wires=observable.wires))
                else:
                    circuit = self.pre_rotated_circuit
                    obs = cirq.PauliSum() + self.to_paulistring(observable)

            # Observables are in tensor form
            else:
                # Projector, Hamiltonian, Hermitian
                for name in observable.name:
                    if self._observable_map[name] is None or name == "Projector":
                        return super().expval(observable, shot_range, bin_size)

                if "Hadamard" in observable.name:
                    list_obs = []
                    for obs in observable.obs:
                        list_obs.append(qml.PauliZ(wires=obs.wires))

                    T = qml.operation.Tensor(*list_obs)
                    circuit = self.circuit
                    obs = cirq.PauliSum() + self.to_paulistring(T)
                else:
                    circuit = self.pre_rotated_circuit
                    obs = cirq.PauliSum() + self.to_paulistring(observable)

            return self._simulator.simulate_expectation_values(
                program=circuit,
                observables=obs,
                initial_state=self._initial_state,
            )[0].real

        # Shots mode
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        return np.squeeze(np.mean(samples, axis=0))


class MixedStateSimulatorDevice(SimulatorDevice):
    r"""Cirq mixed-state simulator device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to >= 1. If ``None``, expectation values are calculated analytically.
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """
    name = "Cirq Mixed-State Simulator device for PennyLane"
    short_name = "cirq.mixedsimulator"

    _mixed_sim_operation_map = {
        "BitFlip": CirqOperation(cirq.bit_flip),
        "PhaseFlip": CirqOperation(cirq.phase_flip),
        "PhaseDamp": CirqOperation(cirq.phase_damp),
        "AmplitudeDamp": CirqOperation(cirq.amplitude_damp),
        "Depolarize": CirqOperation(cirq.depolarize),
    }

    def __init__(self, wires, shots=None, qubits=None):
        self._operation_map = dict(self._operation_map, **self._mixed_sim_operation_map)
        super().__init__(wires, shots, qubits)

        self._simulator = cirq.DensityMatrixSimulator()

        self._initial_state = None
        self._result = None
        self._state = None

    def expval(self, observable, shot_range=None, bin_size=None):
        # The simulate_expectation_values from Cirq for mixed states involves
        # a density matrix check, which does not always pass because the tolerance
        # is too low. If the error is raised we use the PennyLane function for
        # expectation value.
        try:
            return super().expval(observable, shot_range, bin_size)
        except ValueError:
            return qml.QubitDevice.expval(self, observable, shot_range, bin_size)

    def _apply_basis_state(self, basis_state_operation):
        super()._apply_basis_state(basis_state_operation)
        self._initial_state = self._convert_to_density_matrix(self._initial_state)

    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        super()._apply_qubit_state_vector(qubit_state_vector_operation)
        self._initial_state = self._convert_to_density_matrix(self._initial_state)

    @staticmethod
    def _get_state_from_cirq(result):
        """Extract the state array from a Cirq TrialResult"""
        return np.array(result.final_density_matrix)

    def _get_computational_basis_probs(self):
        """Extract the probabilities of all computational basis measurements."""
        return np.diag(self._state).real

    @property
    def state(self):
        """Returns the density matrix of the circuit prior to measurement.

        .. note::

            The state includes possible basis rotations for non-diagonal
            observables. Note that this behaviour differs from PennyLane's
            default.qubit plugin.
        """
        return self._state
