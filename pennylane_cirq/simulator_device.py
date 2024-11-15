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
import numpy as np
import cirq
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

    def capabilities(self):  # pylint: disable=missing-function-docstring
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=self.shots is None  # State information is only set if obtaining shots
        )
        return capabilities

    def _apply_basis_state(self, basis_state_operation):
        # pylint: disable=missing-function-docstring
        if self.shots is not None:
            raise qml.DeviceError("The operation BasisState is only supported in analytic mode.")

        self._initial_state = basis_state_operation.state_vector(wire_order=self.wires).flatten()

    def _apply_state_prep(self, state_prep_operation):
        # pylint: disable=missing-function-docstring
        if self.shots is not None:
            raise qml.DeviceError("The operator StatePrep is only supported in analytic mode.")

        self._initial_state = state_prep_operation.state_vector(wire_order=self.wires).flatten()

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        super().apply(operations, **kwargs)

        if self.shots is None:
            self._result = self._simulator.simulate(
                self.circuit, qubit_order=self.qubits, initial_state=self._initial_state
            )
            self._state = self._get_state_from_cirq(self._result)

    def analytic_probability(self, wires=None):
        # pylint: disable=missing-function-docstring
        if self._state is None:
            return None

        probs = self._get_computational_basis_probs()
        return self.marginal_prob(probs, wires)

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
            all_observables = (
                list(observable.operands) if isinstance(observable, qml.ops.Prod) else [observable]
            )

            for obs in all_observables:
                if self._observable_map[obs.name] is None or obs.name == "Projector":
                    return super().expval(observable, shot_range, bin_size)

            if "Hadamard" in [op.name for op in all_observables]:
                list_obs = []

                for obs in all_observables:
                    list_obs.append(qml.PauliZ(wires=obs.wires))

                T = qml.prod(*list_obs)

                circuit = self.circuit
                obs = cirq.PauliSum() + self.to_paulistring(T)

            else:
                circuit = self.pre_rotated_circuit
                obs = cirq.PauliSum() + self.to_paulistring(observable)

            return self._simulator.simulate_expectation_values(
                program=circuit,
                qubit_order=self.qubits,
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

    def capabilities(self):  # pylint: disable=missing-function-docstring
        capabilities = super().capabilities().copy()
        capabilities.update(
            returns_state=self.shots is None  # State information is only set if obtaining shots
        )
        return capabilities

    def expval(self, observable, shot_range=None, bin_size=None):
        # The simulate_expectation_values from Cirq for mixed states involves
        # a density matrix check, which does not always pass because the tolerance
        # is too low. If the error is raised we use the PennyLane function for
        # expectation value.
        try:
            return super().expval(observable, shot_range, bin_size)
        except ValueError:
            return qml.devices.QubitDevice.expval(self, observable, shot_range, bin_size)

    def _apply_basis_state(self, basis_state_operation):
        super()._apply_basis_state(basis_state_operation)
        self._initial_state = self._convert_to_density_matrix(self._initial_state)

    def _apply_state_prep(self, state_prep_operation):
        super()._apply_state_prep(state_prep_operation)
        self._initial_state = self._convert_to_density_matrix(self._initial_state)

    def _convert_to_density_matrix(self, state_vec):
        """Convert ``state_vec`` into a density matrix."""
        dim = 2**self.num_wires
        return np.kron(state_vec, state_vec.conj()).reshape((dim, dim))

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
