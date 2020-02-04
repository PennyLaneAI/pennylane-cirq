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
Unit tests for the Simulator plugin
"""
import pytest
import math

import pennylane as qml
import numpy as np
from pennylane_cirq import SimulatorDevice
import cirq

class TestDeviceIntegration:
    """Tests that the SimulatorDevice integrates well with PennyLane"""

    def test_device_loading(self):
        """Tests that the cirq.simulator device is properly loaded"""

        dev = qml.device("cirq.simulator", wires=2)

        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.short_name == "cirq.simulator"

        assert isinstance(dev, SimulatorDevice)


@pytest.fixture(scope="function")
def simulator_device_1_wire(shots, analytic):
    """Return a single wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(1, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def simulator_device_2_wires(shots, analytic):
    """Return a two wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(2, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def simulator_device_3_wires(shots, analytic):
    """Return a three wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(3, shots=shots, analytic=analytic)

@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestApply:
    """Tests that gates are correctly applied"""

    # fmt: off
    @pytest.mark.parametrize("name,input,expected_output", [
        ("PauliX", [1, 0], np.array([0, 1])),
        ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        ("PauliY", [1, 0], [0, 1j]),
        ("PauliY", [1 / math.sqrt(2), 1 / math.sqrt(2)], [-1j / math.sqrt(2), 1j / math.sqrt(2)]),
        ("PauliZ", [1, 0], [1, 0]),
        ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
        ("Hadamard", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        ("Hadamard", [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
    ])
    # fmt: on
    def test_apply_operation_single_wire_no_parameters(
        self, simulator_device_1_wire, tol, name, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        simulator_device_1_wire._obs_queue = []

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply(name, wires=[0], par=[])

        simulator_device_1_wire.initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.pre_measure()

        assert np.allclose(
            simulator_device_1_wire.state, np.array(expected_output), **tol
        )

    # fmt: off
    @pytest.mark.parametrize("name,input,expected_output", [
        ("CNOT", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("CNOT", [0, 0, 1, 0], [0, 0, 0, 1]),
        ("CNOT", [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0]),
        ("SWAP", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("SWAP", [0, 0, 1, 0], [0, 1, 0, 0]),
        ("SWAP", [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0], [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0]),
        ("CZ", [1, 0, 0, 0], [1, 0, 0, 0]),
        ("CZ", [0, 0, 0, 1], [0, 0, 0, -1]),
        ("CZ", [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)], [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]),
    ])
    # fmt: on
    def test_apply_operation_two_wires_no_parameters(
        self, simulator_device_2_wires, tol, name, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have no parameters."""

        simulator_device_2_wires._obs_queue = []

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply(name, wires=[0, 1], par=[])

        simulator_device_2_wires.initial_state = np.array(input, dtype=np.complex64)
        simulator_device_2_wires.pre_measure()

        assert np.allclose(
            simulator_device_2_wires.state, np.array(expected_output), **tol
        )

    # fmt: off
    @pytest.mark.parametrize("name,expected_output,par", [
        ("BasisState", [0, 0, 1, 0], [[1, 0]]),
        ("BasisState", [0, 0, 1, 0], [[1, 0]]),
        ("BasisState", [0, 0, 0, 1], [[1, 1]]),
        ("QubitStateVector", [0, 0, 1, 0], [[0, 0, 1, 0]]),
        ("QubitStateVector", [0, 0, 1, 0], [[0, 0, 1, 0]]),
        ("QubitStateVector", [0, 0, 0, 1], [[0, 0, 0, 1]]),
        ("QubitStateVector", [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)], [[1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)]]),
        ("QubitStateVector", [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)], [[1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)]]),
    ])
    # fmt: on
    def test_apply_operation_state_preparation(
        self, simulator_device_2_wires, tol, name, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        simulator_device_2_wires._obs_queue = []

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply(name, wires=[0, 1], par=par)

        simulator_device_2_wires.pre_measure()
        assert np.allclose(
            simulator_device_2_wires.state, np.array(expected_output), **tol
        )

    # fmt: off
    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("PhaseShift", [1, 0], [1, 0], [math.pi / 2]),
        ("PhaseShift", [0, 1], [0, 1j], [math.pi / 2]),
        ("PhaseShift", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / 2 + 1j / 2], [math.pi / 4]),
        ("RX", [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
        ("RX", [1, 0], [0, -1j], [math.pi]),
        ("RX", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 - 1j / 2], [math.pi / 2]),
        ("RY", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        ("RY", [1, 0], [0, 1], [math.pi]),
        ("RY", [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
        ("RZ", [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
        ("RZ", [0, 1], [0, 1j], [math.pi]),
        ("RZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [math.pi / 2]),
        ("Rot", [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
        ("Rot", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        ("Rot", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [0, 0, math.pi / 2]),
        ("Rot", [1, 0], [-1j / math.sqrt(2), -1 / math.sqrt(2)], [math.pi / 2, -math.pi / 2, math.pi / 2]),
        ("Rot", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / 2 + 1j / 2, -1 / 2 + 1j / 2], [-math.pi / 2, math.pi, math.pi]),
        ("QubitUnitary", [1, 0], [1j / math.sqrt(2), 1j / math.sqrt(2)], [
            np.array([
                [1j / math.sqrt(2), 1j / math.sqrt(2)],
                [1j / math.sqrt(2), -1j / math.sqrt(2)]
            ])
        ]),
        ("QubitUnitary", [0, 1], [1j / math.sqrt(2), -1j / math.sqrt(2)], [
            np.array([
                [1j / math.sqrt(2), 1j / math.sqrt(2)],
                [1j / math.sqrt(2), -1j / math.sqrt(2)]
            ])
        ]),
        ("QubitUnitary", [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1j], [
            np.array([
                [1j / math.sqrt(2), 1j / math.sqrt(2)],
                [1j / math.sqrt(2), -1j / math.sqrt(2)]
            ])
        ]),
    ])
    # fmt: on
    def test_apply_operation_single_wire_with_parameters(
        self, simulator_device_1_wire, tol, name, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        simulator_device_1_wire._obs_queue = []

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply(name, wires=[0], par=par)

        simulator_device_1_wire.initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.pre_measure()

        assert np.allclose(
            simulator_device_1_wire.state, np.array(expected_output), **tol
        )

    # fmt: off
    @pytest.mark.parametrize("name,input,expected_output,par", [
        ("CRX", [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
        ("CRX", [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
        ("CRX", [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), 1 / 2, -1j / 2], [math.pi / 2]),
        ("CRY", [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
        ("CRY", [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
        ("CRY", [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [math.pi / 2]),
        ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
        ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
        ("CRZ", [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0], [math.pi / 2]),
        ("CRot", [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2, 0, 0]),
        ("CRot", [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
        ("CRot", [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2], [0, 0, math.pi / 2]),
        ("CRot", [0, 0, 0, 1], [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)], [math.pi / 2, -math.pi / 2, math.pi / 2]),
        ("CRot", [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2], [-math.pi / 2, math.pi, math.pi]),
        ("QubitUnitary", [1, 0, 0, 0], [1, 0, 0, 0], [
            np.array([
                [1, 0, 0, 0],
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                [0, 0, 0, 1],
            ])
        ]),
        ("QubitUnitary", [0, 1, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [
            np.array([
                [1, 0, 0, 0],
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                [0, 0, 0, 1],
            ])
        ]),
        ("QubitUnitary", [1 / 2, 1 / 2, -1 / 2, 1 / 2], [1 / 2, 0, 1 / math.sqrt(2), 1 / 2], [
            np.array([
                [1, 0, 0, 0],
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                [0, 0, 0, 1],
            ])
        ]),
    ])
    # fmt: on
    def test_apply_operation_two_wires_with_parameters(
        self, simulator_device_2_wires, tol, name, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        simulator_device_2_wires._obs_queue = []

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply(name, wires=[0, 1], par=par)

        simulator_device_2_wires.initial_state = np.array(input, dtype=np.complex64)
        simulator_device_2_wires.pre_measure()

        assert np.allclose(
            simulator_device_2_wires.state, np.array(expected_output), **tol
        )

    # fmt: off
    @pytest.mark.parametrize("operation,par,match", [
        ("BasisState", [[2]], "Argument for BasisState can only contain 0 and 1"),
        ("BasisState", [[1.2]], "Argument for BasisState can only contain 0 and 1"),
        ("BasisState", [[0, 0, 1]], "For BasisState, the state has to be specified for the correct number of qubits"),
        ("BasisState", [[0, 0]], "For BasisState, the state has to be specified for the correct number of qubits"),
        ("QubitStateVector", [[0, 0, 1]], "For QubitStateVector, the state has to be specified for the correct number of qubits"),
        ("QubitStateVector", [[0, 0, 1, 0]], "For QubitStateVector, the state has to be specified for the correct number of qubits"),
        ("QubitStateVector", [[1]], "For QubitStateVector, the state has to be specified for the correct number of qubits"),
        ("QubitStateVector", [[0.5, 0.5]], "The given state for QubitStateVector is not properly normalized to 1"),
        ("QubitStateVector", [[1.1, 0]], "The given state for QubitStateVector is not properly normalized to 1"),
        ("QubitStateVector", [[0.7, 0.7j]], "The given state for QubitStateVector is not properly normalized to 1"),
    ])
    # fmt: on
    def test_state_preparation_error(self, simulator_device_1_wire, operation, par, match):
        """Tests that the state preparation routines raise proper errors for wrong parameter values."""

        simulator_device_1_wire._obs_queue = []

        simulator_device_1_wire.pre_apply()

        with pytest.raises(qml.DeviceError, match=match):
            simulator_device_1_wire.apply(operation, wires=[0], par=par)

    def test_basis_state_not_at_beginning_error(self, simulator_device_1_wire):
        """Tests that application of BasisState raises an error if is not
        the first operation."""

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("PauliX", wires=[0], par=[])

        with pytest.raises(qml.DeviceError, match="The operation BasisState is only supported at the beginning of a circuit."):
            simulator_device_1_wire.apply("BasisState", wires=[0], par=[[0]])

    def test_qubit_state_vector_not_at_beginning_error(self, simulator_device_1_wire):
        """Tests that application of QubitStateVector raises an error if is not
        the first operation."""

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("PauliX", wires=[0], par=[])

        with pytest.raises(qml.DeviceError, match="The operation QubitStateVector is only supported at the beginning of a circuit."):
            simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[[0, 1]])

@pytest.mark.parametrize("shots,analytic", [(100, False)])
class TestStatePreparationErrorsNonAnalytic:
    """Tests state preparation errors that occur for non-analytic devices."""

    def test_basis_state_not_analytic_error(self, simulator_device_1_wire):
        """Tests that application of BasisState raises an error if the device
        is not in analytic mode."""

        simulator_device_1_wire.pre_apply()
        with pytest.raises(qml.DeviceError, match="The operation BasisState is only supported in analytic mode."):
            simulator_device_1_wire.apply("BasisState", wires=[0], par=[[0]])

    def test_qubit_state_vector_not_analytic_error(self, simulator_device_1_wire):
        """Tests that application of QubitStateVector raises an error if the device
        is not in analytic mode."""

        dev = qml.device("cirq.simulator", wires=1, shots=1000, analytic=False)

        simulator_device_1_wire.pre_apply()
        with pytest.raises(qml.DeviceError, match="The operation QubitStateVector is only supported in analytic mode."):
            simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[[0, 1]])


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output", [
        (qml.Identity, [1, 0], 1),
        (qml.Identity, [0, 1], 1),
        (qml.Identity, [1/math.sqrt(2), -1/math.sqrt(2)], 1),
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        (qml.PauliX, [1/math.sqrt(2), -1/math.sqrt(2)], -1),
        (qml.PauliX, [1, 0], 0),
        (qml.PauliY, [1/math.sqrt(2), 1j/math.sqrt(2)], 1),
        (qml.PauliY, [1/math.sqrt(2), -1j/math.sqrt(2)], -1),
        (qml.PauliY, [1, 0], 0),
        (qml.PauliZ, [1, 0], 1),
        (qml.PauliZ, [0, 1], -1),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        (qml.Hadamard, [1, 0], 1/math.sqrt(2)),
        (qml.Hadamard, [0, 1], -1/math.sqrt(2)),
        (qml.Hadamard, [1/math.sqrt(2), 1/math.sqrt(2)], 1/math.sqrt(2)),
    ])
    # fmt: on
    def test_expval_single_wire_no_parameters(self, simulator_device_1_wire, tol, operation, input, expected_output):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        op = operation(0, do_queue=False)
        simulator_device_1_wire._obs_queue = [op]

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[input])
        simulator_device_1_wire.post_apply()
        
        simulator_device_1_wire.pre_measure()        
        res = simulator_device_1_wire.expval(op.name, wires=[0], par=[])

        assert np.isclose(res, expected_output, **tol) 

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
        (qml.Hermitian, [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
        (qml.Hermitian, [1/math.sqrt(2), -1/math.sqrt(2)], 1, [np.array([[1, 1j], [-1j, 1]])]),
    ])
    # fmt: on
    def test_expval_single_wire_with_parameters(self, simulator_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        op = operation(par[0], 0, do_queue=False)
        simulator_device_1_wire._obs_queue = [op]

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[input])
        simulator_device_1_wire.post_apply()
        
        simulator_device_1_wire.pre_measure()        
        res = simulator_device_1_wire.expval(op.name, wires=[0], par=par)

        assert np.isclose(res, expected_output, **tol) 

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [0, 1, 0, 0], -1, [
            np.array([
                [1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, -1, 0], 
                [0, 0, 0, 1],
            ])
        ]),
        (qml.Hermitian, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [
            np.array([
                [1, 1j, 0, 1], 
                [-1j, 1, 0, 0], 
                [0, 0, 1, -1j], 
                [1, 0, 1j, 1]
            ])
        ]),
        (qml.Hermitian, [0, 0, 0, 1], 0, [
            np.array([
                [0, 1j, 0, 0], 
                [-1j, 0, 0, 0], 
                [0, 0, 0, -1j], 
                [0, 0, 1j, 0]
            ])
        ]),
        (qml.Hermitian, [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [
            np.array([
                [1, 1j, 0, 0], 
                [-1j, 1, 0, 0], 
                [0, 0, 1, -1j], 
                [0, 0, 1j, 1]
            ])
        ]),
        (qml.Hermitian, [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [
            np.array([
                [1, 1j, 0, .5j], 
                [-1j, 1, 0, 0], 
                [0, 0, 1, -1j], 
                [-.5j, 0, 1j, 1]
            ])
        ]),
        (qml.Hermitian, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [
            np.array([
                [1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, -1, 0], 
                [0, 0, 0, 1]
            ])
        ]),
        (qml.Hermitian, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [
            np.array([
                [1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, -1, 0], 
                [0, 0, 0, 1]
            ])
        ]),
    ])
    # fmt: on
    def test_expval_two_wires_with_parameters(self, simulator_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        op = operation(par[0], [0, 1], do_queue=False)
        simulator_device_2_wires._obs_queue = [op]

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply("QubitStateVector", wires=[0, 1], par=[input])
        simulator_device_2_wires.post_apply()
      
        simulator_device_2_wires.pre_measure()
        res = simulator_device_2_wires.expval(op.name, wires=[0, 1], par=par)

        assert np.isclose(res, expected_output, **tol) 


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestVar:
    """Tests that variances are properly calculated."""

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output", [
        (qml.PauliX, [1/math.sqrt(2), 1/math.sqrt(2)], 0),
        (qml.PauliX, [1/math.sqrt(2), -1/math.sqrt(2)], 0),
        (qml.PauliX, [1, 0], 1),
        (qml.PauliY, [1/math.sqrt(2), 1j/math.sqrt(2)], 0),
        (qml.PauliY, [1/math.sqrt(2), -1j/math.sqrt(2)], 0),
        (qml.PauliY, [1, 0], 1),
        (qml.PauliZ, [1, 0], 0),
        (qml.PauliZ, [0, 1], 0),
        (qml.PauliZ, [1/math.sqrt(2), 1/math.sqrt(2)], 1),
        (qml.Hadamard, [1, 0], 1/2),
        (qml.Hadamard, [0, 1], 1/2),
        (qml.Hadamard, [1/math.sqrt(2), 1/math.sqrt(2)], 1/2),
    ])
    # fmt: on
    def test_var_single_wire_no_parameters(self, simulator_device_1_wire, tol, operation, input, expected_output):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        op = operation(0, do_queue=False)
        simulator_device_1_wire._obs_queue = [op]

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[input])
        simulator_device_1_wire.post_apply()
        
        simulator_device_1_wire.pre_measure()        
        res = simulator_device_1_wire.var(op.name, wires=[0], par=[])

        assert np.isclose(res, expected_output, **tol)

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Identity, [1, 0], 0, []),
        (qml.Identity, [0, 1], 0, []),
        (qml.Identity, [1/math.sqrt(2), -1/math.sqrt(2)], 0, []),
        (qml.Hermitian, [1, 0], 1, [[[1, 1j], [-1j, 1]]]),
        (qml.Hermitian, [0, 1], 1, [[[1, 1j], [-1j, 1]]]),
        (qml.Hermitian, [1/math.sqrt(2), -1/math.sqrt(2)], 1, [[[1, 1j], [-1j, 1]]]),
    ])
    # fmt: on
    def test_var_single_wire_with_parameters(self, simulator_device_1_wire, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        if par:
            op = operation(np.array(*par), 0, do_queue=False)
        else:
            op = operation(0, do_queue=False)

        simulator_device_1_wire._obs_queue = [op]

        simulator_device_1_wire.pre_apply()
        simulator_device_1_wire.apply("QubitStateVector", wires=[0], par=[input])
        simulator_device_1_wire.post_apply()

        simulator_device_1_wire.pre_measure()   
        if par:
            res = simulator_device_1_wire.var(op.name, wires=[0], par=[np.array(*par)])
        else:
            res = simulator_device_1_wire.var(op.name, wires=[0], par=[])

        assert np.isclose(res, expected_output, **tol)

    # fmt: off
    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 11/9, [[[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]]),
        (qml.Hermitian, [0, 0, 0, 1], 1, [[[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [[[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]]),
        (qml.Hermitian, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 0, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
        (qml.Hermitian, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], 0, [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]]),
    ])
    # fmt: on
    def test_var_two_wires_with_parameters(self, simulator_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        op = operation(np.array(*par), [0, 1], do_queue=False)
        simulator_device_2_wires._obs_queue = [op]

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply("QubitStateVector", wires=[0, 1], par=[input])
        simulator_device_2_wires.post_apply()
        
        simulator_device_2_wires.pre_measure()        
        res = simulator_device_2_wires.var(op.name, wires=[0, 1], par=[np.array(*par)])
        
        assert np.isclose(res, expected_output, **tol)

class TestVarEstimate:
    """Test the estimation of variances."""

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("cirq.simulator", wires=1, shots=3, analytic=False)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert var != 1.0

@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestSample:
    """Test sampling."""

    def test_sample_dimensions(self, simulator_device_2_wires):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires._obs_queue = []

        simulator_device_2_wires.apply('RX', wires=[0], par=[1.5708])
        simulator_device_2_wires.apply('RX', wires=[1], par=[1.5708])

        simulator_device_2_wires.post_apply()
        simulator_device_2_wires.pre_measure()

        simulator_device_2_wires.shots = 10
        s1 = simulator_device_2_wires.sample('PauliZ', [0], [])
        assert np.array_equal(s1.shape, (10,))

        simulator_device_2_wires.shots = 12
        s2 = simulator_device_2_wires.sample('PauliZ', [1], [])
        assert np.array_equal(s2.shape, (12,))

        simulator_device_2_wires.shots = 17
        s3 = simulator_device_2_wires.sample('CZ', [0, 1], [])
        assert np.array_equal(s3.shape, (17,))

    def test_sample_values(self, simulator_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires._obs_queue = []

        simulator_device_2_wires.apply('RX', wires=[0], par=[1.5708])

        simulator_device_2_wires.post_apply()
        simulator_device_2_wires.pre_measure()

        s1 = simulator_device_2_wires.sample('PauliZ', [0], [])

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, **tol)