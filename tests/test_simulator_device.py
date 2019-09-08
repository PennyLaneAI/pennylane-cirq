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

# TODO:
# * Test that more samples are used when n > shots for a requested sample


class TestDeviceIntegration:
    """Tests that the SimulatorDevice integrates well with PennyLane"""

    def test_device_loading(self):
        """Tests that the cirq.simulator device is properly loaded"""

        dev = qml.device("cirq.simulator", wires=2)

        assert dev.num_wires == 2
        assert dev.shots == 0
        assert dev.short_name == "cirq.simulator"

        assert isinstance(dev, SimulatorDevice)


@pytest.fixture(scope="function")
def simulator_device_1_wire():
    """A mock instance of the abstract Device class"""
    yield SimulatorDevice(1, 0)


@pytest.fixture(scope="function")
def simulator_device_2_wires():
    """A mock instance of the abstract Device class"""
    yield SimulatorDevice(2, 0)


@pytest.fixture(scope="function")
def simulator_device_3_wires():
    """A mock instance of the abstract Device class"""
    yield SimulatorDevice(3, 0)


class TestApply:
    """Tests that gates are correctly applied"""

    @pytest.mark.parametrize(
        "name,input,expected_output",
        [
            ("PauliX", [1, 0], np.array([0, 1])),
            ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
            ("PauliY", [1, 0], [0, 1j]),
            (
                "PauliY",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [-1j / math.sqrt(2), 1j / math.sqrt(2)],
            ),
            ("PauliZ", [1, 0], [1, 0]),
            ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]),
            ("Hadamard", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
            ("Hadamard", [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
        ],
    )
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
            simulator_device_1_wire.state, np.array(expected_output), atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "name,input,expected_output",
        [
            ("CNOT", [1, 0, 0, 0], [1, 0, 0, 0]),
            ("CNOT", [0, 0, 1, 0], [0, 0, 0, 1]),
            (
                "CNOT",
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
            ),
            ("SWAP", [1, 0, 0, 0], [1, 0, 0, 0]),
            ("SWAP", [0, 0, 1, 0], [0, 1, 0, 0]),
            (
                "SWAP",
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
            ),
            ("CZ", [1, 0, 0, 0], [1, 0, 0, 0]),
            ("CZ", [0, 0, 0, 1], [0, 0, 0, -1]),
            (
                "CZ",
                [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            ),
        ],
    )
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
            simulator_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "name,expected_output,par",
        [
            ("BasisState", [0, 0, 1, 0], [[1, 0]]),
            ("BasisState", [0, 0, 1, 0], [[1, 0]]),
            ("BasisState", [0, 0, 0, 1], [[1, 1]]),
            ("QubitStateVector", [0, 0, 1, 0], [[0, 0, 1, 0]]),
            ("QubitStateVector", [0, 0, 1, 0], [[0, 0, 1, 0]]),
            ("QubitStateVector", [0, 0, 0, 1], [[0, 0, 0, 1]]),
            (
                "QubitStateVector",
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [[1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)]],
            ),
            (
                "QubitStateVector",
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
                [[1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)]],
            ),
        ],
    )
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
            simulator_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "name,input,expected_output,par",
        [
            ("PhaseShift", [1, 0], [1, 0], [math.pi / 2]),
            ("PhaseShift", [0, 1], [0, 1j], [math.pi / 2]),
            (
                "PhaseShift",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), 1 / 2 + 1j / 2],
                [math.pi / 4],
            ),
            ("RX", [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
            ("RX", [1, 0], [0, -1j], [math.pi]),
            (
                "RX",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
                [math.pi / 2],
            ),
            ("RY", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
            ("RY", [1, 0], [0, 1], [math.pi]),
            ("RY", [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
            ("RZ", [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
            ("RZ", [0, 1], [0, 1j], [math.pi]),
            (
                "RZ",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [math.pi / 2],
            ),
            ("Rot", [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2, 0, 0]),
            ("Rot", [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2, 0]),
            (
                "Rot",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [0, 0, math.pi / 2],
            ),
            (
                "Rot",
                [1, 0],
                [-1j / math.sqrt(2), -1 / math.sqrt(2)],
                [math.pi / 2, -math.pi / 2, math.pi / 2],
            ),
            (
                "Rot",
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
                [-math.pi / 2, math.pi, math.pi],
            ),
            (
                "QubitUnitary",
                [1, 0],
                [1j / math.sqrt(2), 1j / math.sqrt(2)],
                [
                    np.array(
                        [
                            [1j / math.sqrt(2), 1j / math.sqrt(2)],
                            [1j / math.sqrt(2), -1j / math.sqrt(2)],
                        ]
                    )
                ],
            ),
            (
                "QubitUnitary",
                [0, 1],
                [1j / math.sqrt(2), -1j / math.sqrt(2)],
                [
                    np.array(
                        [
                            [1j / math.sqrt(2), 1j / math.sqrt(2)],
                            [1j / math.sqrt(2), -1j / math.sqrt(2)],
                        ]
                    )
                ],
            ),
            (
                "QubitUnitary",
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                [0, 1j],
                [
                    np.array(
                        [
                            [1j / math.sqrt(2), 1j / math.sqrt(2)],
                            [1j / math.sqrt(2), -1j / math.sqrt(2)],
                        ]
                    )
                ],
            ),
        ],
    )
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
            simulator_device_1_wire.state, np.array(expected_output), atol=tol, rtol=0
        )

    @pytest.mark.parametrize(
        "name,input,expected_output,par",
        [
            ("CRX", [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
            ("CRX", [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
            (
                "CRX",
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
                [math.pi / 2],
            ),
            ("CRY", [0, 0, 0, 1], [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
            ("CRY", [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
            (
                "CRY",
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [math.pi / 2],
            ),
            ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)], [math.pi / 2]),
            ("CRZ", [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
            (
                "CRZ",
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [math.pi / 2],
            ),
            (
                "CRot",
                [0, 0, 0, 1],
                [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
                [math.pi / 2, 0, 0],
            ),
            (
                "CRot",
                [0, 0, 0, 1],
                [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, math.pi / 2, 0],
            ),
            (
                "CRot",
                [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [0, 0, math.pi / 2],
            ),
            (
                "CRot",
                [0, 0, 0, 1],
                [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
                [math.pi / 2, -math.pi / 2, math.pi / 2],
            ),
            (
                "CRot",
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
                [-math.pi / 2, math.pi, math.pi],
            ),
            (
                "QubitUnitary",
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                            [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                ],
            ),
            (
                "QubitUnitary",
                [0, 1, 0, 0],
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                            [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                ],
            ),
            (
                "QubitUnitary",
                [1 / 2, 1 / 2, -1 / 2, 1 / 2],
                [1 / 2, 0, 1 / math.sqrt(2), 1 / 2],
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                            [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                ],
            ),
        ],
    )
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
            simulator_device_2_wires.state, np.array(expected_output), atol=tol, rtol=0
        )

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
    def test_state_preparation_error(self, simulator_device_1_wire, operation, par, match):
        """Tests that the state preparation routines raise proper errors for wrong parameter values."""

        simulator_device_1_wire._obs_queue = []

        simulator_device_1_wire.pre_apply()
        
        with pytest.raises(qml.DeviceError, match=match):
            simulator_device_1_wire.apply(operation, wires=[0], par=par)

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

        assert np.isclose(res, expected_output, atol=tol, rtol=0) 

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

        assert np.isclose(res, expected_output, atol=tol, rtol=0) 

    # fmt: off
    """
    @pytest.mark.parametrize("operation,input,expected_output,par", [
        (qml.Hermitian, [1/math.sqrt(3), 0, 1/math.sqrt(3), 1/math.sqrt(3)], 5/3, [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])]),
        (qml.Hermitian, [0, 0, 0, 1], 0, [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])]),
        (qml.Hermitian, [1/math.sqrt(2), 0, -1/math.sqrt(2), 0], 1, [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])]),
        (qml.Hermitian, [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(6), 1/math.sqrt(6)], 1, [np.array([[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]])]),
        (qml.Hermitian, [1/math.sqrt(2), 0, 0, 1/math.sqrt(2)], 1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
        (qml.Hermitian, [0, 1/math.sqrt(2), -1/math.sqrt(2), 0], -1, [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
    ])
    """
    # fmt: on
    """
    def test_expval_two_wires_with_parameters(self, simulator_device_2_wires, tol, operation, input, expected_output, par):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        op = operation(par[0], [0, 1], do_queue=False)
        simulator_device_2_wires._obs_queue = [op]

        simulator_device_2_wires.pre_apply()
        simulator_device_2_wires.apply("QubitStateVector", wires=[0, 1], par=[input])
        simulator_device_2_wires.post_apply()
        
        simulator_device_2_wires.pre_measure()
        res = simulator_device_2_wires.expval(op.name, wires=[0, 1], par=par)

        assert np.isclose(res, expected_output, atol=tol, rtol=0) 
    """