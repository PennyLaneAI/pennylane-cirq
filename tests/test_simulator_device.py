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
Unit tests for the SimulatorDevice
"""
import pytest
import math
from unittest.mock import MagicMock, patch
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
        assert not dev.shots
        assert dev.short_name == "cirq.simulator"

        assert isinstance(dev, SimulatorDevice)

    def test_custom_simulator(self):
        """Test that a custom cirq simulator can be used with the cirq device."""
        sim = cirq.Simulator()
        dev = qml.device("cirq.simulator", wires=1, shots=None, simulator=sim)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.expval(qml.PauliX(0))

        assert circuit() == 0.0

    @pytest.mark.parametrize(
        "operation,expected_output",
        [
            (qml.PauliX, -0.95105648),
            (qml.PauliY, -0.95105648),
            (qml.PauliZ, 1),
            (qml.Hadamard, 0.02447175),
        ],
    )
    def test_native_power_support_single_wire(self, operation, expected_output):
        """Test that supported one-wire operators can be raised to a power on a cirq device."""
        dev = qml.device("cirq.simulator", wires=1)

        @qml.qnode(dev)
        def circuit():
            operation(wires=[0]) ** 1.1
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output)

    @pytest.mark.parametrize(
        "operation,expected_output",
        [
            (qml.SWAP, 0.0244717),
            (qml.ISWAP, 0.02447164),
            (qml.CNOT, 0.0244717),
            (qml.CZ, 1),
        ],
    )
    def test_native_power_support_two_wires(self, operation, expected_output):
        """Test that supported two-wire operators can be raised to a power on a cirq device."""
        dev = qml.device("cirq.simulator", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            operation(wires=[0, 1]) ** 1.1
            return qml.expval(qml.PauliZ(1))

        assert np.isclose(circuit(), expected_output)


@pytest.fixture(scope="function")
def simulator_device_1_wire(shots):
    """Return a single wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(1, shots=shots)


@pytest.fixture(scope="function")
def simulator_device_2_wires(shots):
    """Return a two wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(2, shots=shots)


@pytest.fixture(scope="function")
def simulator_device_3_wires(shots):
    """Return a three wire instance of the SimulatorDevice class."""
    yield SimulatorDevice(3, shots=shots)


@pytest.mark.parametrize("shots", [None])
class TestApply:
    """Tests that gates are correctly applied"""

    @pytest.mark.parametrize(
        "op,input,expected_output",
        [
            (qml.PauliX, [1, 0], np.array([0, 1])),
            (
                qml.PauliX,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
            ),
            (qml.PauliY, [1, 0], [0, 1j]),
            (
                qml.PauliY,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [-1j / math.sqrt(2), 1j / math.sqrt(2)],
            ),
            (qml.PauliZ, [1, 0], [1, 0]),
            (
                qml.PauliZ,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
            ),
            (qml.Hadamard, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
            (qml.Hadamard, [1 / math.sqrt(2), -1 / math.sqrt(2)], [0, 1]),
        ],
    )
    def test_apply_operation_single_wire_no_parameters(
        self, simulator_device_1_wire, tol, op, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([op(wires=[0])])

        assert np.allclose(simulator_device_1_wire.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,input,expected_output",
        [
            (qml.CNOT, [1, 0, 0, 0], [1, 0, 0, 0]),
            (qml.CNOT, [0, 0, 1, 0], [0, 0, 0, 1]),
            (
                qml.CNOT,
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                [1 / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
            ),
            (qml.SWAP, [1, 0, 0, 0], [1, 0, 0, 0]),
            (qml.SWAP, [0, 0, 1, 0], [0, 1, 0, 0]),
            (
                qml.SWAP,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                [1 / math.sqrt(2), -1 / math.sqrt(2), 0, 0],
            ),
            (qml.CZ, [1, 0, 0, 0], [1, 0, 0, 0]),
            (qml.CZ, [0, 0, 0, 1], [0, 0, 0, -1]),
            (
                qml.CZ,
                [1 / math.sqrt(2), 0, 0, -1 / math.sqrt(2)],
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
            ),
        ],
    )
    def test_apply_operation_two_wires_no_parameters(
        self, simulator_device_2_wires, tol, op, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have no parameters."""

        simulator_device_2_wires.reset()
        simulator_device_2_wires._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_2_wires.apply([op(wires=[0, 1])])

        assert np.allclose(simulator_device_2_wires.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,expected_output,par",
        [
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 0, 1], [1, 1]),
            (qml.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.StatePrep, [0, 0, 1, 0], [0, 0, 1, 0]),
            (qml.StatePrep, [0, 0, 0, 1], [0, 0, 0, 1]),
            (
                qml.StatePrep,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
            (
                qml.StatePrep,
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
                [1 / math.sqrt(3), 0, -1 / math.sqrt(3), 1 / math.sqrt(3)],
            ),
        ],
    )
    def test_apply_operation_state_preparation(
        self, simulator_device_2_wires, tol, op, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply([op(np.array(par), wires=[0, 1])])

        assert np.allclose(simulator_device_2_wires.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,input,expected_output,par",
        [
            (qml.PhaseShift, [1, 0], [1, 0], [math.pi / 2]),
            (qml.PhaseShift, [0, 1], [0, 1j], [math.pi / 2]),
            (
                qml.PhaseShift,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / math.sqrt(2), 1 / 2 + 1j / 2],
                [math.pi / 4],
            ),
            (qml.RX, [1, 0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2]),
            (qml.RX, [1, 0], [0, -1j], [math.pi]),
            (
                qml.RX,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 - 1j / 2],
                [math.pi / 2],
            ),
            (qml.RY, [1, 0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2]),
            (qml.RY, [1, 0], [0, 1], [math.pi]),
            (qml.RY, [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 1], [math.pi / 2]),
            (qml.RZ, [1, 0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2]),
            (qml.RZ, [0, 1], [0, 1j], [math.pi]),
            (
                qml.RZ,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [math.pi / 2],
            ),
            (
                qml.Rot,
                [1, 0],
                [1 / math.sqrt(2) - 1j / math.sqrt(2), 0],
                [math.pi / 2, 0, 0],
            ),
            (
                qml.Rot,
                [1, 0],
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, math.pi / 2, 0],
            ),
            (
                qml.Rot,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [0, 0, math.pi / 2],
            ),
            (
                qml.Rot,
                [1, 0],
                [-1j / math.sqrt(2), -1 / math.sqrt(2)],
                [math.pi / 2, -math.pi / 2, math.pi / 2],
            ),
            (
                qml.Rot,
                [1 / math.sqrt(2), 1 / math.sqrt(2)],
                [1 / 2 + 1j / 2, -1 / 2 + 1j / 2],
                [-math.pi / 2, math.pi, math.pi],
            ),
            (
                qml.QubitUnitary,
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
                qml.QubitUnitary,
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
                qml.QubitUnitary,
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
        self, simulator_device_1_wire, tol, op, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([op(*par, wires=[0])])

        assert np.allclose(simulator_device_1_wire.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,input,expected_output,par",
        [
            (qml.CRX, [0, 1, 0, 0], [0, 1, 0, 0], [math.pi / 2]),
            (qml.CRX, [0, 0, 0, 1], [0, 0, -1j, 0], [math.pi]),
            (
                qml.CRX,
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), 1 / 2, -1j / 2],
                [math.pi / 2],
            ),
            (
                qml.CRY,
                [0, 0, 0, 1],
                [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)],
                [math.pi / 2],
            ),
            (qml.CRY, [0, 0, 0, 1], [0, 0, -1, 0], [math.pi]),
            (
                qml.CRY,
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [math.pi / 2],
            ),
            (
                qml.CRZ,
                [0, 0, 0, 1],
                [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
                [math.pi / 2],
            ),
            (qml.CRZ, [0, 0, 0, 1], [0, 0, 0, 1j], [math.pi]),
            (
                qml.CRZ,
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
                [math.pi / 2],
            ),
            (
                qml.CRot,
                [0, 0, 0, 1],
                [0, 0, 0, 1 / math.sqrt(2) + 1j / math.sqrt(2)],
                [math.pi / 2, 0, 0],
            ),
            (
                qml.CRot,
                [0, 0, 0, 1],
                [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, math.pi / 2, 0],
            ),
            (
                qml.CRot,
                [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
                [0, 0, 1 / 2 - 1j / 2, 1 / 2 + 1j / 2],
                [0, 0, math.pi / 2],
            ),
            (
                qml.CRot,
                [0, 0, 0, 1],
                [0, 0, 1 / math.sqrt(2), 1j / math.sqrt(2)],
                [math.pi / 2, -math.pi / 2, math.pi / 2],
            ),
            (
                qml.CRot,
                [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
                [0, 1 / math.sqrt(2), 0, -1 / 2 + 1j / 2],
                [-math.pi / 2, math.pi, math.pi],
            ),
            (
                qml.QubitUnitary,
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
                qml.QubitUnitary,
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
                qml.QubitUnitary,
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
        self, simulator_device_2_wires, tol, op, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have no parameters."""

        simulator_device_2_wires.reset()
        simulator_device_2_wires._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_2_wires.apply([op(*par, wires=[0, 1])])

        assert np.allclose(simulator_device_2_wires.state, np.array(expected_output), **tol)

    def test_basis_state_not_at_beginning_error(self, simulator_device_1_wire):
        """Tests that application of BasisState raises an error if is not
        the first operation."""

        simulator_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError,
            match="The operation BasisState is only supported at the beginning of a circuit.",
        ):
            simulator_device_1_wire.apply([qml.PauliX(0), qml.BasisState(np.array([0]), wires=[0])])

    def test_qubit_state_vector_not_at_beginning_error(self, simulator_device_1_wire):
        """Tests that application of StatePrep raises an error if is not
        the first operation."""

        simulator_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError,
            match=f"The operation StatePrep is only supported at the beginning of a circuit.",
        ):
            simulator_device_1_wire.apply([qml.PauliX(0), qml.StatePrep(np.array([0, 1]), wires=[0])])


@pytest.mark.parametrize("shots", [1000])
class TestStatePreparationErrorsNonAnalytic:
    """Tests state preparation errors that occur for non-analytic devices."""

    def test_basis_state_not_analytic_error(self, simulator_device_1_wire):
        """Tests that application of BasisState raises an error if the device
        is not in analytic mode."""

        simulator_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError,
            match="The operation BasisState is only supported in analytic mode.",
        ):
            simulator_device_1_wire.apply([qml.BasisState(np.array([0]), wires=[0])])

    def test_qubit_state_vector_not_analytic_error(self, simulator_device_1_wire):
        """Tests that application of StatePrep raises an error if the device
        is not in analytic mode."""

        simulator_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError,
            match="The operator StatePrep is only supported in analytic mode.",
        ):
            simulator_device_1_wire.apply([qml.StatePrep(np.array([0, 1]), wires=[0])])


@pytest.mark.parametrize("shots", [None])
class TestAnalyticProbability:
    """Tests the analytic_probability method works as expected."""

    def test_analytic_probability_is_none(self, simulator_device_1_wire):
        """Tests that analytic_probability returns None if the state of the
        device is None."""

        simulator_device_1_wire.reset()
        assert simulator_device_1_wire._state is None
        assert simulator_device_1_wire.analytic_probability() is None


@pytest.mark.parametrize("shots", [None])
class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.Identity, [1, 0], 1),
            (qml.Identity, [0, 1], 1),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 1),
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
            (qml.PauliX, [1, 0], 0),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
            (qml.PauliY, [1, 0], 0),
            (qml.PauliZ, [1, 0], 1),
            (qml.PauliZ, [0, 1], -1),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.Hadamard, [1, 0], 1 / math.sqrt(2)),
            (qml.Hadamard, [0, 1], -1 / math.sqrt(2)),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
        ],
    )
    def test_expval_single_wire_no_parameters(
        self, simulator_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        op = operation(0)

        simulator_device_1_wire.reset()
        simulator_device_1_wire.apply(
            [qml.StatePrep(np.array(input), wires=[0])], rotations=op.diagonalizing_gates()
        )

        res = simulator_device_1_wire.expval(op)

        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Hermitian, [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
            (qml.Hermitian, [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                1,
                [np.array([[1, 1j], [-1j, 1]])],
            ),
        ],
    )
    def test_expval_single_wire_with_parameters(
        self, simulator_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        op = operation(par[0], 0)

        simulator_device_1_wire.reset()
        simulator_device_1_wire.apply(
            [qml.StatePrep(np.array(input), wires=[0])], rotations=op.diagonalizing_gates()
        )

        res = simulator_device_1_wire.expval(op)

        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [0, 1, 0, 0],
                -1,
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                ],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                5 / 3,
                [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])],
            ),
            (
                qml.Hermitian,
                [0, 0, 0, 1],
                0,
                [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                1,
                [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])],
            ),
            (
                qml.Hermitian,
                [
                    1 / math.sqrt(3),
                    -1 / math.sqrt(3),
                    1 / math.sqrt(6),
                    1 / math.sqrt(6),
                ],
                1,
                [
                    np.array(
                        [
                            [1, 1j, 0, 0.5j],
                            [-1j, 1, 0, 0],
                            [0, 0, 1, -1j],
                            [-0.5j, 0, 1j, 1],
                        ]
                    )
                ],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                1,
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
            (
                qml.Hermitian,
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                -1,
                [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])],
            ),
        ],
    )
    def test_expval_two_wires_with_parameters(
        self, simulator_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for two-wire observables with parameters."""

        op = operation(par[0], [0, 1])

        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply(
            [qml.StatePrep(np.array(input), wires=[0, 1])],
            rotations=op.diagonalizing_gates(),
        )

        res = simulator_device_2_wires.expval(op)

        assert np.isclose(res, expected_output, **tol)


@pytest.mark.parametrize("shots", [None])
class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
            (qml.PauliX, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0),
            (qml.PauliX, [1, 0], 1),
            (qml.PauliY, [1 / math.sqrt(2), 1j / math.sqrt(2)], 0),
            (qml.PauliY, [1 / math.sqrt(2), -1j / math.sqrt(2)], 0),
            (qml.PauliY, [1, 0], 1),
            (qml.PauliZ, [1, 0], 0),
            (qml.PauliZ, [0, 1], 0),
            (qml.PauliZ, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
            (qml.Hadamard, [1, 0], 1 / 2),
            (qml.Hadamard, [0, 1], 1 / 2),
            (qml.Hadamard, [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / 2),
        ],
    )
    def test_var_single_wire_no_parameters(
        self, simulator_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        op = operation(0)

        simulator_device_1_wire.reset()
        simulator_device_1_wire.apply(
            [qml.StatePrep(np.array(input), wires=[0])],
            rotations=op.diagonalizing_gates(),
        )

        res = simulator_device_1_wire.var(op)

        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Identity, [1, 0], 0, []),
            (qml.Identity, [0, 1], 0, []),
            (qml.Identity, [1 / math.sqrt(2), -1 / math.sqrt(2)], 0, []),
            (qml.Hermitian, [1, 0], 1, [[[1, 1j], [-1j, 1]]]),
            (qml.Hermitian, [0, 1], 1, [[[1, 1j], [-1j, 1]]]),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), -1 / math.sqrt(2)],
                1,
                [[[1, 1j], [-1j, 1]]],
            ),
        ],
    )
    def test_var_single_wire_with_parameters(
        self, simulator_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        if par:
            op = operation(np.array(*par), 0)
        else:
            op = operation(0)

        simulator_device_1_wire.reset()
        simulator_device_1_wire.apply(
            [qml.StatePrep(np.array(input), wires=[0])],
            rotations=op.diagonalizing_gates(),
        )

        res = simulator_device_1_wire.var(op)
        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (
                qml.Hermitian,
                [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)],
                11 / 9,
                [[[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]]],
            ),
            (
                qml.Hermitian,
                [0, 0, 0, 1],
                1,
                [[[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0],
                1,
                [[[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]]],
            ),
            (
                qml.Hermitian,
                [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)],
                0,
                [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]],
            ),
            (
                qml.Hermitian,
                [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
                0,
                [[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]],
            ),
        ],
    )
    def test_var_two_wires_with_parameters(
        self, simulator_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that variances are properly calculated for two-wire observables with parameters."""

        op = operation(np.array(*par), [0, 1])

        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply(
            [qml.StatePrep(np.array(input), wires=[0, 1])],
            rotations=op.diagonalizing_gates(),
        )

        res = simulator_device_2_wires.var(op)

        assert np.isclose(res, expected_output, **tol)


class TestVarEstimate:
    """Test the estimation of variances."""

    def test_var_estimate(self):
        """Test that the variance is not analytically calculated"""

        dev = qml.device("cirq.simulator", wires=1, shots=3)

        @qml.qnode(dev)
        def circuit():
            return qml.var(qml.PauliX(0))

        var = circuit()

        # With 3 samples we are guaranteed to see a difference between
        # an estimated variance an an analytically calculated one
        assert var != 1.0


@pytest.mark.parametrize("shots", [8192])
class TestSample:
    """Test sampling."""

    @pytest.mark.parametrize(
        "new_shots,obs",
        [
            (10, qml.PauliZ(0)),
            (12, qml.PauliZ(1)),
            (17, qml.Hermitian(np.diag([1, 1, 1, -1]), wires=[0, 1])),
        ],
    )
    def test_sample_dimensions(self, simulator_device_2_wires, new_shots, obs):
        """Tests if the samples returned by the sample function have
        the correct dimensions
        """
        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

        simulator_device_2_wires.shots = new_shots
        simulator_device_2_wires._samples = simulator_device_2_wires.generate_samples()
        s1 = simulator_device_2_wires.sample(obs)
        assert np.array_equal(s1.shape, (new_shots,))

    def test_sample_values(self, simulator_device_2_wires, tol):
        """Tests if the samples returned by sample have
        the correct values
        """

        simulator_device_2_wires.reset()

        simulator_device_2_wires.apply([qml.RX(1.5708, wires=[0])])
        simulator_device_2_wires._samples = simulator_device_2_wires.generate_samples()

        s1 = simulator_device_2_wires.sample(qml.PauliZ(0))

        # s1 should only contain 1 and -1, which is guaranteed if
        # they square to 1
        assert np.allclose(s1**2, 1, **tol)


class TestState:
    """Test the state property."""

    @pytest.mark.parametrize("shots", [None])
    @pytest.mark.parametrize(
        "ops,expected_state",
        [
            ([qml.PauliX(0), qml.PauliX(1)], [0, 0, 0, 1]),
            ([qml.PauliX(0), qml.PauliY(1)], [0, 0, 0, 1j]),
            ([qml.PauliZ(0), qml.PauliZ(1)], [1, 0, 0, 0]),
        ],
    )
    def test_state_pauli_operations(self, simulator_device_2_wires, ops, expected_state, tol):
        """Test that the state reflects Pauli operations correctly."""
        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply(ops)

        assert np.allclose(simulator_device_2_wires.state, expected_state, **tol)

    @pytest.mark.parametrize("shots", [None])
    @pytest.mark.parametrize(
        "ops,diag_ops,expected_state",
        [
            ([qml.PauliX(0), qml.PauliX(1)], [], [0, 0, 0, 1]),
            (
                [qml.PauliX(0), qml.PauliY(1)],
                [qml.Hadamard(0)],
                [0, 1j / np.sqrt(2), 0, -1j / np.sqrt(2)],
            ),
            (
                [qml.PauliZ(0), qml.PauliZ(1)],
                [qml.Hadamard(1)],
                [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
            ),
        ],
    )
    def test_state_pauli_operations_and_observables(
        self, simulator_device_2_wires, ops, diag_ops, expected_state, tol
    ):
        """Test that the state reflects Pauli operations and observable rotations correctly."""
        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply(ops, rotations=diag_ops)

        assert np.allclose(simulator_device_2_wires.state, expected_state, **tol)

    @pytest.mark.parametrize("shots", [100])
    def test_state_non_analytic(self, simulator_device_2_wires):
        """Test that the state is None if in non-analytic mode."""
        simulator_device_2_wires.reset()
        simulator_device_2_wires.apply([qml.PauliX(0), qml.PauliX(1)])

        assert simulator_device_2_wires.state is None
