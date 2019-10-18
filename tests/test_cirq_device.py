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
Unit tests for the CirqDevice class
"""
import math
from unittest.mock import MagicMock, patch

import cirq
import pennylane as qml
import pytest
import numpy as np

from pennylane_cirq.cirq_device import CirqDevice


@patch.multiple(CirqDevice, __abstractmethods__=set())
class TestCirqDeviceInit:
    """Tests the routines of the CirqDevice class."""

    def test_default_init(self):
        """Tests that the device is properly initialized."""

        dev = CirqDevice(3, 100)

        assert dev.num_wires == 3
        assert dev.shots == 100

    def test_default_init_of_qubits(self):
        """Tests the default initialization of CirqDevice.qubits."""

        dev = CirqDevice(3, 100)

        assert len(dev.qubits) == 3
        assert dev.qubits[0] == cirq.LineQubit(0)
        assert dev.qubits[1] == cirq.LineQubit(1)
        assert dev.qubits[2] == cirq.LineQubit(2)

    def test_outer_init_of_qubits(self):
        """Tests that giving qubits as parameters to CirqDevice works."""

        qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(1, 1),
        ]

        dev = CirqDevice(4, 100, qubits=qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits[0] == cirq.GridQubit(0, 0)
        assert dev.qubits[1] == cirq.GridQubit(0, 1)
        assert dev.qubits[2] == cirq.GridQubit(1, 0)
        assert dev.qubits[3] == cirq.GridQubit(1, 1)

    def test_outer_init_of_qubits_error(self):
        """Tests that giving the wrong number of qubits as parameters to CirqDevice raises an error."""

        qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(1, 1),
        ]

        with pytest.raises(
            qml.DeviceError,
            match="The number of given qubits and the specified number of wires have to match",
        ):
            dev = CirqDevice(3, 100, qubits=qubits)


@pytest.fixture(scope="function")
def cirq_device_1_wire(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(1, shots=shots)


@pytest.fixture(scope="function")
def cirq_device_2_wires(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(2, shots=shots)


@pytest.fixture(scope="function")
def cirq_device_3_wires(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(3, shots=shots)


@pytest.mark.parametrize("shots", [100])
class TestProperties:
    """Tests that the properties of the CirqDevice are correctly implemented."""

    def test_operations(self, cirq_device_1_wire):
        """Tests that the CirqDevice supports all expected operations"""

        assert cirq_device_1_wire.operations.issuperset(qml.ops.qubit.ops)

        # TODO add cirq specific operations here.

    def test_observables(self, cirq_device_1_wire):
        """Tests that the CirqDevice supports all expected observables"""

        assert cirq_device_1_wire.observables.issuperset(qml.ops.qubit.obs)


@pytest.mark.parametrize("shots", [100])
class TestOperations:
    """Tests that the CirqDevice correctly handles the requested operations."""

    def test_reset_on_empty_circuit(self, cirq_device_1_wire):
        """Tests that reset resets the internal circuit when it is not initialized."""

        assert cirq_device_1_wire.circuit is None

        cirq_device_1_wire.reset()

        # Check if circuit is an empty cirq.Circuit
        assert cirq_device_1_wire.circuit == cirq.Circuit()

    def test_reset_on_full_circuit(self, cirq_device_1_wire):
        """Tests that reset resets the internal circuit when it is filled."""

        cirq_device_1_wire.pre_apply()
        cirq_device_1_wire.apply("PauliX", [0], [])

        # Assert that the queue is filled
        assert list(cirq_device_1_wire.circuit.all_operations())

        cirq_device_1_wire.reset()

        # Assert that the queue is empty
        assert not list(cirq_device_1_wire.circuit.all_operations())

    def test_pre_apply(self, cirq_device_1_wire):
        """Tests that pre_apply calls reset."""

        cirq_device_1_wire.reset = MagicMock()

        cirq_device_1_wire.pre_apply()

        assert cirq_device_1_wire.reset.called

    # fmt: off
    @pytest.mark.parametrize("measurement_gate,expected_diagonalization", [
        (qml.PauliX(0, do_queue=False), [cirq.H]),
        (qml.PauliY(0, do_queue=False), [cirq.Z, cirq.S, cirq.H]),
        (qml.PauliZ(0, do_queue=False), []),
        (qml.Hadamard(0, do_queue=False), [cirq.Ry(-np.pi / 4)]),
    ])
    # fmt: on
    def test_pre_measure_single_wire(
        self, cirq_device_1_wire, measurement_gate, expected_diagonalization
    ):
        """Tests that the correct pre-processing is applied in pre_measure."""

        cirq_device_1_wire.reset()
        cirq_device_1_wire._obs_queue = [measurement_gate]

        cirq_device_1_wire.pre_measure()

        ops = list(cirq_device_1_wire.circuit.all_operations())

        assert len(expected_diagonalization) == len(ops)

        for i in range(len(expected_diagonalization)):
            assert ops[i] == expected_diagonalization[i].on(cirq_device_1_wire.qubits[0])

    # Note that we DO NOT expect the diagonalization matrices to be the same as you would expect
    # for the vanilla operators. This is due to the fact that the eigenvalues are listed in ascending
    # order in the backend. This means if one uses Hermitian(Z), it will actually measure -X.Z.X.
    # fmt: off
    @pytest.mark.parametrize("A,U", [
        (
            [[1, 1j], [-1j, 1]],
            [[-1 / math.sqrt(2), 1j / math.sqrt(2)], 
                [1 / math.sqrt(2), 1j / math.sqrt(2)]],
        ),
        (
            [[0, 1], [1, 0]],
            [[-1 / math.sqrt(2), 1 / math.sqrt(2)], 
                [1 / math.sqrt(2), 1 / math.sqrt(2)]],
        ),
        (
            [[0, 1j], [-1j, 0]],
            [[-1 / math.sqrt(2), 1j / math.sqrt(2)], 
                [1 / math.sqrt(2), 1j / math.sqrt(2)]],
        ),
        ([[1, 0], [0, -1]], [[0, 1], [1, 0]]),
    ])
    # fmt: on
    def test_pre_measure_single_wire_hermitian(self, cirq_device_1_wire, tol, A, U):
        """Tests that the correct pre-processing is applied in pre_measure for single wire hermitian observables."""

        cirq_device_1_wire.reset()
        cirq_device_1_wire._obs_queue = [qml.Hermitian(np.array(A), 0, do_queue=False)]

        cirq_device_1_wire.pre_measure()

        ops = list(cirq_device_1_wire.circuit.all_operations())

        assert len(ops) == 1

        print("Circuit:\n", cirq_device_1_wire.circuit)

        assert np.allclose(ops[0]._gate._matrix, np.array(U), **tol)

    # fmt: off
    @pytest.mark.parametrize("A,U", [
        ([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        [
            [0, 0, -1 / math.sqrt(2), 1 / math.sqrt(2)],
            [-1 / math.sqrt(2), 1j / math.sqrt(2), 0, 0],
            [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), 1j / math.sqrt(2), 0, 0],
        ])
    ])
    # fmt: on
    def test_pre_measure_two_wire_hermitian(self, cirq_device_2_wires, tol, A, U):
        """Tests that the correct pre-processing is applied in pre_measure for two wire hermitian observables."""

        cirq_device_2_wires.reset()
        cirq_device_2_wires._obs_queue = [qml.Hermitian(np.array(A), [0, 1], do_queue=False)]

        cirq_device_2_wires.pre_measure()

        ops = list(cirq_device_2_wires.circuit.all_operations())

        assert len(ops) == 1

        print("Circuit:\n", cirq_device_2_wires.circuit)

        assert np.allclose(ops[0]._gate._matrix, np.array(U), **tol)

    def test_hermitian_error(self, cirq_device_3_wires):
        """Tests that an error is raised for a three-qubit hermitian observable."""
        A = np.eye(6)
        cirq_device_3_wires._obs_queue = [qml.Hermitian(np.array(A), [0, 1, 2], do_queue=False)]

        with pytest.raises(
            qml.DeviceError,
            match="Cirq only supports single-qubit and two-qubit unitary gates and thus only single-qubit and two-qubit Hermitian observables.",
        ):
            cirq_device_3_wires.pre_measure()

    def test_hermitian_matrix_caching(self, cirq_device_1_wire, tol):
        """Tests that the diagonalizations in pre_measure are properly cached."""

        A = np.array([[0, 1], [-1, 0]])
        U = np.array([[-1, 1], [1, 1]]) / math.sqrt(2)
        w = np.array([-1, 1])

        cirq_device_1_wire.reset()
        cirq_device_1_wire._obs_queue = [qml.Hermitian(A, 0, do_queue=False)]

        with patch("numpy.linalg.eigh", return_value=(w, U)) as mock:
            cirq_device_1_wire.pre_measure()

            assert mock.called

            Hkey = list(cirq_device_1_wire._eigs.keys())[0]

            assert np.allclose(cirq_device_1_wire._eigs[Hkey]["eigval"], w, **tol)
            assert np.allclose(cirq_device_1_wire._eigs[Hkey]["eigvec"], U, **tol)

        with patch("numpy.linalg.eigh", return_value=(w, U)) as mock:
            cirq_device_1_wire.pre_measure()

            assert not mock.called

    # fmt: off
    @pytest.mark.parametrize(
        "gate,par,expected_cirq_gates",
        [
            ("PauliX", [], [cirq.X]),
            ("PauliY", [], [cirq.Y]),
            ("PauliZ", [], [cirq.Z]),
            ("Hadamard", [], [cirq.H]),
            ("S", [], [cirq.S]),
            ("PhaseShift", [1.4], [cirq.ZPowGate(exponent=1.4 / np.pi)]),
            ("PhaseShift", [-1.2], [cirq.ZPowGate(exponent=-1.2 / np.pi)]),
            ("PhaseShift", [2], [cirq.ZPowGate(exponent=2 / np.pi)]),
            ("RX", [1.4], [cirq.Rx(1.4)]),
            ("RX", [-1.2], [cirq.Rx(-1.2)]),
            ("RX", [2], [cirq.Rx(2)]),
            ("RY", [1.4], [cirq.Ry(1.4)]),
            ("RY", [0], [cirq.Ry(0)]),
            ("RY", [-1.3], [cirq.Ry(-1.3)]),
            ("RZ", [1.4], [cirq.Rz(1.4)]),
            ("RZ", [-1.1], [cirq.Rz(-1.1)]),
            ("RZ", [1], [cirq.Rz(1)]),
            ("Rot", [1.4, 2.3, -1.2], [cirq.Rz(1.4), cirq.Ry(2.3), cirq.Rz(-1.2)]),
            ("Rot", [1, 2, -1], [cirq.Rz(1), cirq.Ry(2), cirq.Rz(-1)]),
            ("Rot", [-1.1, 0.2, -1], [cirq.Rz(-1.1), cirq.Ry(0.2), cirq.Rz(-1)]),
            (
                "QubitUnitary",
                [np.array([[1, 0], [0, 1]])],
                [cirq.SingleQubitMatrixGate(np.array([[1, 0], [0, 1]]))],
            ),
            (
                "QubitUnitary",
                [np.array([[1, 0], [0, -1]])],
                [cirq.SingleQubitMatrixGate(np.array([[1, 0], [0, -1]]))],
            ),
            (
                "QubitUnitary",
                [np.array([[-1, 1], [1, 1]]) / math.sqrt(2)],
                [cirq.SingleQubitMatrixGate(np.array([[-1, 1], [1, 1]]) / math.sqrt(2))],
            ),
        ],
    )
    # fmt: on
    def test_apply_single_wire(self, cirq_device_1_wire, gate, par, expected_cirq_gates):
        """Tests that apply adds the correct gates to the circuit for single-qubit gates."""

        cirq_device_1_wire.reset()

        cirq_device_1_wire.apply(gate, wires=[0], par=par)

        ops = list(cirq_device_1_wire.circuit.all_operations())

        assert len(ops) == len(expected_cirq_gates)

        for i in range(len(ops)):
            assert ops[i]._gate == expected_cirq_gates[i]

    # fmt: off
    @pytest.mark.parametrize("gate,par,expected_cirq_gates", [
        ("CNOT", [], [cirq.CNOT]),
        ("SWAP", [], [cirq.SWAP]),
        ("CZ", [], [cirq.CZ]),
        ("CRX", [1.4], [cirq.ControlledGate(cirq.Rx(1.4))]),
        ("CRX", [-1.2], [cirq.ControlledGate(cirq.Rx(-1.2))]),
        ("CRX", [2], [cirq.ControlledGate(cirq.Rx(2))]),
        ("CRY", [1.4], [cirq.ControlledGate(cirq.Ry(1.4))]),
        ("CRY", [0], [cirq.ControlledGate(cirq.Ry(0))]),
        ("CRY", [-1.3], [cirq.ControlledGate(cirq.Ry(-1.3))]),
        ("CRZ", [1.4], [cirq.ControlledGate(cirq.Rz(1.4))]),
        ("CRZ", [-1.1], [cirq.ControlledGate(cirq.Rz(-1.1))]),
        ("CRZ", [1], [cirq.ControlledGate(cirq.Rz(1))]),
        ("CRot", [1.4, 2.3, -1.2],
            [
                cirq.ControlledGate(cirq.Rz(1.4)),
                cirq.ControlledGate(cirq.Ry(2.3)),
                cirq.ControlledGate(cirq.Rz(-1.2)),
            ],
        ),
        ("CRot", [1, 2, -1],
            [
                cirq.ControlledGate(cirq.Rz(1)),
                cirq.ControlledGate(cirq.Ry(2)),
                cirq.ControlledGate(cirq.Rz(-1)),
            ],
        ),
        ("CRot", [-1.1, 0.2, -1],
            [
                cirq.ControlledGate(cirq.Rz(-1.1)),
                cirq.ControlledGate(cirq.Ry(0.2)),
                cirq.ControlledGate(cirq.Rz(-1)),
            ],
        ),
        ("QubitUnitary", [np.eye(4)], [cirq.TwoQubitMatrixGate(np.eye(4))]),
        (
            "QubitUnitary",
            [np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])],
            [
                cirq.TwoQubitMatrixGate(
                    np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
                )
            ],
        ),
        (
            "QubitUnitary",
            [np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]]) / 2],
            [
                cirq.TwoQubitMatrixGate(
                    np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]]) / 2
                )
            ],
        ),
    ])
    # fmt: on
    def test_apply_two_wires(self, cirq_device_2_wires, gate, par, expected_cirq_gates):
        """Tests that apply adds the correct gates to the circuit for two-qubit gates."""

        cirq_device_2_wires.reset()

        cirq_device_2_wires.apply(gate, wires=[0, 1], par=par)

        ops = list(cirq_device_2_wires.circuit.all_operations())

        assert len(ops) == len(expected_cirq_gates)

        for i in range(len(ops)):
            assert ops[i]._gate == expected_cirq_gates[i]

    
    # @pytest.mark.parametrize("operation,par,expected_cirq_gates", [
    #     ("BasisState", [[0]]),
    #     ("BasisState", [[1]]),        
    #     ("QubitStateVector", [[1, 0]]),
    #     ("QubitStateVector", [[0, 1]]),
    #     ("QubitStateVector", [[1/math.sqrt(2), 1j/math.sqrt(2)]]),
    #     ("QubitStateVector", [[1/2, math.sqrt(3)/2]]),
    # ])
    # def test_apply_state_preparations_single_wire(self, cirq_device_1_wire, operation, par, expected_cirq_gates):
    #     """Tests that apply adds the correct gates to the circuit for state preparations."""

    #     cirq_device_1_wire.reset()

    #     cirq_device_1_wire.apply(operation, wires=[0], par=par)

    #     ops = list(cirq_device_1_wire.circuit.all_operations())

    #     assert len(ops) == len(expected_cirq_gates)

    #     for i in range(len(ops)):
    #         assert ops[i]._gate == expected_cirq_gates[i]
