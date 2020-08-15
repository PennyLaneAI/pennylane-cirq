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
from pennylane.wires import Wires
import pytest
import numpy as np

from pennylane_cirq.cirq_device import CirqDevice


@patch.multiple(CirqDevice, __abstractmethods__=set())
class TestCirqDeviceInit:
    """Tests the routines of the CirqDevice class."""

    @pytest.mark.parametrize("analytic", [True, False])
    @pytest.mark.parametrize("num_wires", [1, 3, 6])
    @pytest.mark.parametrize("shots", [1, 100, 137])
    def test_default_init(self, analytic, num_wires, shots):
        """Tests that the device is properly initialized."""

        dev = CirqDevice(num_wires, shots, analytic)

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev.analytic == analytic

    def test_default_init_of_qubits(self):
        """Tests the default initialization of CirqDevice.qubits."""

        dev = CirqDevice(3, 100, False)

        assert len(dev.qubits) == 3
        assert dev.qubits[0] == cirq.LineQubit(0)
        assert dev.qubits[1] == cirq.LineQubit(1)
        assert dev.qubits[2] == cirq.LineQubit(2)

    def test_outer_init_of_qubits_ordered(self):
        """Tests that giving qubits as parameters to CirqDevice works when the qubits are already ordered consistently with Cirq's convention."""

        qubits = [
            cirq.GridQubit(0, 0),
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(1, 1),
        ]

        dev = CirqDevice(4, 100, False, qubits=qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits == qubits

    def test_outer_init_of_qubits_unordered(self):
        """Tests that giving qubits as parameters to CirqDevice works when the qubits are not ordered consistently with Cirq's convention."""

        qubits = [
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(0, 0),
            cirq.GridQubit(1, 1),
        ]

        dev = CirqDevice(4, 100, False, qubits=qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits == sorted(qubits)

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
            dev = CirqDevice(3, 100, False, qubits=qubits)


class TestCirqDeviceIntegration:
    """Integration tests for Cirq devices"""

    def test_outer_init_of_qubits_with_wire_number(self):
        """Tests that giving qubits as parameters to CirqDevice works when the user provides a number of wires."""

        unordered_qubits = [
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(0, 0),
            cirq.GridQubit(1, 1),
        ]

        dev = qml.device("cirq.simulator", wires=4, qubits=unordered_qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits == sorted(unordered_qubits)

    def test_outer_init_of_qubits_with_wire_label_strings(self):
        """Tests that giving qubits as parameters to CirqDevice works when the user also provides custom string wire labels."""

        unordered_qubits = [
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(0, 0),
            cirq.GridQubit(1, 1),
        ]

        user_labels = ["alice", "bob", "charlie", "david"]
        sort_order = [2,0,1,3]

        dev = qml.device("cirq.simulator", wires=user_labels, qubits=unordered_qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits == sorted(unordered_qubits)
        assert all(dev.map_wires(Wires(label)) == Wires(idx) for label, idx in zip(user_labels, sort_order))

    def test_outer_init_of_qubits_with_wire_label_ints(self):
        """Tests that giving qubits as parameters to CirqDevice works when the user also provides custom integer wire labels."""

        unordered_qubits = [
            cirq.GridQubit(0, 1),
            cirq.GridQubit(1, 0),
            cirq.GridQubit(0, 0),
            cirq.GridQubit(1, 1),
        ]

        user_labels = [-1,1,66,0]
        sort_order = [2,0,1,3]

        dev = qml.device("cirq.simulator", wires=user_labels, qubits=unordered_qubits)
        assert len(dev.qubits) == 4
        assert dev.qubits == sorted(unordered_qubits)
        assert all(dev.map_wires(Wires(label)) == Wires(idx) for label, idx in zip(user_labels, sort_order))


@pytest.fixture(scope="function")
def cirq_device_1_wire(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(1, shots=shots, analytic=True)


@pytest.fixture(scope="function")
def cirq_device_2_wires(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(2, shots=shots, analytic=True)


@pytest.fixture(scope="function")
def cirq_device_3_wires(shots):
    """A mock instance of the abstract Device class"""

    with patch.multiple(CirqDevice, __abstractmethods__=set()):
        yield CirqDevice(3, shots=shots, analytic=True)


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

        cirq_device_1_wire.reset()
        cirq_device_1_wire.apply([qml.PauliX(0)])

        # Assert that the queue is filled
        assert list(cirq_device_1_wire.circuit.all_operations())

        cirq_device_1_wire.reset()

        # Assert that the queue is empty
        assert not list(cirq_device_1_wire.circuit.all_operations())

    @pytest.mark.parametrize(
        "gate,expected_cirq_gates",
        [
            (qml.PauliX(wires=[0]), [cirq.X]),
            (qml.PauliY(wires=[0]), [cirq.Y]),
            (qml.PauliZ(wires=[0]), [cirq.Z]),
            (qml.PauliX(wires=[0]).inv(), [cirq.X ** -1]),
            (qml.PauliY(wires=[0]).inv(), [cirq.Y ** -1]),
            (qml.PauliZ(wires=[0]).inv(), [cirq.Z ** -1]),
            (qml.Hadamard(wires=[0]), [cirq.H]),
            (qml.Hadamard(wires=[0]).inv(), [cirq.H ** -1]),
            (qml.S(wires=[0]), [cirq.S]),
            (qml.S(wires=[0]).inv(), [cirq.S ** -1]),
            (qml.PhaseShift(1.4, wires=[0]), [cirq.ZPowGate(exponent=1.4 / np.pi)]),
            (qml.PhaseShift(-1.2, wires=[0]), [cirq.ZPowGate(exponent=-1.2 / np.pi)]),
            (qml.PhaseShift(2, wires=[0]), [cirq.ZPowGate(exponent=2 / np.pi)]),
            (qml.PhaseShift(1.4, wires=[0]).inv(), [cirq.ZPowGate(exponent=-1.4 / np.pi)],),
            (qml.PhaseShift(-1.2, wires=[0]).inv(), [cirq.ZPowGate(exponent=1.2 / np.pi)],),
            (qml.PhaseShift(2, wires=[0]).inv(), [cirq.ZPowGate(exponent=-2 / np.pi)]),
            (qml.RX(1.4, wires=[0]), [cirq.rx(1.4)]),
            (qml.RX(-1.2, wires=[0]), [cirq.rx(-1.2)]),
            (qml.RX(2, wires=[0]), [cirq.rx(2)]),
            (qml.RX(1.4, wires=[0]).inv(), [cirq.rx(-1.4)]),
            (qml.RX(-1.2, wires=[0]).inv(), [cirq.rx(1.2)]),
            (qml.RX(2, wires=[0]).inv(), [cirq.rx(-2)]),
            (qml.RY(1.4, wires=[0]), [cirq.ry(1.4)]),
            (qml.RY(0, wires=[0]), [cirq.ry(0)]),
            (qml.RY(-1.3, wires=[0]), [cirq.ry(-1.3)]),
            (qml.RY(1.4, wires=[0]).inv(), [cirq.ry(-1.4)]),
            (qml.RY(0, wires=[0]).inv(), [cirq.ry(0)]),
            (qml.RY(-1.3, wires=[0]).inv(), [cirq.ry(+1.3)]),
            (qml.RZ(1.4, wires=[0]), [cirq.rz(1.4)]),
            (qml.RZ(-1.1, wires=[0]), [cirq.rz(-1.1)]),
            (qml.RZ(1, wires=[0]), [cirq.rz(1)]),
            (qml.RZ(1.4, wires=[0]).inv(), [cirq.rz(-1.4)]),
            (qml.RZ(-1.1, wires=[0]).inv(), [cirq.rz(1.1)]),
            (qml.RZ(1, wires=[0]).inv(), [cirq.rz(-1)]),
            (qml.Rot(1.4, 2.3, -1.2, wires=[0]), [cirq.rz(1.4), cirq.ry(2.3), cirq.rz(-1.2)],),
            (qml.Rot(1, 2, -1, wires=[0]), [cirq.rz(1), cirq.ry(2), cirq.rz(-1)]),
            (qml.Rot(-1.1, 0.2, -1, wires=[0]), [cirq.rz(-1.1), cirq.ry(0.2), cirq.rz(-1)],),
            (
                qml.Rot(1.4, 2.3, -1.2, wires=[0]).inv(),
                [cirq.rz(1.2), cirq.ry(-2.3), cirq.rz(-1.4)],
            ),
            (qml.Rot(1, 2, -1, wires=[0]).inv(), [cirq.rz(1), cirq.ry(-2), cirq.rz(-1)],),
            (qml.Rot(-1.1, 0.2, -1, wires=[0]).inv(), [cirq.rz(1), cirq.ry(-0.2), cirq.rz(1.1)],),
            (
                qml.QubitUnitary(np.array([[1, 0], [0, 1]]), wires=[0]),
                [cirq.MatrixGate(np.array([[1, 0], [0, 1]]))],
            ),
            (
                qml.QubitUnitary(np.array([[1, 0], [0, -1]]), wires=[0]),
                [cirq.MatrixGate(np.array([[1, 0], [0, -1]]))],
            ),
            (
                qml.QubitUnitary(np.array([[-1, 1], [1, 1]]) / math.sqrt(2), wires=[0]),
                [cirq.MatrixGate(np.array([[-1, 1], [1, 1]]) / math.sqrt(2))],
            ),
            (
                qml.QubitUnitary(np.array([[1, 0], [0, 1]]), wires=[0]).inv(),
                [cirq.MatrixGate(np.array([[1, 0], [0, 1]])) ** -1],
            ),
            (
                qml.QubitUnitary(np.array([[1, 0], [0, -1]]), wires=[0]).inv(),
                [cirq.MatrixGate(np.array([[1, 0], [0, -1]])) ** -1],
            ),
            (
                qml.QubitUnitary(np.array([[-1, 1], [1, 1]]) / math.sqrt(2), wires=[0]).inv(),
                [cirq.MatrixGate(np.array([[-1, 1], [1, 1]]) / math.sqrt(2)) ** -1],
            ),
        ],
    )
    def test_apply_single_wire(self, cirq_device_1_wire, gate, expected_cirq_gates):
        """Tests that apply adds the correct gates to the circuit for single-qubit gates."""

        cirq_device_1_wire.reset()

        cirq_device_1_wire.apply([gate])

        ops = list(cirq_device_1_wire.circuit.all_operations())

        assert len(ops) == len(expected_cirq_gates)

        for i in range(len(ops)):
            assert ops[i]._gate == expected_cirq_gates[i]

    @pytest.mark.parametrize(
        "gate,expected_cirq_gates",
        [
            (qml.CNOT(wires=[0, 1]), [cirq.CNOT]),
            (qml.CNOT(wires=[0, 1]).inv(), [cirq.CNOT ** -1]),
            (qml.SWAP(wires=[0, 1]), [cirq.SWAP]),
            (qml.SWAP(wires=[0, 1]).inv(), [cirq.SWAP ** -1]),
            (qml.CZ(wires=[0, 1]), [cirq.CZ]),
            (qml.CZ(wires=[0, 1]).inv(), [cirq.CZ ** -1]),
            (qml.CRX(1.4, wires=[0, 1]), [cirq.ControlledGate(cirq.rx(1.4))]),
            (qml.CRX(-1.2, wires=[0, 1]), [cirq.ControlledGate(cirq.rx(-1.2))]),
            (qml.CRX(2, wires=[0, 1]), [cirq.ControlledGate(cirq.rx(2))]),
            (qml.CRX(1.4, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rx(-1.4))]),
            (qml.CRX(-1.2, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rx(1.2))]),
            (qml.CRX(2, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rx(-2))]),
            (qml.CRY(1.4, wires=[0, 1]), [cirq.ControlledGate(cirq.ry(1.4))]),
            (qml.CRY(0, wires=[0, 1]), [cirq.ControlledGate(cirq.ry(0))]),
            (qml.CRY(-1.3, wires=[0, 1]), [cirq.ControlledGate(cirq.ry(-1.3))]),
            (qml.CRY(1.4, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.ry(-1.4))]),
            (qml.CRY(0, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.ry(0))]),
            (qml.CRY(-1.3, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.ry(1.3))]),
            (qml.CRZ(1.4, wires=[0, 1]), [cirq.ControlledGate(cirq.rz(1.4))]),
            (qml.CRZ(-1.1, wires=[0, 1]), [cirq.ControlledGate(cirq.rz(-1.1))]),
            (qml.CRZ(1, wires=[0, 1]), [cirq.ControlledGate(cirq.rz(1))]),
            (qml.CRZ(1.4, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rz(-1.4))]),
            (qml.CRZ(-1.1, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rz(1.1))]),
            (qml.CRZ(1, wires=[0, 1]).inv(), [cirq.ControlledGate(cirq.rz(-1))]),
            (
                qml.CRot(1.4, 2.3, -1.2, wires=[0, 1]),
                [
                    cirq.ControlledGate(cirq.rz(1.4)),
                    cirq.ControlledGate(cirq.ry(2.3)),
                    cirq.ControlledGate(cirq.rz(-1.2)),
                ],
            ),
            (
                qml.CRot(1, 2, -1, wires=[0, 1]),
                [
                    cirq.ControlledGate(cirq.rz(1)),
                    cirq.ControlledGate(cirq.ry(2)),
                    cirq.ControlledGate(cirq.rz(-1)),
                ],
            ),
            (
                qml.CRot(-1.1, 0.2, -1, wires=[0, 1]),
                [
                    cirq.ControlledGate(cirq.rz(-1.1)),
                    cirq.ControlledGate(cirq.ry(0.2)),
                    cirq.ControlledGate(cirq.rz(-1)),
                ],
            ),
            (
                qml.CRot(1.4, 2.3, -1.2, wires=[0, 1]).inv(),
                [
                    cirq.ControlledGate(cirq.rz(1.2)),
                    cirq.ControlledGate(cirq.ry(-2.3)),
                    cirq.ControlledGate(cirq.rz(-1.4)),
                ],
            ),
            (
                qml.CRot(1, 2, -1, wires=[0, 1]).inv(),
                [
                    cirq.ControlledGate(cirq.rz(1)),
                    cirq.ControlledGate(cirq.ry(-2)),
                    cirq.ControlledGate(cirq.rz(-1)),
                ],
            ),
            (
                qml.CRot(-1.1, 0.2, -1, wires=[0, 1]).inv(),
                [
                    cirq.ControlledGate(cirq.rz(1)),
                    cirq.ControlledGate(cirq.ry(-0.2)),
                    cirq.ControlledGate(cirq.rz(1.1)),
                ],
            ),
            (qml.QubitUnitary(np.eye(4), wires=[0, 1]), [cirq.MatrixGate(np.eye(4))]),
            (
                qml.QubitUnitary(
                    np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
                    wires=[0, 1],
                ),
                [
                    cirq.MatrixGate(
                        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
                    )
                ],
            ),
            (
                qml.QubitUnitary(
                    np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]]) / 2,
                    wires=[0, 1],
                ),
                [
                    cirq.MatrixGate(
                        np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1],])
                        / 2
                    )
                ],
            ),
            (qml.QubitUnitary(np.eye(4), wires=[0, 1]).inv(), [cirq.MatrixGate(np.eye(4)) ** -1],),
            (
                qml.QubitUnitary(
                    np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),
                    wires=[0, 1],
                ).inv(),
                [
                    cirq.MatrixGate(
                        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
                    )
                    ** -1
                ],
            ),
            (
                qml.QubitUnitary(
                    np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1]]) / 2,
                    wires=[0, 1],
                ).inv(),
                [
                    cirq.MatrixGate(
                        np.array([[1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, 1, 1, 1],])
                        / 2
                    )
                    ** -1
                ],
            ),
        ],
    )
    def test_apply_two_wires(self, cirq_device_2_wires, gate, expected_cirq_gates):
        """Tests that apply adds the correct gates to the circuit for two-qubit gates."""

        cirq_device_2_wires.reset()

        cirq_device_2_wires.apply([gate])

        ops = list(cirq_device_2_wires.circuit.all_operations())

        assert len(ops) == len(expected_cirq_gates)

        for i in range(len(ops)):
            assert ops[i].gate == expected_cirq_gates[i]
