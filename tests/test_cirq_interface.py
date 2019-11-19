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
Unit tests for the Cirq interface routines
"""
import cirq
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane_cirq.cirq_interface import CirqOperation, unitary_matrix_gate


class TestCirqOperation:
    """Tests the CirqOperation class."""

    def test_init(self):
        """Tests that the class is properly initialized."""
        fun = lambda x: cirq.Ry(x)

        operation = CirqOperation(fun)

        assert operation.parametrized_cirq_gates is None
        assert operation.parametrization == fun
        assert not operation.is_inverse

    def test_parametrize(self):
        """Tests that parametrize yields the correct queue of operations."""

        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )
        operation.parametrize(0.1, 0.2, 0.3)

        assert operation.parametrized_cirq_gates[0] == cirq.X
        assert operation.parametrized_cirq_gates[1] == cirq.Ry(0.1)
        assert operation.parametrized_cirq_gates[2] == cirq.Rx(0.2)
        assert operation.parametrized_cirq_gates[3] == cirq.Z
        assert operation.parametrized_cirq_gates[4] == cirq.Rz(0.3)

    def test_apply(self):
        """Tests that the operations in the queue are correctly applied."""

        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )
        operation.parametrize(0.1, 0.2, 0.3)

        qubit = cirq.LineQubit(1)

        gate_applications = list(operation.apply(qubit))

        assert gate_applications[0] == cirq.X.on(qubit)
        assert gate_applications[1] == cirq.Ry(0.1).on(qubit)
        assert gate_applications[2] == cirq.Rx(0.2).on(qubit)
        assert gate_applications[3] == cirq.Z.on(qubit)
        assert gate_applications[4] == cirq.Rz(0.3).on(qubit)

    def test_apply_not_parametrized(self):
        """Tests that the proper error is raised if an Operation is applied
        that was not parametrized before."""
        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )
        qubit = cirq.LineQubit(1)

        with pytest.raises(
            qml.DeviceError, match="CirqOperation must be parametrized before it can be applied."
        ):
            operation.apply(qubit)

    def test_inv(self):
        """Test that inv inverts the gate and applying inv twice yields the initial state."""

        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )

        assert not operation.is_inverse

        operation.inv()

        assert operation.is_inverse

        operation.inv()

        assert not operation.is_inverse

    def test_inv_apply(self):
        """Tests that the operations in the queue are correctly applied if the 
        CirqOperation is inverted."""

        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )
        operation.inv()

        operation.parametrize(0.1, 0.2, 0.3)

        qubit = cirq.LineQubit(1)

        gate_applications = list(operation.apply(qubit))

        assert gate_applications[0] == cirq.Rz(-0.3).on(qubit)
        assert gate_applications[1] == (cirq.Z ** -1).on(qubit)
        assert gate_applications[2] == cirq.Rx(-0.2).on(qubit)
        assert gate_applications[3] == cirq.Ry(-0.1).on(qubit)
        assert gate_applications[4] == (cirq.X ** -1).on(qubit)

    def test_inv_error(self):
        """Test that inv raises an error if the CirqOperation was already parametrized."""

        operation = CirqOperation(
            lambda a, b, c: [cirq.X, cirq.Ry(a), cirq.Rx(b), cirq.Z, cirq.Rz(c)]
        )
        operation.parametrize(0.1, 0.2, 0.3)

        with pytest.raises(
            qml.DeviceError, match="CirqOperation must be inverted before it is parametrized"
        ):
            operation.inv()


class TestMethods:
    """Tests the independent methods in the Cirq interface."""

    @pytest.mark.parametrize(
        "U,expected_cirq_operation",
        [
            ([[1, 0], [0, -1]], cirq.SingleQubitMatrixGate(np.array([[1, 0], [0, -1]])),),
            ([[0, 1j], [-1j, 0]], cirq.SingleQubitMatrixGate(np.array([[0, 1j], [-1j, 0]])),),
            (
                [[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]],
                cirq.TwoQubitMatrixGate(
                    np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, -1j, 0]])
                ),
            ),
        ],
    )
    def test_unitary_matrix_gate(self, U, expected_cirq_operation):
        """Tests that the correct Cirq operation is returned for the unitary matrix gate."""

        assert unitary_matrix_gate(np.array(U)) == expected_cirq_operation

    @pytest.mark.parametrize("U", [np.eye(6), np.eye(10), np.eye(3), np.eye(3, 5)])
    def test_unitary_matrix_gate_error(self, U):
        """Tests that an error is raised if the given matrix is of wrong format."""

        with pytest.raises(
            qml.DeviceError,
            match="Cirq only supports single-qubit and two-qubit unitary matrix gates.",
        ):
            unitary_matrix_gate(np.array(U))
