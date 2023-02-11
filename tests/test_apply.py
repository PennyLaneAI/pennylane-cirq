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
"""Tests that application of operations works correctly in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml
from pennylane_cirq import SimulatorDevice, MixedStateSimulatorDevice
from scipy.linalg import block_diag

from conftest import U, U2
from contextlib import contextmanager

np.random.seed(42)


# ==========================================================
# Some useful global variables

# non-parametrized qubit gates
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# parametrized qubit gates
phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)

# list of all non-parametrized single-qubit gates,
# along with the PennyLane operation name
single_qubit = [
    ("PauliX", X),
    ("PauliY", Y),
    ("PauliZ", Z),
    ("Hadamard", H),
    ("S", S),
    ("T", T),
]

# list of all parametrized single-qubit gates
single_qubit_param = [("PhaseShift", phase_shift), ("RX", rx), ("RY", ry), ("RZ", rz)]
# list of all non-parametrized two-qubit gates
two_qubit = [("CNOT", CNOT), ("SWAP", SWAP), ("CZ", CZ)]
# list of all parametrized two-qubit gates
two_qubit_param = [("CRZ", crz)]
# list of all three-qubit gates
three_qubit = [("Toffoli", toffoli), ("CSWAP", CSWAP)]


@contextmanager
def mimic_execution_for_apply(device):
    device.reset()

    with device.execution_context():
        yield


@pytest.mark.parametrize("shots", [None])
class TestApplyPureState:
    """Test application of PennyLane operations on the pure state simulator."""

    def test_basis_state(self, shots, tol):
        """Test basis state initialization"""
        dev = SimulatorDevice(4, shots=shots)
        state = np.array([0, 0, 1, 0])

        with mimic_execution_for_apply(dev):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev._state

        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize(
        "state",
        [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        ],
    )
    @pytest.mark.parametrize("device_wires", [3, 4, 5])
    @pytest.mark.parametrize("op_wires", [[0, 1], [1, 0], [2, 0]])
    def test_basis_state_on_wires_subset(self, state, device_wires, op_wires, tol):
        """Test basis state initialization on a subset of device wires"""
        dev = SimulatorDevice(device_wires)

        with mimic_execution_for_apply(dev):
            dev.apply([qml.BasisState(state, wires=op_wires)])

        res = np.abs(dev.state) ** 2
        # compute expected probabilities
        expected = np.zeros([2 ** len(op_wires)])
        expected[np.ravel_multi_index(state, [2] * len(op_wires))] = 1

        expected = dev._expand_state(expected, op_wires)

        assert np.allclose(res, expected, **tol)

    def test_identity_basis_state(self, shots, tol):
        """Test basis state initialization if identity"""
        dev = SimulatorDevice(4, shots=shots)
        state = np.array([1, 0, 0, 0])

        with mimic_execution_for_apply(dev):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev._state

        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        assert np.allclose(res, expected, **tol)

    def test_qubit_state_vector(self, init_state, shots, tol):
        """Test PauliX application"""
        dev = SimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply([qml.QubitStateVector(state, wires=[0])])

        res = dev._state
        expected = state
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("device_wires", [3, 4, 5])
    @pytest.mark.parametrize("op_wires", [[0], [2], [0, 1], [1, 0], [2, 0]])
    def test_qubit_state_vector_on_wires_subset(
        self, init_state, device_wires, op_wires, shots, tol
    ):
        """Test QubitStateVector application on a subset of device wires"""
        dev = SimulatorDevice(device_wires, shots=shots)
        state = init_state(len(op_wires))

        with mimic_execution_for_apply(dev):
            dev.apply([qml.QubitStateVector(state, wires=op_wires)])

        res = dev.state
        expected = dev._expand_state(state, op_wires)

        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("name,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        """Test application of single qubit gates without parameters"""
        dev = SimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0]),
                    qml.__getattribute__(name)(wires=[0]),
                ]
            )

        res = dev._state
        expected = mat @ state
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, shots, name, func, theta, tol):
        """Test application of single qubit gates with parameters"""
        dev = SimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0]),
                    qml.__getattribute__(name)(theta, wires=[0]),
                ]
            )

        res = dev._state
        expected = func(theta) @ state
        assert np.allclose(res, expected, **tol)

    def test_rotation(self, init_state, shots, tol):
        """Test three axis rotation gate"""
        dev = SimulatorDevice(1, shots=shots)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        with mimic_execution_for_apply(dev):
            dev.apply([qml.QubitStateVector(state, wires=[0]), qml.Rot(a, b, c, wires=[0])])

        res = dev._state
        expected = rot(a, b, c) @ state
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("name,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        """Test PauliX application"""
        dev = SimulatorDevice(2, shots=shots)
        state = init_state(2)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1]),
                    qml.__getattribute__(name)(wires=[0, 1]),
                ]
            )

        res = dev._state
        expected = mat @ state
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, shots, mat, tol):
        N = int(np.log2(len(mat)))
        dev = SimulatorDevice(N, shots=shots)
        state = init_state(N)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=list(range(N))),
                    qml.QubitUnitary(mat, wires=list(range(N))),
                ]
            )

        res = dev._state
        expected = mat @ state
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_state_unitary(self, shots):
        """Test that an exception is raised if the
        unitary matrix is the wrong size"""
        dev = SimulatorDevice(2, shots=shots)
        state = np.array([[0, 123.432], [-0.432, 023.4]])

        with pytest.raises(ValueError, match=r"Input unitary must be of shape"):
            with mimic_execution_for_apply(dev):
                dev.apply([qml.QubitUnitary(state, wires=[0, 1])])

    @pytest.mark.parametrize("name, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        dev = SimulatorDevice(3, shots=shots)
        state = init_state(3)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1, 2]),
                    qml.__getattribute__(name)(wires=[0, 1, 2]),
                ]
            )

        res = dev._state
        expected = mat @ state
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", two_qubit_param)
    def test_two_qubits_parameters(self, init_state, shots, name, func, theta, tol):
        """Test application of two qubit gates with parameters"""
        dev = SimulatorDevice(2, shots=shots)
        state = init_state(2)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1]),
                    qml.__getattribute__(name)(theta, wires=[0, 1]),
                ]
            )

        res = dev._state
        expected = func(theta) @ state
        assert np.allclose(res, expected, **tol)


@pytest.mark.parametrize("shots", [None])
class TestApplyMixedState:
    """Test application of PennyLane operations on the mixed state simulator."""

    def test_basis_state(self, shots, tol):
        """Test basis state initialization"""
        dev = MixedStateSimulatorDevice(4, shots=shots)
        state = np.array([0, 0, 1, 0])

        with mimic_execution_for_apply(dev):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev._state

        expected = np.zeros([16])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        expected = np.kron(expected, expected.conj()).reshape([2**4, 2**4])
        assert np.allclose(res, expected, **tol)

    def test_identity_basis_state(self, shots, tol):
        """Test basis state initialization if identity"""
        dev = MixedStateSimulatorDevice(4, shots=shots)
        state = np.array([1, 0, 0, 0])

        with mimic_execution_for_apply(dev):
            dev.apply([qml.BasisState(state, wires=[0, 1, 2, 3])])

        res = dev._state

        expected = np.zeros([16])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        expected = np.kron(expected, expected.conj()).reshape([16, 16])
        assert np.allclose(res, expected, **tol)

    def test_qubit_state_vector(self, init_state, shots, tol):
        """Test PauliX application"""
        dev = MixedStateSimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply([qml.QubitStateVector(state, wires=[0])])

        res = dev._state
        expected = state
        expected = np.kron(state, state.conj()).reshape([2, 2])
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("name,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        """Test application of single qubit gates without parameters"""
        dev = MixedStateSimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0]),
                    qml.__getattribute__(name)(wires=[0]),
                ]
            )

        res = dev._state
        expected = mat @ state
        expected = np.kron(expected, expected.conj()).reshape([2, 2])
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, shots, name, func, theta, tol):
        """Test application of single qubit gates with parameters"""
        dev = MixedStateSimulatorDevice(1, shots=shots)
        state = init_state(1)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0]),
                    qml.__getattribute__(name)(theta, wires=[0]),
                ]
            )

        res = dev._state
        expected = func(theta) @ state
        expected = np.kron(expected, expected.conj()).reshape([2, 2])
        assert np.allclose(res, expected, **tol)

    def test_rotation(self, init_state, shots, tol):
        """Test three axis rotation gate"""
        dev = MixedStateSimulatorDevice(1, shots=shots)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        with mimic_execution_for_apply(dev):
            dev.apply([qml.QubitStateVector(state, wires=[0]), qml.Rot(a, b, c, wires=[0])])

        res = dev._state
        expected = rot(a, b, c) @ state
        expected = np.kron(expected, expected.conj()).reshape([2, 2])
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("name,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        """Test PauliX application"""
        dev = MixedStateSimulatorDevice(2, shots=shots)
        state = init_state(2)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1]),
                    qml.__getattribute__(name)(wires=[0, 1]),
                ]
            )

        res = dev._state
        expected = mat @ state
        expected = np.kron(expected, expected.conj()).reshape([4, 4])
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, shots, mat, tol):
        N = int(np.log2(len(mat)))
        dev = MixedStateSimulatorDevice(N, shots=shots)
        state = init_state(N)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=list(range(N))),
                    qml.QubitUnitary(mat, wires=list(range(N))),
                ]
            )

        res = dev._state
        expected = mat @ state
        expected = np.kron(expected, expected.conj()).reshape([2**N, 2**N])
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_state_unitary(self, shots):
        """Test that an exception is raised if the
        unitary matrix is the wrong size"""
        dev = MixedStateSimulatorDevice(2, shots=shots)
        state = np.array([[0, 123.432], [-0.432, 023.4]])

        with pytest.raises(ValueError, match=r"Input unitary must be of shape"):
            with mimic_execution_for_apply(dev):
                dev.apply([qml.QubitUnitary(state, wires=[0, 1])])

    @pytest.mark.parametrize("name, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, shots, name, mat, tol):
        dev = MixedStateSimulatorDevice(3, shots=shots)
        state = init_state(3)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1, 2]),
                    qml.__getattribute__(name)(wires=[0, 1, 2]),
                ]
            )

        res = dev._state
        expected = mat @ state
        expected = np.kron(expected, expected.conj()).reshape([8, 8])
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("name,func", two_qubit_param)
    def test_two_qubits_parameters(self, init_state, shots, name, func, theta, tol):
        """Test application of single qubit gates with parameters"""
        dev = MixedStateSimulatorDevice(2, shots=shots)
        state = init_state(2)

        with mimic_execution_for_apply(dev):
            dev.apply(
                [
                    qml.QubitStateVector(state, wires=[0, 1]),
                    qml.__getattribute__(name)(theta, wires=[0, 1]),
                ]
            )

        res = dev._state
        expected = func(theta) @ state
        expected = np.kron(expected, expected.conj()).reshape([4, 4])
        assert np.allclose(res, expected, **tol)
