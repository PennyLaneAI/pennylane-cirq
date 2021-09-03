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
"""Tests that exectation values are correctly computed in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml

from conftest import A, B


np.random.seed(42)


@pytest.mark.parametrize("shots", [None, 8192])
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, shots, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Identity(wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]

        assert np.allclose(res, np.array([1, 1]), **tol)

    def test_pauliz_expectation(self, device, shots, tol):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]

        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), **tol)

    def test_paulix_expectation(self, device, shots, tol):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]
        assert np.allclose(res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), **tol)

    def test_pauliy_expectation(self, device, shots, tol):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        O = qml.PauliY

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]
        assert np.allclose(res, np.array([0, -(np.cos(theta)) * np.sin(phi)]), **tol)

    def test_hadamard_expectation(self, device, shots, tol):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hadamard(wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hadamard(wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]
        expected = (
            np.array(
                [
                    np.sin(theta) * np.sin(phi) + np.cos(theta),
                    np.cos(theta) * np.cos(phi) + np.sin(phi),
                ]
            )
            / np.sqrt(2)
        )
        assert np.allclose(res, expected, **tol)

    def test_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit0(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A, wires=[0]))

        @qml.qnode(dev)
        def circuit1(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(A, wires=[1]))

        res = [circuit0(phi, theta), circuit1(phi, theta)]

        a = A[0, 0]
        re_b = A[0, 1].real
        d = A[1, 1]
        ev1 = ((a - d) * np.cos(theta) + 2 * re_b * np.sin(theta) * np.sin(phi) + a + d) / 2
        ev2 = ((a - d) * np.cos(theta) * np.cos(phi) + 2 * re_b * np.sin(phi) + a + d) / 2
        expected = np.array([ev1, ev2])

        assert np.allclose(res, expected, **tol)

    def test_multi_mode_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary multi-mode Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Hermitian(B, wires=[0, 1]))

        res = circuit(phi, theta)
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

        assert np.allclose(res, expected, **tol)

    def test_projector_expectation(self, device, shots, tol):
        """Test that arbitrary Projector expectation values are correct"""
        theta = 0.732
        phi = 0.523

        dev = device(2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector([0, 0], wires=[0, 1]))

        res = circuit(phi, theta)
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2

        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector([0, 1], wires=[0, 1]))

        res = circuit(phi, theta)
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2

        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector([1, 0], wires=[0, 1]))

        res = circuit(phi, theta)

        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(phi, theta):
            qml.RY(theta, wires=[0])
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Projector([1, 1], wires=[0, 1]))

        res = circuit(phi, theta)
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2

        assert np.allclose(res, expected, **tol)


@pytest.mark.parametrize("shots", [None, 8192])
class TestTensorExpval:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2]))

        res = circuit(theta, phi, varphi)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, **tol)

    def test_pauliz_hadamard(self, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(
                qml.PauliZ(wires=[0]) @ qml.Hadamard(wires=[1]) @ qml.PauliY(wires=[2])
            )

        res = circuit(theta, phi, varphi)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)

        assert np.allclose(res, expected, **tol)

    def test_hermitian(self, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Hermitian(A, wires=[1, 2]))

        res = circuit(theta, phi, varphi)
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, **tol)

    def test_projector(self, device, shots, tol):
        """Test that a tensor product involving qml.Projector works correctly"""
        theta = 0.732
        phi = 0.523
        varphi = -0.543

        dev = device(3)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Projector([0, 0], wires=[1, 2]))

        res = circuit(theta, phi, varphi)
        expected = (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Projector([0, 1], wires=[1, 2]))

        res = circuit(theta, phi, varphi)
        expected = (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Projector([1, 0], wires=[1, 2]))

        res = circuit(theta, phi, varphi)
        expected = (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, **tol)

        @qml.qnode(dev)
        def circuit(theta, phi, varphi):
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1])
            qml.RX(varphi, wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.expval(qml.PauliZ(wires=[0]) @ qml.Projector([1, 1], wires=[1, 2]))

        res = circuit(theta, phi, varphi)
        expected = (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)
        ) ** 2
        assert np.allclose(res, expected, **tol)
