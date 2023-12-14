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
"""Tests that variances are correctly computed in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A
from contextlib import contextmanager


@contextmanager
def mimic_execution_for_var(device):
    device.reset()

    with device.execution_context():
        yield

        if not device.shots is None:
            device._samples = device.generate_samples()


np.random.seed(42)


@pytest.mark.parametrize("shots", [None, 8192])
class TestVar:
    """Tests for the variance"""

    def test_var(self, device, shots, tol):
        """Tests for variance calculation"""
        dev = device(2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

        # test correct variance for <Z> of a rotated state
        with mimic_execution_for_var(dev):
            dev.apply([qml.RX(phi, wires=[0]), qml.RY(theta, wires=[0])])

        # Here the observable is already diagonal
        var = dev.var(qml.PauliZ(wires=[0]))
        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))

        assert np.allclose(var, expected, **tol)

    def test_var_hermitian(self, device, shots, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = device(2)
        dev.active_wires = {0}

        phi = 0.543
        theta = 0.6543

        # test correct variance for <H> of a rotated state
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])
        obs = qml.Hermitian(H, wires=[0])

        with mimic_execution_for_var(dev):
            dev.apply(
                [qml.RX(phi, wires=[0]), qml.RY(theta, wires=[0])],
                rotations=obs.diagonalizing_gates(),
            )

        var = dev.var(obs)
        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )

        assert np.allclose(var, expected, **tol)

    def test_var_projector(self, device, shots, tol):
        """Tests for variance calculation using an arbitrary Projector observable"""
        dev = device(2)

        phi = 0.543
        theta = 0.654

        with mimic_execution_for_var(dev):
            dev.apply([qml.RX(phi, wires=[0]), qml.RY(theta, wires=[1]), qml.CNOT(wires=[0, 1])])

        obs = qml.Projector([0, 0], wires=[0, 1])
        var = dev.var(obs)
        expected = (np.cos(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(var, expected, **tol)

        obs = qml.Projector([0, 1], wires=[0, 1])
        var = dev.var(obs)
        expected = (np.cos(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(var, expected, **tol)

        obs = qml.Projector([1, 0], wires=[0, 1])
        var = dev.var(obs)
        expected = (np.sin(phi / 2) * np.sin(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(var, expected, **tol)

        obs = qml.Projector([1, 1], wires=[0, 1])
        var = dev.var(obs)
        expected = (np.sin(phi / 2) * np.cos(theta / 2)) ** 2 - (
            (np.sin(phi / 2) * np.cos(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(var, expected, **tol)


@pytest.mark.parametrize("shots", [None, 8192])
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)
        obs = qml.PauliX(wires=[0]) @ qml.PauliY(wires=[2])

        with mimic_execution_for_var(dev):
            dev.apply(
                [
                    qml.RX(theta, wires=[0]),
                    qml.RX(phi, wires=[1]),
                    qml.RX(varphi, wires=[2]),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[1, 2]),
                ],
                rotations=obs.diagonalizing_gates(),
            )

        res = dev.var(obs)

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16

        assert np.allclose(res, expected, **tol)

    def test_pauliz_hadamard(self, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        obs = (
            qml.PauliZ(wires=[0])
            @ qml.Hadamard(wires=[1])
            @ qml.PauliY(wires=[2])
        )

        with mimic_execution_for_var(dev):
            dev.apply(
                [
                    qml.RX(theta, wires=[0]),
                    qml.RX(phi, wires=[1]),
                    qml.RX(varphi, wires=[2]),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[1, 2]),
                ],
                rotations=obs.diagonalizing_gates(),
            )

        res = dev.var(obs)

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

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
        obs = qml.PauliZ(wires=[0]) @ qml.Hermitian(A, wires=[1, 2])

        with mimic_execution_for_var(dev):
            dev.apply(
                [
                    qml.RX(theta, wires=[0]),
                    qml.RX(phi, wires=[1]),
                    qml.RX(varphi, wires=[2]),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[1, 2]),
                ],
                rotations=obs.diagonalizing_gates(),
            )

        res = dev.var(obs)

        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16

        assert np.allclose(res, expected, **tol)

    def test_projector(self, device, shots, tol):
        """Test that a tensor product involving qml.Projector works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        with mimic_execution_for_var(dev):
            dev.apply(
                [
                    qml.RX(theta, wires=[0]),
                    qml.RX(phi, wires=[1]),
                    qml.RX(varphi, wires=[2]),
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[1, 2]),
                ]
            )

        obs = qml.PauliZ(wires=[0]) @ qml.Projector([0, 0], wires=[1, 2])
        res = dev.var(obs)
        expected = (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.cos(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.cos(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, **tol)

        obs = qml.PauliZ(wires=[0]) @ qml.Projector([0, 1], wires=[1, 2])
        res = dev.var(obs)
        expected = (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.sin(varphi / 2) * np.cos(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.sin(varphi / 2) * np.sin(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, **tol)

        obs = qml.PauliZ(wires=[0]) @ qml.Projector([1, 0], wires=[1, 2])
        res = dev.var(obs)
        expected = (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.sin(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.sin(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, **tol)

        obs = qml.PauliZ(wires=[0]) @ qml.Projector([1, 1], wires=[1, 2])
        res = dev.var(obs)
        expected = (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            + (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) - (
            (np.cos(varphi / 2) * np.sin(phi / 2) * np.cos(theta / 2)) ** 2
            - (np.cos(varphi / 2) * np.cos(phi / 2) * np.sin(theta / 2)) ** 2
        ) ** 2
        assert np.allclose(res, expected, **tol)
