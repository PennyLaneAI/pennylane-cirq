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
"""Tests that samples are correctly computed in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml

from conftest import U, U2, A
from contextlib import contextmanager


np.random.seed(42)


@contextmanager
def mimic_execution_for_sample(device):
    device.reset()

    with device.execution_context():
        yield

        device._samples = device.generate_samples()


@pytest.mark.parametrize("shots,analytic", [(1000, True), (8192, False)])
class TestSample:
    """Tests for the sample return type"""

    def test_sample_values(self, device, shots, tol):
        """Tests if the samples returned by sample have
        the correct values
        """
        # TODO: remove when random seeds in tests are fixed
        np.random.seed(42)

        dev = device(1)

        with mimic_execution_for_sample(dev):
            dev.apply([qml.RX(1.5708, wires=[0])])

        dev._obs_queue = [qml.PauliZ(wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        s1 = dev.sample(qml.PauliZ(wires=[0]))

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

    def test_sample_values_hermitian(self, device, shots, tol):
        """Tests if the samples of a Hermitian observable returned by sample have
        the correct values
        """
        dev = device(1)

        theta = 0.543
        A = np.array([[1, 2j], [-2j, 0]])

        with mimic_execution_for_sample(dev):
            dev.apply(
                [qml.RX(theta, wires=[0])],
                rotations=qml.Hermitian(A, wires=[0], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [qml.Hermitian(A, wires=[0], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        s1 = dev.sample(qml.Hermitian(A, wires=[0]))

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), **tol)

        # the analytic mean is 2*sin(theta)+0.5*cos(theta)+0.5
        assert np.allclose(np.mean(s1), 2 * np.sin(theta) + 0.5 * np.cos(theta) + 0.5, **tol)

        # the analytic variance is 0.25*(sin(theta)-4*cos(theta))^2
        assert np.allclose(np.var(s1), 0.25 * (np.sin(theta) - 4 * np.cos(theta)) ** 2, **tol)

    def test_sample_values_hermitian_multi_qubit(self, device, shots, tol):
        """Tests if the samples of a multi-qubit Hermitian observable returned by sample have
        the correct values
        """
        dev = device(2)
        theta = 0.543

        A = np.array(
            [
                [1, 2j, 1 - 2j, 0.5j],
                [-2j, 0, 3 + 4j, 1],
                [1 + 2j, 3 - 4j, 0.75, 1.5 - 2j],
                [-0.5j, 1, 1.5 + 2j, -1],
            ]
        )

        with mimic_execution_for_sample(dev):
            dev.apply(
                [qml.RX(theta, wires=[0]), qml.RY(2 * theta, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=qml.Hermitian(A, wires=[0, 1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [qml.Hermitian(A, wires=[0, 1], do_queue=False)]

        for idx in range(len(dev._obs_queue)):
            dev._obs_queue[idx].return_type = qml.operation.Sample

        s1 = dev.sample(qml.Hermitian(A, wires=[0, 1]))

        # s1 should only contain the eigenvalues of
        # the hermitian matrix
        eigvals = np.linalg.eigvalsh(A)
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), **tol)

        # make sure the mean matches the analytic mean
        expected = (
            88 * np.sin(theta)
            + 24 * np.sin(2 * theta)
            - 40 * np.sin(3 * theta)
            + 5 * np.cos(theta)
            - 6 * np.cos(2 * theta)
            + 27 * np.cos(3 * theta)
            + 6
        ) / 32
        assert np.allclose(np.mean(s1), expected, **tol)


@pytest.mark.parametrize("shots,analytic", [(1000, True), (8192, False)])
class TestTensorSample:
    """Test tensor expectation values"""

    def test_paulix_pauliy(self, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)
        obs = qml.PauliX(wires=[0], do_queue=False) @ qml.PauliY(wires=[2], do_queue=False)

        with mimic_execution_for_sample(dev):
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

        s1 = dev.sample(obs)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

        mean = np.mean(s1)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(var, expected, **tol)

    def test_pauliz_hadamard(self, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        dev = device(3)

        obs = (
            qml.PauliZ(wires=[0], do_queue=False)
            @ qml.Hadamard(wires=[1], do_queue=False)
            @ qml.PauliY(wires=[2], do_queue=False)
        )

        with mimic_execution_for_sample(dev):
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

        s1 = dev.sample(obs)

        # s1 should only contain 1 and -1
        assert np.allclose(s1 ** 2, 1, **tol)

        mean = np.mean(s1)
        expected = -(np.cos(varphi) * np.sin(phi) + np.sin(varphi) * np.cos(theta)) / np.sqrt(2)
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4
        assert np.allclose(var, expected, **tol)

    def test_hermitian(self, device, shots, tol):
        """Test that a tensor product involving qml.Hermitian works correctly"""
        theta = 0.432
        phi = 0.123
        varphi = -0.543

        # Reseed here so that all eigenvalues are guaranteed to appear in the sample
        np.random.seed(143)

        dev = device(3)

        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        obs = qml.PauliZ(wires=[0], do_queue=False) @ qml.Hermitian(A, wires=[1, 2], do_queue=False)

        with mimic_execution_for_sample(dev):
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

        s1 = dev.sample(obs)

        # s1 should only contain the eigenvalues of
        # the hermitian matrix tensor product Z
        Z = np.diag([1, -1])
        eigvals = np.linalg.eigvalsh(np.kron(Z, A))
        assert np.allclose(sorted(list(set(s1))), sorted(eigvals), **tol)

        mean = np.mean(s1)
        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )
        assert np.allclose(mean, expected, **tol)

        var = np.var(s1)
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
        assert np.allclose(var, expected, **tol)
