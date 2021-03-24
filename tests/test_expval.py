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
from contextlib import contextmanager

from conftest import U, U2, A, B


np.random.seed(42)


@contextmanager
def mimic_execution_for_expval(device):
    device.reset()

    with device.execution_context():
        yield

        if not device.shots is None:
            device._samples = device.generate_samples()


@pytest.mark.parametrize("shots", [None, 8192])
class TestExpval:
    """Test expectation values"""

    def test_identity_expectation(self, device, shots, tol):
        """Test that identity expectation value (i.e. the trace) is 1"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1]),]
            )

        O = qml.Identity
        name = "Identity"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]

        res = np.array(
            [dev.expval(O(wires=[0], do_queue=False)), dev.expval(O(wires=[1], do_queue=False)),]
        )

        assert np.allclose(res, np.array([1, 1]), **tol)

    def test_pauliz_expectation(self, device, shots, tol):
        """Test that PauliZ expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1]),]
            )

        O = qml.PauliZ
        name = "PauliZ"

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]

        res = np.array(
            [dev.expval(O(wires=[0], do_queue=False)), dev.expval(O(wires=[1], do_queue=False)),]
        )

        assert np.allclose(res, np.array([np.cos(theta), np.cos(theta) * np.cos(phi)]), **tol)

    def test_paulix_expectation(self, device, shots, tol):
        """Test that PauliX expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        O = qml.PauliX

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=O(wires=[0], do_queue=False).diagonalizing_gates()
                + O(wires=[1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]

        res = np.array(
            [dev.expval(O(wires=[0], do_queue=False)), dev.expval(O(wires=[1], do_queue=False)),]
        )
        assert np.allclose(res, np.array([np.sin(theta) * np.sin(phi), np.sin(phi)]), **tol)

    def test_pauliy_expectation(self, device, shots, tol):
        """Test that PauliY expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        O = qml.PauliY

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RX(theta, wires=[0]), qml.RX(phi, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=O(wires=[0], do_queue=False).diagonalizing_gates()
                + O(wires=[1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]

        res = np.array(
            [dev.expval(O(wires=[0], do_queue=False)), dev.expval(O(wires=[1], do_queue=False)),]
        )
        assert np.allclose(res, np.array([0, -(np.cos(theta)) * np.sin(phi)]), **tol)

    def test_hadamard_expectation(self, device, shots, tol):
        """Test that Hadamard expectation value is correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        O = qml.Hadamard

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=O(wires=[0], do_queue=False).diagonalizing_gates()
                + O(wires=[1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [O(wires=[0], do_queue=False), O(wires=[1], do_queue=False)]

        res = np.array(
            [dev.expval(O(wires=[0], do_queue=False)), dev.expval(O(wires=[1], do_queue=False)),]
        )
        expected = np.array(
            [
                np.sin(theta) * np.sin(phi) + np.cos(theta),
                np.cos(theta) * np.cos(phi) + np.sin(phi),
            ]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, **tol)

    def test_hermitian_expectation(self, device, shots, tol):
        """Test that arbitrary Hermitian expectation values are correct"""
        theta = 0.432
        phi = 0.123

        dev = device(2)
        O = qml.Hermitian

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=O(A, wires=[0], do_queue=False).diagonalizing_gates()
                + O(A, wires=[1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [
            O(A, wires=[0], do_queue=False),
            O(A, wires=[1], do_queue=False),
        ]

        res = np.array(
            [
                dev.expval(O(A, wires=[0], do_queue=False)),
                dev.expval(O(A, wires=[1], do_queue=False)),
            ]
        )

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
        O = qml.Hermitian

        with mimic_execution_for_expval(dev):
            dev.apply(
                [qml.RY(theta, wires=[0]), qml.RY(phi, wires=[1]), qml.CNOT(wires=[0, 1]),],
                rotations=O(B, wires=[0, 1], do_queue=False).diagonalizing_gates(),
            )

        dev._obs_queue = [O(B, wires=[0, 1], do_queue=False)]

        res = np.array([dev.expval(O(B, wires=[0, 1], do_queue=False))])

        # below is the analytic expectation value for this circuit with arbitrary
        # Hermitian observable B
        expected = 0.5 * (
            6 * np.cos(theta) * np.sin(phi)
            - np.sin(theta) * (8 * np.sin(phi) + 7 * np.cos(phi) + 3)
            - 2 * np.sin(phi)
            - 6 * np.cos(phi)
            - 6
        )

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

        obs = qml.PauliX(wires=[0], do_queue=False) @ qml.PauliY(wires=[2], do_queue=False)

        with mimic_execution_for_expval(dev):
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

        res = dev.expval(obs)
        expected = np.sin(theta) * np.sin(phi) * np.sin(varphi)

        assert np.allclose(res, expected, **tol)

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

        with mimic_execution_for_expval(dev):
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

        res = dev.expval(obs)
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
        obs = qml.PauliZ(wires=[0], do_queue=False) @ qml.Hermitian(A, wires=[1, 2], do_queue=False)

        with mimic_execution_for_expval(dev):
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

        res = dev.expval(obs)

        expected = 0.5 * (
            -6 * np.cos(theta) * (np.cos(varphi) + 1)
            - 2 * np.sin(varphi) * (np.cos(theta) + np.sin(phi) - 2 * np.cos(phi))
            + 3 * np.cos(varphi) * np.sin(phi)
            + np.sin(phi)
        )

        assert np.allclose(res, expected, **tol)
