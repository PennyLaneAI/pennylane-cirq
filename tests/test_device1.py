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
Unit tests for the Fock plugin.
"""
import pytest

import strawberryfields as sf

import pennylane as qml
from pennylane import numpy as np


class TestVariance:
    """Test for the device variance"""

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=15)

        @qml.qnode(dev)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        r = 0.105
        phi = -0.654

        var = circuit(r, phi)
        expected = np.exp(2*r)*np.sin(phi)**2 + np.exp(-2*r)*np.cos(phi)**2
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([r, phi], method='A')
        gradF = circuit.jacobian([r, phi], method='F')
        expected = np.array([
            2*np.exp(2*r)*np.sin(phi)**2 - 2*np.exp(-2*r)*np.cos(phi)**2,
            2*np.sinh(2*r)*np.sin(2*phi)
        ])
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=15)

        @qml.qnode(dev)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.MeanPhoton(0))

        n = 0.12
        a = 0.105

        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradF = circuit.jacobian([n, a], method='F')
        expected = np.array([2*a**2+2*n+1, 2*a*(2*n+1)])
        assert np.allclose(gradF, expected, atol=tol, rtol=0)
