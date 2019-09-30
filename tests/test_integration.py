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
"""Tests that plugin devices are accessible and integrate with PennyLane"""
import numpy as np
import pennylane as qml
import pytest

from conftest import shortnames


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortnames)
    def test_load_device(self, d, backend):
        """Test that the QVM device loads correctly"""
        dev = qml.device(d, wires=2, backend=backend, shots=1024)
        assert dev.num_wires == 2
        assert dev.shots == 1024
        assert dev.short_name == d

    def test_args(self):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("pluginname.device1")

        # a hardware device will not allow shots=0
        with pytest.raises(ValueError, match="must be a positive integer"):
            qml.device("pluginname.device2", wires=1, shots=0)

        # a state simulator will allow shots=0
        qml.device("pluginname.device1", wires=1, shots=0)
        qml.device("pluginname.device1", wires=1, shots=0)

    @pytest.mark.parametrize("d", shortnames)
    @pytest.mark.parametrize("shots", [0, 8192])
    def test_one_qubit_circuit(self, shots, d, backend, tol):
        """Test that devices provide correct result for a simple circuit"""
        if backend not in state_backends and shots == 0:
            pytest.skip("Hardware simulators do not support analytic mode")

        dev = qml.device(d, wires=1, backend=backend, shots=shots)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)
