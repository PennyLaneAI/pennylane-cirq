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
Unit tests for the Simulator plugin
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
import pennylane_cirq
import cirq

class TestDeviceIntegration:
    """Tests that the SimulatorDevice integrates well with PennyLane"""

    def test_device_loading(self):
        """Tests that the cirq.simulator device is properly loaded"""

        dev = qml.device("cirq.simulator", wires=2)

        assert dev.num_wires == 2
        assert dev.shots == 0
        assert dev.short_name == "cirq.simulator"


class TestApply:
    """Tests that gates are correctly applied"""

    def test_simple_circuit(self):
        """Tests a simple circuit"""

        qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
        dev = qml.device("cirq.simulator", wires=3, qubits=qubits)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.CZ(wires=[0, 1])
            qml.CNOT(wires=[2, 0])
            qml.PauliY(wires=1)
            qml.RX(0.342, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.342, wires=1)
            qml.Hadamard(wires=1)
            qml.SWAP(wires=[1, 2])
            qml.RZ(0.342, wires=1)
            qml.Rot(0.342, 0.2, 0.1, wires=0)
            qml.PauliX(wires=0)
            qml.Hadamard(wires=2)
            qml.SWAP(wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 0])

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        print(circuit())

        raise Exception()
        

