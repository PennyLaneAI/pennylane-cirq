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
Unit tests for the PasqalDevice
"""
import pytest

import pennylane as qml
from pennylane_cirq import PasqalDevice, SimulatorDevice
from cirq.pasqal import ThreeDQubit, PasqalVirtualDevice


class TestDeviceIntegration:
    """Tests that the SimulatorDevice integrates well with PennyLane"""

    def test_device_loading(self):
        """Tests that the cirq.pasqal device is properly loaded"""

        control_radius = 1.0
        dev = qml.device("cirq.pasqal", wires=2, control_radius=1.0)

        assert dev.num_wires == 2
        assert len(dev.qubits) == 2
        assert dev.shots == 1000
        assert dev.short_name == "cirq.pasqal"
        assert dev.control_radius == 1.0
        assert dev.qubits == sorted([ThreeDQubit(0, 0, 0), ThreeDQubit(control_radius / 2, 0, 0)])
        assert isinstance(dev, SimulatorDevice)


class TestDevice:
    """Unit tests for the PasqalDevice"""

    @pytest.mark.parametrize("control_radius", [1.0, 2.0, 99.99])
    def test_device_creation(self, control_radius):
        """Tests that the cirq.pasqal device is properly created"""

        dev = PasqalDevice(wires=2, shots=123, control_radius=control_radius)

        assert dev.num_wires == 2
        assert len(dev.qubits) == 2
        assert dev.shots == 123
        assert dev.short_name == "cirq.pasqal"
        assert dev.control_radius == control_radius
        assert dev.qubits == [ThreeDQubit(0, 0, 0), ThreeDQubit(control_radius / 2, 0, 0)]
        assert isinstance(dev, SimulatorDevice)
        assert isinstance(dev.cirq_device, PasqalVirtualDevice)

    @pytest.mark.parametrize(
        "coord_idxs",
        [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)],
            [(-1, -1, -1000), (1, 2, 3), (3, 2, 3), (3, 3, 3)],
        ],
    )
    def test_device_creation_threeDqubits_ordered(self, coord_idxs):
        """Tests that a PasqalDevice can be properly instantiated with ThreeDQubits that are ordered following Cirq's convention."""

        qubits = [ThreeDQubit(*idxs) for idxs in coord_idxs]
        dev = PasqalDevice(wires=4, qubits=qubits, control_radius=3)

        assert dev.qubits == qubits

    @pytest.mark.parametrize(
        "coord_idxs",
        [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
            [(1, 2, 3), (3, 2, 1), (0, 0, 0), (-1, -1, -10)],
        ],
    )
    def test_device_creation_threeDqubits_unordered(self, coord_idxs):
        """Tests that a PasqalDevice can be properly instantiated with ThreeDQubits that are not ordered following Cirq's convention."""

        qubits = [ThreeDQubit(*idxs) for idxs in coord_idxs]
        dev = PasqalDevice(wires=4, qubits=qubits, control_radius=3)

        assert dev.qubits == sorted(qubits)

    def test_control_radius_negative_exception(self):
        """Tests that an exception is raised when the supplied control_radius parameter
        is a negative real number"""

        with pytest.raises(ValueError, match="must be a non-negative real number"):
            dev = PasqalDevice(wires=2, shots=123, control_radius=-5.0)
