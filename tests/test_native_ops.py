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
Unit tests for the native Cirq ops
"""
import pytest
import math

import pennylane as qml
import numpy as np
from pennylane_cirq import ops, MixedStateSimulatorDevice
import cirq


@pytest.fixture(scope="function")
def simulator_device_1_wire(shots, analytic):
    """Return a single wire instance of the MixedStateSimulatorDevice class."""
    yield MixedStateSimulatorDevice(1, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def simulator_device_2_wires(shots, analytic):
    """Return a two wire instance of the MixedStateSimulatorDevice class."""
    yield MixedStateSimulatorDevice(2, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def simulator_device_3_wires(shots, analytic):
    """Return a three wire instance of the MixedStateSimulatorDevice class."""
    yield MixedStateSimulatorDevice(3, shots=shots, analytic=analytic)


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestApply:
    """Tests that ops are correctly applied"""

    @pytest.mark.parametrize(
        "op,par,input,expected_density_matrix",
        [
            (ops.BitFlip, [0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.BitFlip, [0.5], [1, 0], np.array([[1, 0], [0, 1]]) / 2),
            (ops.BitFlip, [1.0], [1, 0], np.array([[0, 0], [0, 1]])),
            (ops.BitFlip, [0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.BitFlip, [0.5], [0, 1], np.array([[1, 0], [0, 1]]) / 2),
            (ops.BitFlip, [1.0], [0, 1], np.array([[1, 0], [0, 0]])),
            (ops.BitFlip, [0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.BitFlip, [0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.BitFlip, [1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.BitFlip, [0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.BitFlip, [0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.BitFlip, [1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.PhaseFlip, [0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseFlip, [0.5], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseFlip, [1.0], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseFlip, [0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseFlip, [0.5], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseFlip, [1.0], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseFlip, [0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.PhaseFlip, [0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            (ops.PhaseFlip, [1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.PhaseFlip, [0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.PhaseFlip, [0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            (ops.PhaseFlip, [1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.PhaseDamp, [0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseDamp, [0.5], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseDamp, [1.0], [1, 0], np.array([[1, 0], [0, 0]])),
            (ops.PhaseDamp, [0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseDamp, [0.5], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseDamp, [1.0], [0, 1], np.array([[0, 0], [0, 1]])),
            (ops.PhaseDamp, [0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            (ops.PhaseDamp, [0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, np.sqrt(1/2)], [np.sqrt(1/2), 1]]) / 2),
            (ops.PhaseDamp, [1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            (ops.PhaseDamp, [0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            (ops.PhaseDamp, [0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, -np.sqrt(1/2)], [-np.sqrt(1/2), 1]]) / 2),
            (ops.PhaseDamp, [1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
        ],
    )
    def test_apply_operation_single_wire_with_parameters(
        self, simulator_device_1_wire, tol, op, par, input, expected_density_matrix
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have parameters."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([op(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )