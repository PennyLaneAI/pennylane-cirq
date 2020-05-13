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


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestApply:
    """Tests that ops are correctly applied"""

    @pytest.mark.parametrize(
        "par,input,expected_density_matrix",
        [
            ([0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.5], [1, 0], np.array([[2, 0], [0, 1]]) / 3),
            ([1.0], [1, 0], np.array([[1, 0], [0, 2]]) / 3),
            ([0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.5], [0, 1], np.array([[1, 0], [0, 2]]) / 3),
            ([1.0], [0, 1], np.array([[2, 0], [0, 1]]) / 3),
            ([0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.5], np.array([1, 1]) / np.sqrt(2), np.array([[3, 1], [1, 3]]) / 6),
            ([1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, -1 / 3], [-1 / 3, 1]]) / 2),
            ([0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.5], np.array([1, -1]) / np.sqrt(2), np.array([[3, -1], [-1, 3]]) / 6),
            ([1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 1 / 3], [1 / 3, 1]]) / 2),
        ],
    )
    def test_apply_depolarize_single_wire(
        self, simulator_device_1_wire, tol, par, input, expected_density_matrix
    ):
        """Tests that applying a depolarizing operation yields the expected output state for single wire."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([ops.Depolarize(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )

    @pytest.mark.parametrize(
        "par,input,expected_density_matrix",
        [
            ([0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.5], [1, 0], np.array([[1, 0], [0, 1]]) / 2),
            ([1.0], [1, 0], np.array([[0, 0], [0, 1]])),
            ([0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.5], [0, 1], np.array([[1, 0], [0, 1]]) / 2),
            ([1.0], [0, 1], np.array([[1, 0], [0, 0]])),
            ([0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
        ]
    )
    def test_apply_bit_flip_single_wire(
        self, simulator_device_1_wire, tol, par, input, expected_density_matrix
    ):
        """Tests that applying a bit flip operation yields the expected output state for single wire."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([ops.BitFlip(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )

    @pytest.mark.parametrize(
        "par,input,expected_density_matrix",
        [
            ([0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.5], [1, 0], np.array([[1, 0], [0, 0]])),
            ([1.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.5], [0, 1], np.array([[0, 0], [0, 1]])),
            ([1.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            ([1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            ([1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
        ]
    )
    def test_apply_phase_flip_single_wire(
        self, simulator_device_1_wire, tol, par, input, expected_density_matrix
    ):
        """Tests that applying a phase flip operation yields the expected output state for single wire."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([ops.PhaseFlip(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )


    @pytest.mark.parametrize(
        "par,input,expected_density_matrix",
        [
            ([0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.5], [1, 0], np.array([[1, 0], [0, 0]])),
            ([1.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.5], [0, 1], np.array([[0, 0], [0, 1]])),
            ([1.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.5], np.array([1, 1]) / np.sqrt(2), np.array([[1, np.sqrt(1/2)], [np.sqrt(1/2), 1]]) / 2),
            ([1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
            ([0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.5], np.array([1, -1]) / np.sqrt(2), np.array([[1, -np.sqrt(1/2)], [-np.sqrt(1/2), 1]]) / 2),
            ([1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 0], [0, 1]]) / 2),
        ]
    )
    def test_apply_phase_damp_single_wire(
        self, simulator_device_1_wire, tol, par, input, expected_density_matrix
    ):
        """Tests that applying a phase damping operation yields the expected output state for single wire."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([ops.PhaseDamp(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )
        
    
    @pytest.mark.parametrize(
        "par,input,expected_density_matrix",
        [
            ([0.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.5], [1, 0], np.array([[1, 0], [0, 0]])),
            ([1.0], [1, 0], np.array([[1, 0], [0, 0]])),
            ([0.0], [0, 1], np.array([[0, 0], [0, 1]])),
            ([0.5], [0, 1], np.array([[1, 0], [0, 1]]) / 2),
            ([1.0], [0, 1], np.array([[1, 0], [0, 0]])),
            ([0.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 1], [1, 1]]) / 2),
            ([0.5], np.array([1, 1]) / np.sqrt(2), np.array([[3 / 2, np.sqrt(1 / 2)], [np.sqrt(1 / 2), 1 / 2]]) / 2),
            ([1.0], np.array([1, 1]) / np.sqrt(2), np.array([[1, 0], [0, 0]])),
            ([0.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, -1], [-1, 1]]) / 2),
            ([0.5], np.array([1, -1]) / np.sqrt(2), np.array([[3 / 2, -np.sqrt(1 / 2)], [-np.sqrt(1 / 2), 1 / 2]]) / 2),
            ([1.0], np.array([1, -1]) / np.sqrt(2), np.array([[1, 0], [0, 0]])),
        ]
    )
    def test_apply_amplitude_damp_single_wire(
        self, simulator_device_1_wire, tol, par, input, expected_density_matrix
    ):
        """Tests that applying an amplitude damping operation yields the expected output state for single wire."""

        simulator_device_1_wire.reset()
        simulator_device_1_wire._initial_state = np.array(input, dtype=np.complex64)
        simulator_device_1_wire.apply([ops.AmplitudeDamp(*par, wires=[0])])

        assert np.allclose(
            simulator_device_1_wire.state, expected_density_matrix, **tol
        )