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
# pylint: disable=too-many-arguments
"""
This module provides the ``PasqalDevice`` from Cirq.
"""
from cirq import pasqal

from .simulator_device import SimulatorDevice


class PasqalDevice(SimulatorDevice):
    r"""Cirq Pasqal device for PennyLane.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.ThreeDGridQubit]): A list of Cirq ThreeDGridQubits that are used
            as wires. If not specified, the ThreeDGridQubits are put in a linear
            arrangement along the first coordinate axis,
            i.e., ``(0,0,0), (1,0,0), (2,0,0)``, etc.
        control_radius (float): The maximum distance between qubits for a controlled
                gate. Distance is measured in units of the ``ThreeDGridQubit`` indices.
    """
    name = "Cirq Pasqal device for PennyLane"
    short_name = "cirq.pasqal"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None, control_radius=1.0):

        if not qubits:
            # a linear chain in 3D with x coords uniform between 0 and 1
            max_pos = wires
            qubits = [pasqal.ThreeDQubit(wire / max_pos, 0, 0) for wire in range(wires)]
        self.control_radius = float(control_radius)
        if self.control_radius < 0:
            raise ValueError("The control_radius must be a non-negative real number.")
        super().__init__(wires, shots, analytic, qubits)
