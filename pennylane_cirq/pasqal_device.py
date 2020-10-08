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
        control_radius (float): The maximum distance between qubits for a controlled
                gate. Distance is measured in units of the ``ThreeDGridQubit`` indices.
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.ThreeDGridQubit]): A list of Cirq ThreeDGridQubits that are used
            as wires. If not specified, the ThreeDGridQubits are put in a linear
            arrangement along the first coordinate axis, separated by a distance of
            ``control_radius / 2``.
            i.e., ``(0,0,0), (control_radius/2,0,0), (control_radius,0,0)``, etc.
    """
    name = "Cirq Pasqal device for PennyLane"
    short_name = "cirq.pasqal"

    def __init__(self, wires, control_radius, shots=1000, analytic=True, qubits=None):

        if not qubits:
            qubits = [pasqal.ThreeDQubit(wire * control_radius / 2, 0, 0) for wire in range(wires)]
        self.control_radius = float(control_radius)
        if self.control_radius < 0:
            raise ValueError("The control_radius must be a non-negative real number.")
        super().__init__(wires, shots, analytic, qubits)
        self.cirq_device = pasqal.PasqalVirtualDevice(self.control_radius, qubits)
