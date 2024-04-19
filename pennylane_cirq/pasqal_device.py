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
import cirq_pasqal

from .simulator_device import SimulatorDevice


class PasqalDevice(SimulatorDevice):
    r"""Cirq Pasqal device for PennyLane.

    Args:
        wires (int): the number of wires to initialize the device with
        control_radius (float): The maximum distance between qubits for a controlled
                gate. Distance is measured in units of the ``ThreeDGridQubit`` indices.
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. If ``None``, expecation values are calculated analytically.
        qubits (List[cirq_pasqal.ThreeDGridQubit]): A list of Cirq ThreeDGridQubits that are used
            as wires. If not specified, the ThreeDGridQubits are put in a linear
            arrangement along the first coordinate axis, separated by a distance of
            ``control_radius / 2``.
            i.e., ``(0,0,0), (control_radius/2,0,0), (control_radius,0,0)``, etc.
    """

    name = "Cirq Pasqal device for PennyLane"
    short_name = "cirq.pasqal"

    def __init__(self, wires, control_radius, shots=None, qubits=None):
        if isinstance(wires, int):
            # interpret wires as the number of consecutive wires
            wires = range(wires)
        if not qubits:
            qubits = [
                cirq_pasqal.ThreeDQubit(wire * control_radius / 2, 0, 0)
                for wire in range(len(wires))
            ]
        self.control_radius = float(control_radius)
        if self.control_radius < 0:
            raise ValueError("The control_radius must be a non-negative real number.")
        super().__init__(wires, shots, qubits)

        self.cirq_device = cirq_pasqal.PasqalVirtualDevice(self.control_radius, qubits)
