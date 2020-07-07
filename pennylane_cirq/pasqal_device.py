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
Cirq Pasqal Devices
======================

**Module name:** :mod:`pennylane_cirq.pasqal_device`

.. currentmodule:: pennylane_cirq.pasqal_device

This Device implements exposes the ``PasqalDevice`` from Cirq.

Classes
-------

.. autosummary::
   PasqalDevice

----
"""
from cirq import pasqal

from .simulator_device import SimulatorDevice


class PasqalDevice(SimulatorDevice):
    r"""Cirq Pasqal device for PennyLane.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): Indicates that expectation values and variances should
            be calculated analytically. Defaults to ``True``.
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """
    name = "Cirq Pasqal device for PennyLane"
    short_name = "cirq.pasqal"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None, control_radius=None):

        if not qubits:
            qubits = [pasqal.ThreeDGridQubit(wire) for wire in range(wires)]
        self.control_radius = control_radius or 1.0
        super().__init__(wires, shots, analytic, qubits)

