# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Base Cirq device class
===========================

**Module name:** :mod:`plugin_name.device`

.. currentmodule:: plugin_name.device

An abstract base class for constructing Cirq devices for PennyLane.

This should contain all the boilerplate for supporting PennyLane
from Cirq, making it easier to create new devices.
The abstract base class below should contain all common code required
by Cirq.

This abstract base class will not be used by the user. Add/delete
methods and attributes below where needed.

See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of how the Device class works.

Classes
-------

.. autosummary::
   CirqDevice

Code details
~~~~~~~~~~~~
"""
import abc

# we always import NumPy directly
import numpy as np
import cirq

from pennylane import Device

from ._version import __version__


class CirqDevice(Device, abc.ABC):
    r"""Abstract Cirq device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        additional_option (float): as many additional arguments can be
            added as needed
    """
    name = "Cirq Abstract PennyLane plugin baseclass"
    pennylane_requires = ">=0.4.0"
    version = __version__
    author = "Johannes Jakob Meyer"

    short_name = "cirq.device"

    def rot3(a, b, c):
        return cirq.Rz(c), cirq.Ry(b), cirq.Rz(a)

    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": cirq.SingleQubitMatrixGate,
        "PauliX": cirq.X,
        "PauliY": cirq.Y,
        "PauliZ": cirq.Z,
        "Hadamard": cirq.H,
        "CNOT": cirq.CNOT,
        "SWAP": cirq.SWAP,
        "CZ": cirq.CZ,
        "PhaseShift": cirq.S,
        "RX": cirq.Rx,
        "RY": cirq.Ry,
        "RZ": cirq.Rz,
        "Rot": rot3,
    }

    _observable_map = {}

    def pre_apply(self):
        self.circuit = cirq.Circuit()

    def apply(self, operation, wires, par):
        operation = _operation_map[operation](*par)

        self.circuit.append(operation)

    def post_apply(self):
        print(self.circuit)

    # _observable_map = {
    #     'PauliX': X,
    #     'PauliY': Y,
    #     'PauliZ': Z,
    #     'Hadamard': H,
    #     'Hermitian': hermitian,
    #     'Identity': identity
    # }
