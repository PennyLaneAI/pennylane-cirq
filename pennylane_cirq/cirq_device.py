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

**Module name:** :mod:`pennylane_cirq.device`

.. currentmodule:: pennylane_cirq.device

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
   CirqCommand
   CirqDevice

Code details
~~~~~~~~~~~~
"""
import numpy as np
import cirq

import pennylane as qml
from pennylane import Device

from ._version import __version__

class CirqCommand:
    """A helper class that wraps the native Cirq commands and provides an 
       interface for parametrization and application."""

    def __init__(self, cirq_gate, is_parametrized=False):
        """Initializes the CirqCommand.

        Args:
            cirq_gate (Cirq:Qid): the Cirq gate to be wrapped
            is_parametrized (Bool): Indicates if the Cirq gate is parametrized
        """

        self.cirq_gate = cirq_gate
        self.parametrized_cirq_gate = None
        self.is_parametrized = is_parametrized
    
    def parametrize(self, *args):
        """Parametrizes the CirqCommand.

        Args:
            *args: the Cirq arguments to be passed to the Cirq gate.
        """
        if self.is_parametrized:
            self.parametrized_cirq_gate = self.cirq_gate(*args)

    def apply(self, *qubits):
        """Parametrizes the CirqCommand.

        Args:
            *qubits (Cirq:Qid): the qubits on which the Cirq gate should be performed.
        """
        if self.is_parametrized:
            if not self.parametrized_cirq_gate:
                raise qml.DeviceError("Gate must be parametrized before it can be applied.")
            
            return self.parametrized_cirq_gate(*qubits)
        else:
            return self.cirq_gate(*qubits)

class CirqDevice(Device):
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

    def __init__(self, wires, shots, qubits=None):
        super().__init__(wires, shots)
        
        if qubits:
            if wires != len(qubits):
                raise qml.DeviceError("The number of given qubits and the specified number of wires have to match. Got {} wires and {} qubits.".format(wires, len(qubits)))

            self.qubits = qubits
        else:
            self.qubits = [cirq.LineQubit(wire) for wire in range(wires)]

    def rot3(a, b, c):
        return cirq.Rz(c) @ cirq.Ry(b) @ cirq.Rz(a)


    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": CirqCommand(cirq.SingleQubitMatrixGate, True),
        "PauliX": CirqCommand(cirq.X),
        "PauliY": CirqCommand(cirq.Y),
        "PauliZ": CirqCommand(cirq.Z),
        "Hadamard": CirqCommand(cirq.H),
        "CNOT": CirqCommand(cirq.CNOT),
        "SWAP": CirqCommand(cirq.SWAP),
        "CZ": CirqCommand(cirq.CZ),
        "PhaseShift": None,
        "RX": CirqCommand(cirq.Rx, True),
        "RY": CirqCommand(cirq.Ry, True),
        "RZ": CirqCommand(cirq.Rz, True),
        "Rot": None,
    }

    _observable_map = {
        'PauliX': None,
        'PauliY': None,
        'PauliZ': None,
        'Hadamard': None,
        'Hermitian': None,
        'Identity': None
    }

    def reset(self):
        pass
        
    @property
    def observables(self):
        return set(self._observable_map.keys())

    @property
    def operations(self):
        return set(self._operation_map.keys())

    def pre_apply(self):
        self.circuit = cirq.Circuit()

    def apply(self, operation, wires, par):
        command = self._operation_map[operation]
        command.parametrize(*par)

        self.circuit.append(command.apply(*[self.qubits[wire] for wire in wires]))

    def post_apply(self):
        print(self.circuit)
