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

**Module name:** :mod:`pennylane_cirq.cirq_interface`

.. currentmodule:: pennylane_cirq.cirq_interface

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
   CirqOperation

Code details
~~~~~~~~~~~~
"""
from collections.abc import Sequence
import cirq
import pennylane as qml


class CirqOperation:
    """A helper class that wraps the native Cirq operations and provides an 
       interface for parametrization and application."""

    def __init__(self, parametrization):
        """Initializes the CirqOperation

        Args:
            parametrization (Tuple[float] -> Union[Cirq:Qid, List[Cirq:Qid]]): Converts the PennyLane gate parameters to an ordered list of gates
              that are to be applied            
        """

        self.parametrization = parametrization
        self.parametrized_cirq_gates = None

    def parametrize(self, *args):
        """Parametrizes the CirqOperation.

        Args:
            *args (float): the parameters for the operations
        """
        self.parametrized_cirq_gates = self.parametrization(*args)

        if not isinstance(self.parametrized_cirq_gates, Sequence):
            self.parametrized_cirq_gates = [self.parametrized_cirq_gates]

    def apply(self, *qubits):
        """Applies the CirqOperation.

        Args:
            *qubits (Cirq:Qid): the qubits on which the Cirq gates should be performed.
        """
        if not self.parametrized_cirq_gates:
            raise qml.DeviceError("CirqOperation must be parametrized before it can be applied.")

        return (parametrized_gate(*qubits) for parametrized_gate in self.parametrized_cirq_gates)


def unitary_matrix_gate(U):
    """Creates a Cirq unitary matrix gate from a given matrix.

        Args:
            U (numpy.ndarray): an array representing the gate matrix.
    """
    if U.shape == (2, 2):
        return cirq.SingleQubitMatrixGate(U)
    if U.shape == (4, 4):
        return cirq.TwoQubitMatrixGate(U)
    else:
        raise qml.DeviceError(
            "Cirq only supports single-qubit and two-qubit unitary matrix gates. The given matrix had shape {}".format(
                U.shape
            )
        )
