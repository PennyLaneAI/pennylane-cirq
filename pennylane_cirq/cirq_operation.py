# Copyright 2019-2020 Xanadu Quantum Technologies Inc.

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
Cirq Operation class
====================

**Module name:** :mod:`pennylane_cirq.cirq_operation`

.. currentmodule:: pennylane_cirq.cirq_operation

An helper class that wraps the native Cirq operations and provides an interface for PennyLane.

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
            parametrization (Tuple[float] -> Union[Cirq:Qid, List[Cirq:Qid]]): Converts the
                PennyLane gate parameters to an ordered list of gates that are to be applied.
        """

        self.parametrization = parametrization
        self.parametrized_cirq_gates = None
        self.is_inverse = False

    def parametrize(self, *args):
        """Parametrizes the CirqOperation.

        Args:
            *args (float): the parameters for the operations
        """
        self.parametrized_cirq_gates = self.parametrization(*args)

        if not isinstance(self.parametrized_cirq_gates, Sequence):
            self.parametrized_cirq_gates = [self.parametrized_cirq_gates]

        if self.is_inverse:
            # Cirq automatically reverses the order if it gets an iterable
            self.parametrized_cirq_gates = cirq.inverse(self.parametrized_cirq_gates)

    def apply(self, *qubits):
        """Applies the CirqOperation.

        Args:
            *qubits (Cirq:Qid): the qubits on which the Cirq gates should be performed.
        """
        if not self.parametrized_cirq_gates:
            raise qml.DeviceError("CirqOperation must be parametrized before it can be applied.")

        return (parametrized_gate(*qubits) for parametrized_gate in self.parametrized_cirq_gates)

    def inv(self):
        """Inverses the CirqOperation."""
        # We can also support inversion after parametrization, but this is not necessary for the
        # PennyLane-Cirq codebase at the moment.

        if self.parametrized_cirq_gates:
            raise qml.DeviceError("CirqOperation can't be inverted after it was parametrized.")

        self.is_inverse = not self.is_inverse
