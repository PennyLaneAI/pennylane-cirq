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
Base device class for PennyLane-Cirq
===========================

**Module name:** :mod:`pennylane_cirq.cirq_device`

.. currentmodule:: pennylane_cirq.cirq_device

An abstract base class for constructing Cirq devices for PennyLane.
This abstract base class will not be used by the user.

Classes
-------

.. autosummary::
   CirqDevice

Code details
~~~~~~~~~~~~
"""
import abc
import cirq
import numpy as np
import pennylane as qml
from pennylane import QubitDevice
from pennylane.operation import Operation

from ._version import __version__
from .cirq_operation import CirqOperation


class CirqDevice(QubitDevice, abc.ABC):
    """Abstract base device for PennyLane-Cirq.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need to be >= 1.
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """

    name = "Cirq Abstract PennyLane plugin baseclass"
    pennylane_requires = ">=0.9.0"
    version = __version__
    author = "Xanadu Inc"
    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

    short_name = "cirq.base_device"

    def __init__(self, wires, shots, analytic, qubits=None):
        super().__init__(wires, shots, analytic)

        self.circuit = None

        if qubits:
            if wires != len(qubits):
                raise qml.DeviceError(
                    "The number of given qubits and the specified number of wires have to match. Got {} wires and {} qubits.".format(
                        wires, len(qubits)
                    )
                )

            self.qubits = qubits
        else:
            self.qubits = [cirq.LineQubit(wire) for wire in range(wires)]

        # Add inverse operations
        self._inverse_operation_map = {}
        for key in self._operation_map:
            if not self._operation_map[key]:
                continue

            # We have to use a new CirqOperation instance because .inv() acts in-place
            inverted_operation = CirqOperation(self._operation_map[key].parametrization)
            inverted_operation.inv()

            self._inverse_operation_map[key + Operation.string_for_inverse] = inverted_operation

        self._complete_operation_map = {
            **self._operation_map,
            **self._inverse_operation_map,
        }

    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": CirqOperation(cirq.MatrixGate),
        "PauliX": CirqOperation(lambda: cirq.X),
        "PauliY": CirqOperation(lambda: cirq.Y),
        "PauliZ": CirqOperation(lambda: cirq.Z),
        "Hadamard": CirqOperation(lambda: cirq.H),
        "S": CirqOperation(lambda: cirq.S),
        "T": CirqOperation(lambda: cirq.T),
        "CNOT": CirqOperation(lambda: cirq.CNOT),
        "SWAP": CirqOperation(lambda: cirq.SWAP),
        "CZ": CirqOperation(lambda: cirq.CZ),
        "PhaseShift": CirqOperation(lambda phi: cirq.ZPowGate(exponent=phi / np.pi)),
        "RX": CirqOperation(cirq.rx),
        "RY": CirqOperation(cirq.ry),
        "RZ": CirqOperation(cirq.rz),
        "Rot": CirqOperation(lambda a, b, c: [cirq.rz(a), cirq.ry(b), cirq.rz(c)]),
        "CRX": CirqOperation(lambda phi: cirq.ControlledGate(cirq.rx(phi))),
        "CRY": CirqOperation(lambda phi: cirq.ControlledGate(cirq.ry(phi))),
        "CRZ": CirqOperation(lambda phi: cirq.ControlledGate(cirq.rz(phi))),
        "CRot": CirqOperation(
            lambda a, b, c: [
                cirq.ControlledGate(cirq.rz(a)),
                cirq.ControlledGate(cirq.ry(b)),
                cirq.ControlledGate(cirq.rz(c)),
            ]
        ),
        "CSWAP": CirqOperation(lambda: cirq.CSWAP),
        "Toffoli": CirqOperation(lambda: cirq.TOFFOLI),
    }

    _observable_map = {
        "PauliX": None,
        "PauliY": None,
        "PauliZ": None,
        "Hadamard": None,
        "Hermitian": None,
        "Identity": None,
    }

    def reset(self):
        # pylint: disable=missing-function-docstring
        super().reset()

        self.circuit = cirq.Circuit()

    @property
    def observables(self):
        # pylint: disable=missing-function-docstring
        return set(self._observable_map.keys())

    @property
    def operations(self):
        # pylint: disable=missing-function-docstring
        return set(self._operation_map.keys())

    @abc.abstractmethod
    def _apply_basis_state(self, basis_state_operation):
        """Apply a basis state preparation.

        Args:
            basis_state_operation (pennylane.BasisState): the BasisState operation instance that shall be applied

        Raises:
            NotImplementedError: when not implemented in the subclass
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_qubit_state_vector(self, qubit_state_vector_operation):
        """Apply a state vector preparation.

        Args:
            qubit_state_vector_operation (pennylane.QubitStateVector): the QubitStateVector operation instance that shall be applied

        Raises:
            NotImplementedError: when not implemented in the subclass
        """
        raise NotImplementedError

    def _apply_operation(self, operation):
        """Apply a single PennyLane Operation.

        Args:
            operation (pennylane.Operation): The operation that shall be applied
        """
        cirq_operation = self._complete_operation_map[operation.name]

        # If command is None do nothing
        if cirq_operation:
            cirq_operation.parametrize(*operation.parameters)

            self.circuit.append(
                cirq_operation.apply(*[self.qubits[wire] for wire in operation.wires])
            )

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        rotations = kwargs.pop("rotations", [])

        for i, operation in enumerate(operations):
            if i > 0 and operation.name in {"BasisState", "QubitStateVector"}:
                raise qml.DeviceError(
                    "The operation {} is only supported at the beginning of a circuit.".format(
                        operation.name
                    )
                )

            if operation.name == "BasisState":
                self._apply_basis_state(operation)
            elif operation.name == "QubitStateVector":
                self._apply_qubit_state_vector(operation)
            else:
                self._apply_operation(operation)

        # TODO: get pre rotated state here

        # Diagonalize the given observables
        for operation in rotations:
            self._apply_operation(operation)
