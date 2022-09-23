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
from collections.abc import Iterable  # pylint: disable=no-name-in-module
from collections import OrderedDict
import functools
import operator

import cirq
import numpy as np
import pennylane as qml
from pennylane import QubitDevice
from pennylane.operation import Tensor
from pennylane.wires import Wires

from ._version import __version__
from .cirq_operation import CirqOperation


class CirqDevice(QubitDevice, abc.ABC):
    """Abstract base device for PennyLane-Cirq.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need to be >= 1.
        qubits (List[cirq.Qubit]): A list of Cirq qubits that are used
            as wires. By default, an array of ``cirq.LineQubit`` instances is created.
            Wires are mapped to qubits using Cirq's internal mechanism for ordering
            qubits. For example, if ``wires=2`` and ``qubits=[q1, q2]``, with
            ``q1>q2``, then the wire indices 0 and 1 are mapped to q2 and q1, respectively.
            If the user provides their own wire labels, e.g., ``wires=["alice", "bob"]``, and the
            qubits are the same as the previous example, then "alice" would map to qubit q2
            and "bob" would map to qubit q1.
    """

    name = "Cirq Abstract PennyLane plugin base class"
    pennylane_requires = ">=0.17.0"
    version = __version__
    author = "Xanadu Inc"
    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

    short_name = "cirq.base_device"

    def __init__(self, wires, shots, qubits=None):

        if not isinstance(wires, Iterable):
            # interpret wires as the number of consecutive wires
            wires = range(wires)
        num_wires = len(wires)

        if qubits:
            if num_wires != len(qubits):
                raise qml.DeviceError(
                    f"The number of given qubits and the specified number of wires have to match. Got {wires} wires and {len(qubits)} qubits."
                )
        else:
            qubits = [cirq.LineQubit(idx) for idx in range(num_wires)]

        # cirq orders the subsystems based on a total order defined on qubits.
        # For consistency, this plugin uses that same total order
        self._unsorted_qubits = qubits
        self.qubits = sorted(qubits)

        super().__init__(wires, shots)

        self.circuit = None
        self.pre_rotated_circuit = None
        self.cirq_device = None

        # Add inverse operations
        self._inverse_operation_map = {}
        for key, value in self._operation_map.items():
            if not value:
                continue

            # We have to use a new CirqOperation instance because .inv() acts in-place
            inverted_operation = CirqOperation(value.parametrization)
            inverted_operation.inv()

            self._inverse_operation_map[key + ".inv"] = inverted_operation

        self._complete_operation_map = {
            **self._operation_map,
            **self._inverse_operation_map,
        }

    _pow_operation_map = {
        "PauliX": CirqOperation(lambda exp: cirq.XPowGate(exponent=exp)),
        "PauliY": CirqOperation(lambda exp: cirq.YPowGate(exponent=exp)),
        "PauliZ": CirqOperation(lambda exp: cirq.ZPowGate(exponent=exp)),
        "Hadamard": CirqOperation(lambda exp: cirq.HPowGate(exponent=exp)),
        "SWAP": CirqOperation(lambda exp: cirq.SwapPowGate(exponent=exp)),
        "ISWAP": CirqOperation(lambda exp: cirq.ISwapPowGate(exponent=exp)),
        "CNOT": CirqOperation(lambda exp: cirq.CXPowGate(exponent=exp)),
        "CZ": CirqOperation(lambda exp: cirq.CZPowGate(exponent=exp)),
    }

    _operation_map = {
        **{f"Pow({k})": v for k, v in _pow_operation_map.items()},
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
        "ISWAP": CirqOperation(lambda: cirq.ISWAP),
        "Adjoint(ISWAP)": CirqOperation(lambda: cirq.ISWAP_INV),
        "SISWAP": CirqOperation(lambda: cirq.SQRT_ISWAP),
        "Adjoint(SISWAP)": CirqOperation(lambda: cirq.SQRT_ISWAP_INV),
        "CZ": CirqOperation(lambda: cirq.CZ),
        "PhaseShift": CirqOperation(lambda phi: cirq.ZPowGate(exponent=phi / np.pi)),
        "ControlledPhaseShift": CirqOperation(lambda phi: cirq.CZPowGate(exponent=phi / np.pi)),
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
        "PauliX": CirqOperation(lambda: cirq.X),
        "PauliY": CirqOperation(lambda: cirq.Y),
        "PauliZ": CirqOperation(lambda: cirq.Z),
        "Hadamard": CirqOperation(lambda: cirq.H),
        "Hermitian": None,
        # TODO: Consider using qml.utils.decompose_hamiltonian() to support this observable.
        "Identity": CirqOperation(lambda: cirq.I),
        "Projector": CirqOperation(lambda: cirq.ProductState.projector),
    }

    def supports_operation(self, operation):
        # pylint: disable=missing-function-docstring
        if isinstance(operation, str):
            op_with_power = operation.split("**")
            operation = f"Pow({op_with_power[0]})" if len(op_with_power) == 2 else operation
        return super().supports_operation(operation)

    def to_paulistring(self, observable):
        """Convert an observable to a cirq.PauliString"""
        if isinstance(observable, Tensor):
            obs = [self.to_paulistring(o) for o in observable.obs]
            return functools.reduce(operator.mul, obs)
        cirq_op = self._observable_map[observable.name]
        if cirq_op is None:
            raise NotImplementedError(f"{observable.name} is not currently supported.")
        cirq_op.parametrize(*observable.parameters)
        device_wires = self.map_wires(observable.wires)
        return functools.reduce(
            operator.mul, cirq_op.apply(*[self.qubits[w] for w in device_wires.labels])
        )

    def reset(self):
        # pylint: disable=missing-function-docstring
        super().reset()

        if self.cirq_device:
            self.circuit = cirq.Circuit(device=self.cirq_device)
        else:
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
            operation (pennylane.Operation): the operation that shall be applied
        """
        if isinstance(operation, qml.ops.Pow):
            op_name = f"Pow({operation.base.name})"
            params = [operation.z, *operation.parameters]
        else:
            op_name = operation.name
            params = operation.parameters

        cirq_operation = self._complete_operation_map[op_name]

        # If command is None do nothing
        if cirq_operation:
            cirq_operation.parametrize(*params)

            device_wires = self.map_wires(operation.wires)
            self.circuit.append(
                cirq_operation.apply(*[self.qubits[w] for w in device_wires.labels])
            )

    def apply(self, operations, **kwargs):
        # pylint: disable=missing-function-docstring
        rotations = kwargs.pop("rotations", [])

        for i, operation in enumerate(operations):
            if i > 0 and operation.name in {"BasisState", "QubitStateVector"}:
                raise qml.DeviceError(
                    f"The operation {operation.name} is only supported at the beginning of a circuit."
                )

            if operation.name == "BasisState":
                self._apply_basis_state(operation)
            elif operation.name == "QubitStateVector":
                self._apply_qubit_state_vector(operation)
            else:
                self._apply_operation(operation)

        self.pre_rotated_circuit = self.circuit.copy()

        # Diagonalize the given observables
        for operation in rotations:
            self._apply_operation(operation)

    def define_wire_map(self, wires):  # pylint: disable=missing-function-docstring
        cirq_order = np.argsort(self._unsorted_qubits)
        consecutive_wires = Wires(cirq_order)

        wire_map = zip(wires, consecutive_wires)
        return OrderedDict(wire_map)
