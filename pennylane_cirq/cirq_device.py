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
import cirq
import numpy as np
import pennylane as qml
from pennylane import Device

from ._version import __version__
from .cirq_interface import CirqOperation, unitary_matrix_gate


class CirqDevice(Device):
    """Abstract base device for PennyLane-Cirq.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need 
            to >= 1.
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used 
            as wires. The wire number corresponds to the index in the list.
            By default, an array of `cirq.LineQubit` instances is created.
    """

    name = "Cirq Abstract PennyLane plugin baseclass"
    pennylane_requires = ">=0.6.0"
    version = __version__
    author = "Johannes Jakob Meyer"

    short_name = "cirq.base_device"

    @staticmethod
    def _convert_measurements(measurements, eigenvalues):
        r"""Convert measurements from boolean to numeric values.

        Args:
            measurements (np.array[bool]): the measurements as boolean values
            eigenvalues (np.array[float]): eigenvalues corresponding to the observed basis states
        
        Returns:
            (np.array[float]): the converted measurements
        """
        N = measurements.shape[0]

        indices = np.ravel_multi_index(measurements, [2] * N)
        converted_measurements = eigenvalues[indices]

        return converted_measurements

    def __init__(self, wires, shots, qubits=None):
        super().__init__(wires, shots)

        self._eigs = dict()
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

    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": CirqOperation(unitary_matrix_gate),
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
        "RX": CirqOperation(lambda phi: cirq.Rx(phi)),
        "RY": CirqOperation(lambda phi: cirq.Ry(phi)),
        "RZ": CirqOperation(lambda phi: cirq.Rz(phi)),
        "Rot": CirqOperation(lambda a, b, c: [cirq.Rz(a), cirq.Ry(b), cirq.Rz(c)]),
        "CRX": CirqOperation(lambda phi: cirq.ControlledGate(cirq.Rx(phi))),
        "CRY": CirqOperation(lambda phi: cirq.ControlledGate(cirq.Ry(phi))),
        "CRZ": CirqOperation(lambda phi: cirq.ControlledGate(cirq.Rz(phi))),
        "CRot": CirqOperation(
            lambda a, b, c: [
                cirq.ControlledGate(cirq.Rz(a)),
                cirq.ControlledGate(cirq.Ry(b)),
                cirq.ControlledGate(cirq.Rz(c)),
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
        self.circuit = cirq.Circuit()

    @property
    def observables(self):
        return set(self._observable_map.keys())

    @property
    def operations(self):
        return set(self._operation_map.keys())

    def pre_apply(self):
        self.reset()

    def apply(self, operation, wires, par):
        operation = self._operation_map[operation]

        # If command is None do nothing
        if operation:
            operation.parametrize(*par)

            self.circuit.append(operation.apply(*[self.qubits[wire] for wire in wires]))

    def pre_measure(self):
        # Cirq only measures states in the computational basis, i.e. 0 and 1
        # To measure different observables, we have to go to their eigenbases

        # This code is adapted from the pennylane-qiskit plugin
        for e in self.obs_queue:
            # Identity and PauliZ need no changes
            if e.name == "PauliX":
                # X = H.Z.H
                self.apply("Hadamard", wires=e.wires, par=[])

            elif e.name == "PauliY":
                # Y = (HS^)^.Z.(HS^) and S^=SZ
                self.apply("PauliZ", wires=e.wires, par=[])
                self.apply("S", wires=e.wires, par=[])
                self.apply("Hadamard", wires=e.wires, par=[])

            elif e.name == "Hadamard":
                # H = Ry(-pi/4)^.Z.Ry(-pi/4)
                self.apply("RY", e.wires, [-np.pi / 4])

            elif e.name == "Hermitian":
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                Hmat = e.parameters[0]
                Hkey = tuple(Hmat.flatten().tolist())

                if Hmat.shape not in [(2, 2), (4, 4)]:
                    raise qml.DeviceError(
                        "Cirq only supports single-qubit and two-qubit unitary gates and thus only single-qubit and two-qubit Hermitian observables."
                    )

                if Hkey in self._eigs:
                    # retrieve eigenvectors
                    U = self._eigs[Hkey]["eigvec"]
                else:
                    # store the eigenvalues corresponding to H
                    # in a dictionary, so that they do not need to
                    # be calculated later
                    w, U = np.linalg.eigh(Hmat)
                    self._eigs[Hkey] = {"eigval": w, "eigvec": U}

                # Perform a change of basis before measuring by applying U^ to the circuit
                self.apply("QubitUnitary", e.wires, [U.conj().T])

            # No measurements are added here because they can't be added for simulations
