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

class CompositeCirqCommand:
    """A helper class that wraps multiple native Cirq commands and provides an 
       interface for parametrization and application."""

    def __init__(self, cirq_gate_map):
        """Initializes the CirqCommand.

        Args:
            cirq_gate (List[Tuple(Cirq:Qid, List[int])]): A list of tuples consisting of a Cirq gate and a list of 
                indices indicating which parameters are used for the gate parameterization. The gates are applied in the
                order given.
        """
        self.cirq_gate_map = cirq_gate_map
        self.parametrized_cirq_gates = []

    def parametrize(self, *args):
        """Parametrizes the CompositeCirqCommand.

        Args:
            *qubits (Cirq:Qid): the qubits on which the Cirq gate should be performed.
        """
        self.parametrized_cirq_gates = []
        
        for gate, arg_indices in self.cirq_gate_map:
            if arg_indices:
                self.parametrized_cirq_gates.append(gate(*[args[idx] for idx in arg_indices]))
            else:
                self.parametrized_cirq_gates.append(gate)

    def apply(self, *qubits):
        """Parametrizes the CompositeCirqCommand.

        Args:
            *qubits (Cirq:Qid): the qubits on which the Cirq gate should be performed.
        """
        if not self.parametrized_cirq_gates:
            raise qml.DeviceError("Command must be parametrized before it can be applied.")

        return (parametrized_gate(*qubits) for parametrized_gate in self.parametrized_cirq_gates)

class CirqCommand(CompositeCirqCommand):
    """A helper class that wraps the native Cirq commands and provides an 
       interface for parametrization and application."""

    def __init__(self, cirq_gate, num_params=0):
        """Initializes the CirqCommand.

        Args:
            cirq_gate (Cirq:Qid): the Cirq gate to be wrapped
            is_parametrized (Bool): Indicates if the Cirq gate is parametrized
        """
        if num_params == 0:
            param_indices = []
        else:
            param_indices = range(num_params)

        super().__init__([(cirq_gate, param_indices)])

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

    @staticmethod
    def _convert_measurements(measurements, zero_value, one_value):
        conversion = np.vectorize(lambda x: one_value if x else zero_value)

        return conversion(measurements.flatten())

    def __init__(self, wires, shots, qubits=None):
        super().__init__(wires, shots)

        self._eigs = dict()
        
        if qubits:
            if wires != len(qubits):
                raise qml.DeviceError("The number of given qubits and the specified number of wires have to match. Got {} wires and {} qubits.".format(wires, len(qubits)))

            self.qubits = qubits
        else:
            self.qubits = [cirq.LineQubit(wire) for wire in range(wires)]


    _operation_map = {
        "BasisState": None,
        "QubitStateVector": None,
        "QubitUnitary": CirqCommand(cirq.SingleQubitMatrixGate, 1),
        "PauliX": CirqCommand(cirq.X),
        "PauliY": CirqCommand(cirq.Y),
        "PauliZ": CirqCommand(cirq.Z),
        "Hadamard": CirqCommand(cirq.H),
        "S": CirqCommand(cirq.S),
        "CNOT": CirqCommand(cirq.CNOT),
        "SWAP": CirqCommand(cirq.SWAP),
        "CZ": CirqCommand(cirq.CZ),
        "PhaseShift": None,
        "RX": CirqCommand(cirq.Rx, 1),
        "RY": CirqCommand(cirq.Ry, 1),
        "RZ": CirqCommand(cirq.Rz, 1),
        "Rot": CompositeCirqCommand([(cirq.Rz, [0]), (cirq.Ry, [1]), (cirq.Rz, [2])]),
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

        # If command is None do nothing
        if command:
            command.parametrize(*par)

            self.circuit.append(command.apply(*[self.qubits[wire] for wire in wires]))

    def post_apply(self):
        pass

    def pre_measure(self):
        # Cirq only measures states in the computational basis, i.e. 0 and 1
        # To measure different observables, we have to go to their eigenbases

        # This code is adapted from the pennylane-qiskit plugin
        for e in self.obs_queue:
            wire = e.wires[0]
            
            # Identity and PauliZ need no changes
            if e.name == "PauliX":
                # X = H.Z.H
                self.apply("Hadamard", wires=[wire], par=[])

            elif e.name == "PauliY":
                # Y = (HS^)^.Z.(HS^) and S^=SZ
                self.apply("PauliZ", wires=[wire], par=[])
                self.apply("S", wires=[wire], par=[])
                self.apply("Hadamard", wires=[wire], par=[])

            elif e.name == "Hadamard":
                # H = Ry(-pi/4)^.Z.Ry(-pi/4)
                self.apply("RY", [wire], [-np.pi / 4])

            elif e.name == "Hermitian":
                # For arbitrary Hermitian matrix H, let U be the unitary matrix
                # that diagonalises it, and w_i be the eigenvalues.
                Hmat = e.parameters[0]
                Hkey = tuple(Hmat.flatten().tolist())

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
                self.apply("QubitUnitary", [wire], [U.conj().T])

            # No measurements are added here because they can't be added for simulations