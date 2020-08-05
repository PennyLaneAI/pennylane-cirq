# Copyright 2020 Xanadu Quantum Technologies Inc.

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
This module provides the ``QSimDevice`` and ``QSimhDevice`` from Cirq.
"""
import cirq

try:
    import qsimcirq
except ImportError:
    raise ImportError(
        "QSim Cirq is needed for the QSim devices to work."
        "\nIt can be installed using pip:"
        "\n\npip install qsimcirq"
    )

from .simulator_device import SimulatorDevice
from .cirq_device import CirqDevice


class QSimDevice(SimulatorDevice):
    r"""QSim device for PennyLane.
    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
    """
    name = "QSim device for PennyLane"
    short_name = "cirq.qsim"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None):
        super().__init__(wires, shots, analytic, qubits)
        self.circuit = qsimcirq.QSimCircuit(cirq_circuit=cirq.Circuit())
        self._simulator = qsimcirq.QSimSimulator()

        self._initial_state = None
        self._result = None
        self._state = None


class QSimhDevice(SimulatorDevice):
    r"""QSimh device for PennyLane.
    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Shots need
            to be >= 1. In analytic mode, shots indicates the number of entries
            that are returned by ``device.sample``.
        analytic (bool): indicates whether expectation values and variances should
            be calculated analytically
        qubits (List[cirq.Qubit]): a list of Cirq qubits that are used
            as wires. The wire number corresponds to the index in the list.
            By default, an array of ``cirq.LineQubit`` instances is created.
        qsimh_options (dict): a dictionary with options for the QSimh simulator. See the `QSim
            usage documentation <https://github.com/quantumlib/qsim/blob/master/docs/usage.md>`__
            for further details.
    """
    name = "QSimh device for PennyLane"
    short_name = "cirq.qsimh"

    def __init__(self, wires, shots=1000, analytic=True, qubits=None, qsimh_options=None):
        super().__init__(wires, shots, analytic, qubits)
        if qsimh_options is None:
            qsimh_options = {
                'k': [0],
                'w': 0,
                'p': 0,
                'r': 1
            }
        self.circuit = cirq.Circuit()
        self._simulator = qsimcirq.QSimhSimulator(qsimh_options)

        self._initial_state = None
        self._result = None
        self._state = None

    def apply(self, operations, **kwargs):
        CirqDevice.apply(self, operations, **kwargs)

        # We apply an identity gate to all wires, otherwise Cirq would ignore
        # wires that are not acted upon
        for qb in self.qubits:
            self.circuit.append(cirq.IdentityGate(1)(qb))

        if self.analytic:
            self._state = self._simulator.compute_amplitudes(
                program=self.circuit,
                bitstrings=list(range(2 ** len(self.wires)))
            )
