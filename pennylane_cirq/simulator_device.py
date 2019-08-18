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
SimulatorDevice
========

**Module name:** :mod:`pennylane_cirq.simulator_device`

.. currentmodule:: pennylane_cirq.simulator_device

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Target Framework device/simulator as a PennyLane device.

It can inherit from the abstract FrameworkDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   SimulatorDevice

----
"""
# we always import NumPy directly
import numpy as np

import cirq
from .cirq_device import CirqDevice


class SimulatorDevice(CirqDevice):
    r"""Cirq simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
    """
    name = "Cirq Simulator device for PennyLane"
    short_name = "cirq.simulator"
    
    def __init__(self, wires, shots=0, qubits=None):
        super().__init__(wires, shots, qubits)
        
        self.simulator = cirq.Simulator()

    
    def expval(self, observable, wires, par):
        return 1

        if self.shots == 0:
            return self.simulator.simulate(self.circuit) # TODO
        else:
            return self.sample(observable, wires, par, n=self.shots).mean()
            
    def var(self, observable, wires, par):
        return 0

        if self.shots == 0:
            return self.simulator.simulate(self.circuit) # TODO
        else:
            return self.sample(observable, wires, par, n=self.shots).var()

    def sample(self, observable, wires, par, n=None):
        sample = self.simulator.run(self.circuit, repetitions=n)
