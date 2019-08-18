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
Device 1
========

**Module name:** :mod:`plugin_name.Device2`

.. currentmodule:: plugin_name.Device2

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Target Framework device/simulator as a PennyLane device.

It can inherit from the abstract FrameworkDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   Device2

----
"""

# we always import NumPy directly
import numpy as np

import TargetFramework as tf

from .framework_device import FrameworkDevice


class Device2(FrameworkDevice):
    r"""Device2 for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        additional_option (float): as many additional arguments can be
            added as needed
        specific_option_for_Device2 (int): another example
    """
    name = "Target Framework Device2 for PennyLane"
    short_name = "plugin_name.device2"

    _operation_map = {
        "PauliX": tf.X,
        "PauliY": tf.Y,
        "PauliZ": tf.Z,
        "Hadamard": tf.H,
        "CNOT": tf.CNOT,
        "SWAP": tf.SWAP,
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    _circuits = {}

    def __init__(self, wires, *, additional_option, shots=0, specific_option_for_Device2=2):
        super().__init__(wires, shots=shots, additional_option=additional_option)
        self.specific_option_for_Device2 = specific_option_for_Device2

    def apply(self, operation, wires, par):
        pass

    def expval(self, observable, wires, par):
        pass

    def var(self, observable, wires, par):
        pass

    def sample(self, observable, wires, par, n=None):
        pass
