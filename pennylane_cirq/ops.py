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
Custom operations
=================

**Module name:** :mod:`pennylane_cirq.ops`

.. currentmodule:: pennylane_cirq.ops

Sometimes the Target Framework may accept more operations
than available by core PennyLane. The plugin can define
these operations such that PennyLane can understand/apply them,
and even differentiate them.

This module contains some example PennyLane qubit operations.

The user would import them via

.. code-block:: python

    from pennylane_cirq.ops import S, T, CCNOT

To see more details about defining custom PennyLane operations,
including more advanced cases such as defining gradient rules,
see https://pennylane.readthedocs.io/en/latest/API/overview.html

Operations
----------

.. autosummary::
    S
    T
    CCNOT
    CSWAP


Code details
~~~~~~~~~~~~
"""
# TODO[CUSTOM OPS]: Uncomment and replace with Cirq-specific ops
# from pennylane.operation import Operation


# class S(Operation):
#     r"""S(wires)
#     S gate.

#     .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

#     **Details:**

#     * Number of wires: 1
#     * Number of parameters: 0

#     Args:
#         wires (int): the subsystem the gate acts on
#     """
#     num_params = 0
#     num_wires = 1
#     par_domain = None


# class T(Operation):
#     r"""T(wires)
#     T gate.

#     .. math:: T = \begin{bmatrix}1&0\\0&e^{i \pi / 4}\end{bmatrix}

#     **Details:**

#     * Number of wires: 1
#     * Number of parameters: 0

#     Args:
#         wires (int): the subsystem the gate acts on
#     """
#     num_params = 0
#     num_wires = 1
#     par_domain = None


# class CCNOT(Operation):
#     r"""CCNOT(wires)
#     Controlled-controlled-not gate.

#     .. math::

#         CCNOT = \begin{bmatrix}
#             1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
#             0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
#             0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
#             0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
#             0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
#             0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
#             0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
#             0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
#         \end{bmatrix}

#     **Details:**

#     * Number of wires: 3
#     * Number of parameters: 0

#     Args:
#         wires (int): the subsystem the gate acts on
#     """
#     num_params = 0
#     num_wires = 3
#     par_domain = None


# class CSWAP(Operation):
#     r"""CSWAP(wires)
#     Controlled-swap gate.

#     .. math::

#         CSWAP = \begin{bmatrix}
#             1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
#              0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
#              0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
#              0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
#              0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
#              0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
#              0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
#              0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
#         \end{bmatrix}

#     **Details:**

#     * Number of wires: 3
#     * Number of parameters: 0

#     Args:
#         wires (int): the subsystem the gate acts on
#     """
#     num_params = 0
#     num_wires = 3
#     par_domain = None
