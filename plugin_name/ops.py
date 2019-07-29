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

**Module name:** :mod:`pennylane_forest.ops`

.. currentmodule:: pennylane_forest.ops

Sometimes the Target Framework may accept more operations
than available by core PennyLane. The plugin can define
these operations such that PennyLane can understand/apply them,
and even differentiate them.

This module contains some example PennyLane qubit operations.

The user would import them via

.. code-block:: python

    from plugin_name.ops import S, T, CCNOT

To see more details about defining custom PennyLane operations,
including more advanced cases such as defining gradient rules,
see https://pennylane.readthedocs.io/en/latest/API/overview.html

Operations
----------

.. autosummary::
    S
    T
    CCNOT
    CPHASE
    CSWAP
    ISWAP
    PSWAP


Code details
~~~~~~~~~~~~
"""
from pennylane.operation import Operation


class S(Operation):
    r"""S(wires)
    S gate.

    .. math:: S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class T(Operation):
    r"""T(wires)
    T gate.

    .. math:: T = \begin{bmatrix}1&0\\0&e^{i \pi / 4}\end{bmatrix}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None


class CCNOT(Operation):
    r"""CCNOT(wires)
    Controlled-controlled-not gate.

    .. math::

        CCNOT = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{bmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None


class CPHASE(Operation):
    r"""CHPASE(phi, q, wires)
    Controlled-phase gate.

    .. math::

        CPHASE_{ij}(phi, q) = \begin{cases}
            0, & i\neq j\\
            1, & i=j, i\neq q\\
            e^{i\phi}, & i=j=q
        \end{cases}\in\mathbb{C}^{4\times 4}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{d\phi}CPHASE(\phi) = \frac{1}{2}\left[CPHASE(\phi+\pi/2)+CPHASE(\phi-\pi/2)\right]`
      Note that the gradient recipe only applies to parameter :math:`\phi`.
      Parameter :math:`q\in\mathbb{N}_0` and thus ``CPHASE`` can not be differentiated
      with respect to :math:`q`.

    Args:
        phi (float): the controlled phase angle
        q (int): an integer between 0 and 3 that corresponds to a state
            :math:`\{00, 01, 10, 11\}` on which the conditional phase
            gets applied
        wires (int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 2
    par_domain = "R"
    grad_method = "A"


class CSWAP(Operation):
    r"""CSWAP(wires)
    Controlled-swap gate.

    .. math::

        CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
             0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
             0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
             0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
             0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
             0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
             0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
             0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None


class ISWAP(Operation):
    r"""ISWAP(wires)
    iSWAP gate.

    .. math:: ISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0\\
            0 & i & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None


class PSWAP(Operation):
    r"""PSWAP(wires)
    Phase-SWAP gate.

    .. math:: PSWAP(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i\phi} & 0\\
            0 & e^{i\phi} & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}PSWAP(\phi) = \frac{1}{2}\left[PSWAP(\phi+\pi/2)+PSWAP(\phi-\pi/2)\right]`


    Args:
        wires (int): the subsystem the gate acts on
        phi (float): the phase angle
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
