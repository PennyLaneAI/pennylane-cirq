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
    BitFlip
    PhaseFlip
    PhaseDamp
    AmplitudeDamp
    Depolarize


Code details
~~~~~~~~~~~~
"""
from pennylane.operation import Operation

# pylint: disable=missing-function-docstring


class BitFlip(Operation):
    """Cirq ``bit_flip`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.bit_flip.html>`_
    for further details."""

    num_params = 1
    num_wires = 1
    par_domain = "R"

    grad_method = None
    grad_recipe = None


class PhaseFlip(Operation):
    """Cirq ``phase_flip`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.phase_flip.html>`_
    for further details."""

    num_params = 1
    num_wires = 1
    par_domain = "R"

    grad_method = None
    grad_recipe = None


class PhaseDamp(Operation):
    """Cirq ``phase_damp`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.phase_damp.html>`_
    for further details."""

    num_params = 1
    num_wires = 1
    par_domain = "R"

    grad_method = None
    grad_recipe = None


class AmplitudeDamp(Operation):
    """Cirq ``amplitude_damp`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.amplitude_damp.html>`_
    for further details."""

    num_params = 1
    num_wires = 1
    par_domain = "R"

    grad_method = None
    grad_recipe = None


class Depolarize(Operation):
    """Cirq ``depolarize`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.depolarize.html>`_
    for further details."""

    num_params = 1
    num_wires = 1
    par_domain = "R"

    grad_method = None
    grad_recipe = None


class ISWAP(Operation):
    """Cirq ``ISWAP`` operation.

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.ISWAP.html>`_
    for further details."""

    num_params = 0
    num_wires = 2
    par_domain = None

    grad_method = None
    grad_recipe = None


class CPhase(Operation):
    r"""Conditional phase operation following PennyLane conventions.

    Implemented as Cirq ``CZPowGate(exponent=phi / np.pi)``.

    .. math::

        CPhase(\phi) =
            \begin{bmatrix}
                1 & 0 & 0 & 0\\
                0 & 1 & 0 & 0\\
                0 & 0 & 1 & 0\\
                0 & 0 & 0 & e^{i\phi}
            \end{bmatrix}

    See the `Cirq docs <https://cirq.readthedocs.io/en/stable/generated/cirq.CZPowGate.html>`_
    for further details."""

    num_params = 1
    num_wires = 2
    par_domain = "R"

    grad_method = None
    grad_recipe = None
