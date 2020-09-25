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
Unit tests for the QSimhDevice
"""
import pytest
import math

import pennylane as qml
import numpy as np
from pennylane_cirq import QSimhDevice
import cirq


class TestDeviceIntegration:
    """Tests that the QSimhDevice integrates well with PennyLane"""

    def test_device_loading(self):
        """Tests that the cirq.qsimh device is properly loaded"""

        dev = qml.device("cirq.qsimh", wires=2)

        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.short_name == "cirq.qsimh"

        assert isinstance(dev, QSimhDevice)


@pytest.fixture(scope="function")
def qsimh_device_1_wire(shots, analytic):
    """Return a single wire instance of the QSimhDevice class."""
    yield QSimhDevice(1, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def qsimh_device_2_wires(shots, analytic):
    """Return a two wire instance of the QSimhDevice class."""
    yield QSimhDevice(2, shots=shots, analytic=analytic)


@pytest.fixture(scope="function")
def qsimh_device_3_wires(shots, analytic):
    """Return a three wire instance of the QSimhDevice class."""
    yield QSimhDevice(3, shots=shots, analytic=analytic)


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestApply:
    """Tests that gates are correctly applied"""

    @pytest.mark.parametrize(
        "op,input,expected_output",
        [
            (qml.PauliX, [0], np.array([0, 1])),
            (qml.PauliY, [0], [0, 1j]),
            (qml.PauliZ, [0], [1, 0]),
            (qml.Hadamard, [0], [1 / math.sqrt(2), 1 / math.sqrt(2)]),
        ],
    )
    def test_apply_operation_single_wire_no_parameters(
        self, qsimh_device_1_wire, tol, op, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply([qml.BasisState(np.array(input), wires=[0, 1])])
        qsimh_device_1_wire.apply([op(wires=[0])])

        assert np.allclose(qsimh_device_1_wire.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,input,expected_output",
        [
            (qml.CNOT, [0, 0], [1, 0, 0, 0]),
            (qml.CNOT, [1, 0], [0, 0, 0, 1]),
            (qml.SWAP, [0, 0], [1, 0, 0, 0]),
            (qml.SWAP, [1, 0], [0, 1, 0, 0]),
            (qml.CZ, [0, 0], [1, 0, 0, 0]),
            (qml.CZ, [1, 1], [0, 0, 0, -1]),
        ],
    )
    def test_apply_operation_two_wires_no_parameters(
        self, qsimh_device_2_wires, tol, op, input, expected_output
    ):
        """Tests that applying an operation yields the expected output state for two wire
           operations that have no parameters."""

        qsimh_device_2_wires.reset()
        qsimh_device_2_wires.apply([qml.BasisState(np.array(input), wires=[0, 1])])
        qsimh_device_2_wires.apply([op(wires=[0, 1])])

        assert np.allclose(qsimh_device_2_wires.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,expected_output,par",
        [
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 1, 0], [1, 0]),
            (qml.BasisState, [0, 0, 0, 1], [1, 1]),
        ],
    )
    def test_apply_operation_state_preparation(
        self, qsimh_device_2_wires, tol, op, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qsimh_device_2_wires.reset()
        qsimh_device_2_wires.apply([op(np.array(par), wires=[0, 1])])

        assert np.allclose(qsimh_device_2_wires.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "op,input,expected_output,par",
        [
            (qml.PhaseShift, [0], [1, 0], [math.pi / 2]),
            (qml.PhaseShift, [1], [0, 1j], [math.pi / 2]),
            (qml.RX, [0], [1 / math.sqrt(2), -1j * 1 / math.sqrt(2)], [math.pi / 2 + 0.0000001]),
            (qml.RX, [0], [0, -1j], [math.pi+0.0000001]),
            (qml.RY, [0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [math.pi / 2 + 0.0000001]),
            (qml.RY, [0], [0, 1], [math.pi+0.0000001]),
            (qml.RZ, [0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2 + 0.0000001]),
            (qml.RZ, [1], [0, 1j], [math.pi+0.0000001]),
            (qml.Rot, [0], [1 / math.sqrt(2) - 1j / math.sqrt(2), 0], [math.pi / 2 + 0.0000001, 0, 0],),
            (qml.Rot, [0], [1 / math.sqrt(2), 1 / math.sqrt(2)], [0, math.pi / 2 + 0.0000001, 0],),
            (
                qml.Rot,
                [0],
                [-1j / math.sqrt(2), -1 / math.sqrt(2)],
                [math.pi / 2 + 0.0000001, -math.pi / 2 + 0.0000001, math.pi / 2 + 0.0000001],
            ),
            (
                qml.QubitUnitary,
                [0],
                [1j / math.sqrt(2), 1j / math.sqrt(2)],
                [
                    np.array(
                        [
                            [1j / math.sqrt(2), 1j / math.sqrt(2)],
                            [1j / math.sqrt(2), -1j / math.sqrt(2)],
                        ]
                    )
                ],
            ),
            (
                qml.QubitUnitary,
                [1],
                [1j / math.sqrt(2), -1j / math.sqrt(2)],
                [
                    np.array(
                        [
                            [1j / math.sqrt(2), 1j / math.sqrt(2)],
                            [1j / math.sqrt(2), -1j / math.sqrt(2)],
                        ]
                    )
                ],
            ),
        ],
    )
    def test_apply_operation_single_wire_with_parameters(
        self, qsimh_device_1_wire, tol, op, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
           operations that have no parameters."""

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply([qml.BasisState(np.array(input), wires=[0])])
        qsimh_device_1_wire.apply([op(*par, wires=[0])])

        assert np.allclose(qsimh_device_1_wire.state, np.array(expected_output), **tol)

    # FIXME: no Schmidt decomposition for gate kind 36.
    # qsimh simulation of the circuit errored out.
    #
    # @pytest.mark.parametrize(
    #     "op,input,expected_output,par",
    #     [
    #         (
    #             qml.QubitUnitary,
    #             [0, 0],
    #             [1, 0, 0, 0],
    #             [
    #                 np.array(
    #                     [
    #                         [1, 0, 0, 0],
    #                         [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
    #                         [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
    #                         [0, 0, 0, 1],
    #                     ]
    #                 )
    #             ],
    #         ),
    #         # Doesn't provide the correct answer with vanilla qsim.
    #         # Output is [0, -1 / math.sqrt(2), 1 / math.sqrt(2), 0]
    #         (
    #             qml.QubitUnitary,
    #             [0, 1],
    #             [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
    #             [
    #                 np.array(
    #                     [
    #                         [1, 0, 0, 0],
    #                         [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0],
    #                         [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
    #                         [0, 0, 0, 1],
    #                     ]
    #                 )
    #             ],
    #         ),
    #     ],
    # )
    # def test_apply_operation_two_wires_with_parameters(
    #     self, qsimh_device_2_wires, tol, op, input, expected_output, par
    # ):
    #     """Tests that applying an operation yields the expected output state for single wire
    #        operations that have no parameters."""

    #     qsimh_device_2_wires.reset()
    #     qsimh_device_2_wires.apply([qml.BasisState(np.array(input), wires=[0, 1])])
    #     qsimh_device_2_wires.apply([op(*par, wires=[0, 1])])
    #     # asdsa
    #     assert np.allclose(qsimh_device_2_wires.state, np.array(expected_output), **tol)

    @pytest.mark.parametrize(
        "operation,par,match",
        [
            (qml.BasisState, [2], "Argument for BasisState can only contain 0 and 1"),
            (qml.BasisState, [1.2], "Argument for BasisState can only contain 0 and 1"),
            (
                qml.BasisState,
                [0, 0, 1],
                "For BasisState, the state has to be specified for the correct number of qubits",
            ),
            (
                qml.BasisState,
                [0, 0],
                "For BasisState, the state has to be specified for the correct number of qubits",
            ),
        ],
    )
    def test_state_preparation_error(self, qsimh_device_1_wire, operation, par, match):
        """Tests that the state preparation routines raise proper errors for wrong parameter values."""

        qsimh_device_1_wire.reset()

        with pytest.raises(qml.DeviceError, match=match):
            qsimh_device_1_wire.apply([operation(np.array(par), wires=[0])])

    def test_basis_state_not_at_beginning_error(self, qsimh_device_1_wire):
        """Tests that application of BasisState raises an error if is not
        the first operation."""

        qsimh_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError,
            match="The operation BasisState is only supported at the beginning of a circuit.",
        ):
            qsimh_device_1_wire.apply([qml.PauliX(0), qml.BasisState(np.array([0]), wires=[0])])
@pytest.mark.parametrize("shots,analytic", [(100, False)])
class TestStatePreparationErrorsNonAnalytic:
    """Tests state preparation errors that occur for non-analytic devices."""

    def test_basis_state_not_analytic_error(self, qsimh_device_1_wire):
        """Tests that application of BasisState raises an error if the device
        is not in analytic mode."""

        qsimh_device_1_wire.reset()

        with pytest.raises(
            qml.DeviceError, match="The operation BasisState is only supported in analytic mode.",
        ):
            qsimh_device_1_wire.apply([qml.BasisState(np.array([0]), wires=[0])])


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestAnalyticProbability:
    """Tests the analytic_probability method works as expected."""

    def test_analytic_probability_is_none(self, qsimh_device_1_wire):
        """Tests that analytic_probability returns None if the state of the
        device is None."""

        qsimh_device_1_wire.reset()
        assert qsimh_device_1_wire._state is None
        assert qsimh_device_1_wire.analytic_probability() is None


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestExpval:
    """Tests that expectation values are properly calculated or that the proper errors are raised."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.Identity, [0], 1),
            (qml.Identity, [1], 1),
            (qml.PauliX, [0], 0),
            (qml.PauliY, [0], 0),
            (qml.PauliZ, [0], 1),
            (qml.PauliZ, [1], -1),
            (qml.Hadamard, [0], 1 / math.sqrt(2)),
            (qml.Hadamard, [1], -1 / math.sqrt(2)),
        ],
    )
    def test_expval_single_wire_no_parameters(
        self, qsimh_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that expectation values are properly calculated for single-wire observables without parameters."""

        op = operation(0, do_queue=False)

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply(
            [qml.BasisState(np.array(input), wires=[0])], rotations=op.diagonalizing_gates()
        )

        res = qsimh_device_1_wire.expval(op)

        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Hermitian, [0], 1, [np.array([[1, 1j], [-1j, 1]])]),
            (qml.Hermitian, [1], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ],
    )
    def test_expval_single_wire_with_parameters(
        self, qsimh_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        op = operation(par[0], 0, do_queue=False)

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply(
            [qml.BasisState(np.array(input), wires=[0])], rotations=op.diagonalizing_gates()
        )

        res = qsimh_device_1_wire.expval(op)

        assert np.isclose(res, expected_output, **tol)

    # FIXME: no Schmidt decomposition for gate kind 36.
    # qsimh simulation of the circuit errored out.
    #
    # @pytest.mark.parametrize(
    #     "operation,input,expected_output,par",
    #     [
    #         # Doesn't provide the correct answer with vanilla qsim.
    #         # Output is [0, -1 / math.sqrt(2), 1 / math.sqrt(2), 0]
    #         (
    #             qml.Hermitian,
    #             [0, 1],
    #             -1,
    #             [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1],])],
    #         ),
    #         (
    #             qml.Hermitian,
    #             [1, 1],
    #             0,
    #             [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])],
    #         ),
    #     ],
    # )
    # def test_expval_two_wires_with_parameters(
    #     self, qsimh_device_2_wires, tol, operation, input, expected_output, par
    # ):
    #     """Tests that expectation values are properly calculated for two-wire observables with parameters."""

    #     op = operation(par[0], [0, 1], do_queue=False)

    #     qsimh_device_2_wires.reset()
    #     qsimh_device_2_wires.apply(
    #         [qml.BasisState(np.array(input), wires=[0, 1])],
    #         rotations=op.diagonalizing_gates(),
    #     )

    #     sdfsd
    #     res = qsimh_device_2_wires.expval(op)
    #     assert np.isclose(res, expected_output, **tol)


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestVar:
    """Tests that variances are properly calculated."""

    @pytest.mark.parametrize(
        "operation,input,expected_output",
        [
            (qml.PauliX, [0], 1),
            (qml.PauliY, [0], 1),
            (qml.PauliZ, [0], 0),
            (qml.PauliZ, [1], 0),
            (qml.Hadamard, [0], 1 / 2),
            (qml.Hadamard, [1], 1 / 2),
        ],
    )
    def test_var_single_wire_no_parameters(
        self, qsimh_device_1_wire, tol, operation, input, expected_output
    ):
        """Tests that variances are properly calculated for single-wire observables without parameters."""

        op = operation(0, do_queue=False)

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply(
            [qml.BasisState(np.array(input), wires=[0, 1])],
            rotations=op.diagonalizing_gates(),
        )

        res = qsimh_device_1_wire.var(op)

        assert np.isclose(res, expected_output, **tol)

    @pytest.mark.parametrize(
        "operation,input,expected_output,par",
        [
            (qml.Identity, [0], 0, []),
            (qml.Identity, [1], 0, []),
            (qml.Hermitian, [0], 1, [[[1, 1j], [-1j, 1]]]),
            (qml.Hermitian, [1], 1, [[[1, 1j], [-1j, 1]]]),
        ],
    )
    def test_var_single_wire_with_parameters(
        self, qsimh_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that expectation values are properly calculated for single-wire observables with parameters."""

        if par:
            op = operation(np.array(*par), 0, do_queue=False)
        else:
            op = operation(0, do_queue=False)

        qsimh_device_1_wire.reset()
        qsimh_device_1_wire.apply(
            [qml.BasisState(np.array(input), wires=[0, 1])],
            rotations=op.diagonalizing_gates(),
        )

        if par:
            res = qsimh_device_1_wire.var(op)
        else:
            res = qsimh_device_1_wire.var(op)

        assert np.isclose(res, expected_output, **tol)

    # FIXME: no Schmidt decomposition for gate kind 36.
    # qsimh simulation of the circuit errored out.
    #
    # @pytest.mark.parametrize(
    #     "operation,input,expected_output,par",
    #     [
    #         (
    #             qml.Hermitian,
    #             [1, 1],
    #             1,
    #             [[[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]],
    #         ),
    #     ],
    # )
    # def test_var_two_wires_with_parameters(
    #     self, qsimh_device_2_wires, tol, operation, input, expected_output, par
    # ):
    #     """Tests that variances are properly calculated for two-wire observables with parameters."""

    #     op = operation(np.array(*par), [0, 1], do_queue=False)

    #     qsimh_device_2_wires.reset()
    #     qsimh_device_2_wires.apply(
    #         [qml.BasisState(np.array(input), wires=[0, 1])],
    #         rotations=op.diagonalizing_gates(),
    #     )
    #     sdfsdf

    #     res = qsimh_device_2_wires.var(op)

    #     assert np.isclose(res, expected_output, **tol)


# FIXME: no Schmidt decomposition for gate kind 36.
# qsimh simulation of the circuit errored out.
#
# class TestVarEstimate:
#     """Test the estimation of variances."""

#     def test_var_estimate(self):
#         """Test that the variance is not analytically calculated"""

#         dev = qml.device("cirq.qsimh", wires=1, shots=3, analytic=False)

#         @qml.qnode(dev)
#         def circuit():
#             return qml.var(qml.PauliX(0))

#         var = circuit()

#         # With 3 samples we are guaranteed to see a difference between
#         # an estimated variance an an analytically calculated one
#         assert var != 1.0


@pytest.mark.parametrize("shots,analytic", [(100, True)])
class TestSample:
    """Test sampling."""

    # FIXME: Doesn't work because np.abs(QSimhDevice._state) ** 2 doesn't exactly sum up to 1;
    # it's a very close 0.999999910599974, which np.random.choice doesn't like.

    # def test_sample_dimensions(self, qsimh_device_2_wires):
    #     """Tests if the samples returned by the sample function have
    #     the correct dimensions
    #     """
    #     qsimh_device_2_wires.reset()
    #     qsimh_device_2_wires.apply([qml.RX(1.5708, wires=[0]), qml.RX(1.5708, wires=[1])])

    #     qsimh_device_2_wires.shots = 10
    #     qsimh_device_2_wires._samples = qsimh_device_2_wires.generate_samples()
    #     s1 = qsimh_device_2_wires.sample(qml.PauliZ(0))
    #     assert np.array_equal(s1.shape, (10,))

    #     qsimh_device_2_wires.shots = 12
    #     qsimh_device_2_wires._samples = qsimh_device_2_wires.generate_samples()
    #     s2 = qsimh_device_2_wires.sample(qml.PauliZ(1))
    #     assert np.array_equal(s2.shape, (12,))

    #     qsimh_device_2_wires.shots = 17
    #     qsimh_device_2_wires._samples = qsimh_device_2_wires.generate_samples()
    #     s3 = qsimh_device_2_wires.sample(qml.Hermitian(np.diag([1, 1, 1, -1]), wires=[0, 1]))
    #     assert np.array_equal(s3.shape, (17,))

    # def test_sample_values(self, qsimh_device_2_wires, tol):
    #     """Tests if the samples returned by sample have
    #     the correct values
    #     """

    #     qsimh_device_2_wires.reset()

    #     qsimh_device_2_wires.apply([qml.RX(1.5708, wires=[0])])
    #     qsimh_device_2_wires._samples = qsimh_device_2_wires.generate_samples()

    #     s1 = qsimh_device_2_wires.sample(qml.PauliZ(0))

    #     # s1 should only contain 1 and -1, which is guaranteed if
    #     # they square to 1
    #     assert np.allclose(s1 ** 2, 1, **tol)


class TestState:
    """Test the state property."""

    @pytest.mark.parametrize("shots,analytic", [(100, True)])
    @pytest.mark.parametrize(
        "ops,expected_state",
        [
            ([qml.PauliX(0), qml.PauliX(1)], [0, 0, 0, 1]),
            ([qml.PauliX(0), qml.PauliY(1)], [0, 0, 0, 1j]),
            ([qml.PauliZ(0), qml.PauliZ(1)], [1, 0, 0, 0]),
        ],
    )
    def test_state_pauli_operations(self, qsimh_device_2_wires, ops, expected_state, tol):
        """Test that the state reflects Pauli operations correctly."""
        qsimh_device_2_wires.reset()
        qsimh_device_2_wires.apply(ops)

        assert np.allclose(qsimh_device_2_wires.state, expected_state, **tol)

    @pytest.mark.parametrize("shots,analytic", [(100, True)])
    @pytest.mark.parametrize(
        "ops,diag_ops,expected_state",
        [
            ([qml.PauliX(0), qml.PauliX(1)], [], [0, 0, 0, 1]),
            (
                [qml.PauliX(0), qml.PauliY(1)],
                [qml.Hadamard(0)],
                [0, 1j / np.sqrt(2), 0, -1j / np.sqrt(2)],
            ),
            (
                [qml.PauliZ(0), qml.PauliZ(1)],
                [qml.Hadamard(1)],
                [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
            ),
        ],
    )
    def test_state_pauli_operations_and_observables(
        self, qsimh_device_2_wires, ops, diag_ops, expected_state, tol
    ):
        """Test that the state reflects Pauli operations and observable rotations correctly."""
        qsimh_device_2_wires.reset()
        qsimh_device_2_wires.apply(ops, rotations=diag_ops)

        assert np.allclose(qsimh_device_2_wires.state, expected_state, **tol)

    @pytest.mark.parametrize("shots,analytic", [(100, False)])
    def test_state_non_analytic(self, qsimh_device_2_wires):
        """Test that the state is None if in non-analytic mode."""
        qsimh_device_2_wires.reset()
        qsimh_device_2_wires.apply([qml.PauliX(0), qml.PauliX(1)])

        assert qsimh_device_2_wires.state is None
