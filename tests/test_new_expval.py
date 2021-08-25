import pytest

import numpy as np
import pennylane as qml
from contextlib import contextmanager

from conftest import U, U2, A, B


np.random.seed(42)

@pytest.mark.parametrize("shots", [None, 10])
class TestNewExpval:
    """Test tensor expectation values"""
    def test_new_expval_Z(self, device, shots, tol):
        """Test that PauliZ expectation value is correct"""

        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=[0])
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        @qml.qnode(dev)
        def circuit2():
            qml.PauliZ(wires=[0])
            return qml.expval(qml.PauliZ(wires=[1]))

        print("res1", circuit())
        print("res2", circuit2())

    def test_new_expval_X(self, device, shots, tol):
        """Test that PauliX expectation value is correct"""

        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(wires=[0])
            return [qml.expval(qml.PauliX(i)) for i in range(2)]

        @qml.qnode(dev)
        def circuit2():
            qml.PauliX(wires=[0])
            return qml.expval(qml.PauliY(wires=[1]))

        print("res1", circuit())
        print("res2", circuit2())