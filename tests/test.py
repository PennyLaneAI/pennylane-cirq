import pennylane as qml
import numpy as np

dev_q = qml.device("cirq.qsim", wires=2)
dev = qml.device("cirq.simulator", wires=2)

@qml.qnode(dev)
def circuit():
    """Reference QNode"""
    return qml.expval(qml.Identity(0))


print("1", circuit())

@qml.qnode(dev_q)
def circuit2():
    """Reference QNode"""
    return qml.expval(qml.Identity(0))

print("2", circuit2())