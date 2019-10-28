.. _usage:

Plugin usage
############

PennyLane-Cirq provides one Cirq device for PennyLane:

* :class:`pennylane_cirq.SimulatorDevice <~SimulatorDevice>`: provides a PennyLane device for the Cirq simulator backend


Using the devices
=================

Once Cirq and the PennyLane-Cirq plugin are installed, the device can be accessed straight away in PennyLane.

You can instantiate the device in PennyLane as follows:

>>> import pennylane as qml
>>> from pennylane import numpy as np
>>>      
>>> dev = qml.device('cirq.simulator', wires=2, shots=100, analytic=False)

The device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.


Device options
==============

The Cirq devices accept additional arguments beyond the PennyLane default device arguments.

``qubits=None``
    Cirq has different ways of defining qubits, e.g. `LineQubit` or `GridQubit`. You can define your own
    qubits and give them to the device as a list. 
    
    >>> import cirq
    >>>     
    >>> qubits = [
    >>>     cirq.GridQubit(0, 0),
    >>>     cirq.GridQubit(0, 1),
    >>>     cirq.GridQubit(1, 0),
    >>>     cirq.GridQubit(1, 1),
    >>> ]
    >>>     
    >>> dev = qml.device("cirq.simulator", wires=4, shots=100, qubits=qubits)

	The wire of each qubit corresponds to its index in the `qubit` list. In the above example, 
    the wire 2 corresponds to `cirq.GridQubit(1, 0)`. If no qubits are given, the plugin will
    create an array of `LineQubit` instances.


Supported operations
====================

All devices support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_.

In the future, the devices will also support all Cirq operations.