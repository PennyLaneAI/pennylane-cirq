The Simulator device
====================

You can instantiate the device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('cirq.simulator', wires=2)

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.
A simple quantum function that returns the expectation value of a measurement and depends on three classical input
parameters would look like:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

Device options
~~~~~~~~~~~~~~

The Cirq devices accept additional arguments beyond the PennyLane default device arguments:

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
    >>> dev = qml.device("cirq.simulator", wires=4, qubits=qubits)

	The wire of each qubit corresponds to its index in the `qubit` list. In the above example,
    the wire 2 corresponds to `cirq.GridQubit(1, 0)`. If no qubits are given, the plugin will
    create an array of `LineQubit` instances.


Supported operations
~~~~~~~~~~~~~~~~~~~~

The ``cirq.simulator`` device supports all PennyLane
`operations and observables <https://pennylane.readthedocs.io/en/latest/code/ops/qubit.html>`_.

In the future, the device will also support all Cirq operations.