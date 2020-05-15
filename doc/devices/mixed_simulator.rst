The Mixed Simulator Device
==========================

You can instantiate the mixed-state simulator device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('cirq.mixedsimulator', wires=2)

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.
Unlike the ``cirq.simulator`` backend, this device also supports several of Cirq's custom non-unitary channels,
e.g., ``BitFlip`` or ``Depolarize``.

.. code-block:: python

    from pennylane_cirq import ops

    @qml.qnode(dev)
    def circuit(x, p, q):
        qml.RX(x, wires=[0])
        ops.BitFlip(p, wires=[0])
        ops.Depolarize(q, wires=[1])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

    circuit(0.2, 0.1, 0.3)

This device stores the internal state of the quantum simulation as a density matrix.
This has additional memory overhead compared to pure-state simulation, but allows for
additional channels to be performed. The density matrix can be accessed after a circuit
execution using ``dev.state``.

Device options
~~~~~~~~~~~~~~

Cirq has different ways of defining qubits, e.g., ``LineQubit`` or ``GridQubit``. The Cirq device therefore accepts
an additional argument ``qubits=None`` that you can use to define your own
qubits and give them to the device as a list.

.. code-block:: python

    import cirq

    qubits = [
      cirq.GridQubit(0, 0),
      cirq.GridQubit(0, 1),
      cirq.GridQubit(1, 0),
      cirq.GridQubit(1, 1),
    ]

    dev = qml.device("cirq.mixedsimulator", wires=4, qubits=qubits)

The wire of each qubit corresponds to its index in the ``qubit`` list. In the above example,
the wire 2 corresponds to ``cirq.GridQubit(1, 0)``.

If no qubits are given, the plugin will create an array of ``LineQubit`` instances.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The ``cirq.mixedsimulator`` device supports all PennyLane
`operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html>`_.

It also supports the following non-unitary channels from Cirq (found in ``pennylane_cirq.ops``):
``BitFlip``, ``PhaseFlip``, ``PhaseDamp``, ``AmplitudeDamp``, and ``Depolarize``.
