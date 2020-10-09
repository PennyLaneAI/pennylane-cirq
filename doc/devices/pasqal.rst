Pasqal Device
=============

You can instantiate a simulator for Pasqal's neutral-atom devices in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device("cirq.pasqal", wires=2, control_radius=1.5)


This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

The Pasqal device supports unique features of Pasqal's quantum computing hardware provided via Cirq, namely
the ``ThreeDQubit`` and the notion of a ``control_radius``.

.. code-block:: python

    from cirq.pasqal import ThreeDQubit
    qubits = [ThreeDQubit(x, y, z)
              for x in range(2)
              for y in range(2)
              for z in range(2)]
    dev = qml.device("cirq.pasqal", control_radius = 2., qubits=qubits, wires=len(qubits))

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

    circuit(0.3)

Note that if not specified, ``ThreeDGridQubits`` are automatically arranged in a linear
arrangement along the first coordinate axis, separated by a distance of ``control_radius / 2``.
That is, ``(0, 0, 0), (control_radius/2, 0, 0), (control_radius, 0, 0)``.

For more details about Pasqal devices, consult the `Cirq docs <https://cirq.readthedocs.io/en/stable/docs/pasqal/getting_started.html>`_.
