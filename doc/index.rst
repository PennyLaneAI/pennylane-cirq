PennyLane-Cirq Plugin
#####################

:Release: |release|

.. include:: ../README.rst
  :start-after:    header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once Pennylane-Cirq is installed, the provided Cirq devices can be accessed straight
away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~

Currently, PennyLane-Cirq provides four Cirq devices for PennyLane:

.. title-card::
    :name: 'cirq.simulator'
    :description: Cirq's simulator backend.
    :link: devices/simulator.html

.. title-card::
    :name: 'cirq.mixedsimulator'
    :description: Cirq's density matrix simulator backend.
    :link: devices/mixed_simulator.html

.. title-card::
    :name: 'cirq.qsim' and 'cirq.qsimh'
    :description: The qsim and qsimh simulator backends.
    :link: devices/qsim.html

.. title-card::
    :name: 'cirq.pasqal'
    :description: Simulator for Pasqal's neutral atom devices.
    :link: devices/pasqal.html

.. raw:: html

        <div style='clear:both'></div>
        </br>

Tutorials
~~~~~~~~~

Check out these demos to see the PennyLane-Cirq plugin in action:

.. raw:: html

    <div class="row">

.. title-card::
    :name: Optimizing noisy circuits with Cirq
    :description: <img src="https://pennylane.ai/_static/demonstration_assets/noisy_circuit_optimization/noisy_circuit_optimization_thumbnail.png" width="100%" />
    :link: https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization.html

.. title-card::
    :name: Quantum Generative Adversarial Networks with Cirq + TensorFlow
    :description: <img src="https://pennylane.ai/_static/demonstration_assets/QGAN/qgan3.png" width="100%" />
    :link:  https://pennylane.ai/qml/demos/tutorial_QGAN.html

.. title-card::
    :name: Quantum computation with neutral atoms
    :description: <img src="https://pennylane.ai/_static/demonstration_assets/pasqal/pasqal_thumbnail.png" width="100%" />
    :link: https://pennylane.ai/qml/demos/tutorial_pasqal.html

.. title-card::
    :name: Variationally optimizing measurement protocols
    :description: <img src="https://pennylane.ai/_static/demonstration_assets/quantum_metrology/illustration.png" width="100%" />
    :link: https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html

.. title-card::
    :name: Beyond classical computing with qsim
    :description: <img src="https://pennylane.ai/_images/sycamore.png" width="100%" />
    :link:  https://pennylane.ai/qml/demos/qsim_beyond_classical.html

.. raw:: html

    </div></div><div style='clear:both'> <br/>

You can also try it out using any of the qubit based `demos from the PennyLane documentation
<https://pennylane.ai/qml/demonstrations.html>`_, for example the tutorial on
`qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_.
Simply replace ``'default.qubit'`` with the ``'cirq.simulator'`` device


.. code-block:: python

    dev = qml.device('cirq.simulator', wires=XXX)


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices/simulator
   devices/mixed_simulator
   devices/qsim
   devices/pasqal

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
