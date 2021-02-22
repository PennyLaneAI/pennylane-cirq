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

.. devicegalleryitem::
    :name: 'cirq.simulator'
    :description: Cirq's simulator backend.
    :link: devices/simulator.html

.. devicegalleryitem::
    :name: 'cirq.mixedsimulator'
    :description: Cirq's density matrix simulator backend.
    :link: devices/mixed_simulator.html

.. devicegalleryitem::
    :name: 'cirq.qsim' and 'cirq.qsimh'
    :description: The qsim and qsimh simulator backends.
    :link: devices/qsim.html

.. devicegalleryitem::
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

.. demogalleryitem::
    :name: Optimizing noisy circuits with Cirq
    :figure: https://pennylane.ai/qml/_images/noisy_circuit_optimization_thumbnail.png
    :link: https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization.html
    :tooltip: Learn how noise can affect the optimization and training of quantum computations.

.. demogalleryitem::
    :name: Quantum Generative Adversarial Networks with Cirq + TensorFlow
    :figure: https://pennylane.ai/qml/_images/qgan3.png
    :link:  https://pennylane.ai/qml/demos/tutorial_QGAN.html
    :tooltip: Create a simple QGAN with Cirq and TensorFlow.

.. demogalleryitem::
    :name: Quantum computation with neutral atoms
    :figure: https://pennylane.ai/qml/_images/pasqal_thumbnail.png
    :link: https://pennylane.ai/qml/demos/tutorial_pasqal.html
    :tooltip: Making a quantum machine learning model using neutral atoms.

.. demogalleryitem::
    :name: Variationally optimizing measurement protocols
    :figure: https://pennylane.ai/qml/_images/illustration.png
    :link: https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html
    :tooltip: Optimizing measurement protocols with variational methods.

.. demogalleryitem::
    :name: Beyond classical computing with qsim
    :figure: https://pennylane.ai/qml/_images/sycamore.png
    :link:  https://pennylane.ai/qml/demos/qsim_beyond_classical.html
    :tooltip: Beyond classical computing with qsim.

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
