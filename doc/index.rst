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

To see the PennyLane-Cirq plugin in action, you can use any of the qubit based `demos
from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with the ``'cirq.simulator'`` device:

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
