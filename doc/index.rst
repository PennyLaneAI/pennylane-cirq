PennyLane Cirq Plugin
#################################

:Release: |release|
:Date: |today|


This PennyLane plugin allows the Cirq simulators/hardware to be used as PennyLane devices.


`Cirq <https://cirq.readthedocs.io>`_ is a Python library for writing, manipulating,
and optimizing quantum circuits and running them against quantum computers and simulators.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization
and automatic differentiation of hybrid quantum-classical computations.



Features
========

* Access to Cirq's simulator backend via the `cirq.simulator` device

* Support for all PennyLane core functionality


To get started with the PennyLane Cirq plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.


Authors
=======

Johannes Jakob Meyer

If you are doing research using PennyLane, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.
    *Evaluating analytic gradients on quantum hardware.* 2018.
    `Phys. Rev. A 99, 032331 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.032331>`_


Contents
========

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installing
   usage


.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Tutorials (external links)

   Notebook downloads <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/cirq_device
   code/simulator_device
