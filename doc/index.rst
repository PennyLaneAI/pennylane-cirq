PennyLane Target Framework Plugin
#################################

:Release: |release|
:Date: |today|


This PennyLane plugin allows the Target Framework simulators/hardware to be used as PennyLane devices.


`Target framework <https://targetframework.readthedocs.io>`_ is a full-stack Python library
for doing things.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization
and automatic differentiation of hybrid quantum-classical computations.



Features
========

* List the features provided by the plugin here. This can include:

* The devices made available to PennyLane, as well as any special features of the devices

* The core PennyLane operations and observables supported

* Any additional operations and observables provided by the plugin


To get started with the PennyLane Strawberry Fields plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.


Authors
=======

John Smith.

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

   Photon redirection <https://pennylane.readthedocs.io/en/latest/tutorials/plugins_hybrid.html>
   Notebook downloads <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/ops
   code/framework_device
   code/device1
   code/device2
