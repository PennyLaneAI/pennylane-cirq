PennyLane Cirq Plugin
#########################

.. image:: https://img.shields.io/travis/com/XanaduAI/pennylane-cirq/master.svg
    :alt: Travis
    :target: https://travis-ci.com/XanaduAI/pennylane-cirq

.. image:: https://img.shields.io/codecov/c/github/xanaduai/pennylane-cirq/master.svg
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/pennylane-cirq

.. image:: https://img.shields.io/codacy/grade/33d12f7d2d0644968087e33966ed904e.svg
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/pennylane-cirq

.. image:: https://img.shields.io/readthedocs/pennylane-cirq.svg
    :alt: Read the Docs
    :target: https://pennylane-cirq.readthedocs.io

.. image:: https://img.shields.io/pypi/v/pennylane-cirq.svg
    :alt: PyPI
    :target: https://pypi.org/project/pennylane-cirq


`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Cirq <https://github.com/quantumlib/Cirq>`_ is a Python library for writing, manipulating, and optimizing quantum circuits and running them against quantum computers and simulators.

This PennyLane plugin allows to use both the software and hardware backends of Cirq as devices for PennyLane.


Features
========

* Access to Cirq's simulator backend via the `cirq.simulator` device

* Support for all PennyLane core functionality


Installation
============

Plugin Name requires both PennyLane and Cirq. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-cirq


Getting started
===============

Once Pennylane Cirq is installed, the provided Cirq devices can be accessed straight
away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev = qml.device('cirq.simulator', wires=2, shots=100, analytic=True)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, see the
`plugin usage guide <https://pennylane-cirq.readthedocs.io/en/latest/usage.html>`_ and refer
to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the Plugin Name repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributors to PennyLane-Cirq will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and Cirq.


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


Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane-cirq
- **Issue Tracker:** https://github.com/XanaduAI/pennylane-cirq/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.


License
=======

Plugin Name is **free** and **open source**, released under the Apache License, Version 2.0.
