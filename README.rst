PennyLane-Cirq Plugin
######################

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

.. image:: https://github.com/XanaduAI/pennylane-cirq/blob/master/doc/_static/puzzle_cirq.png?raw=true
    :align: center
    :width: 200px
    :target: javascript:void(0);

|

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Cirq plugin integrates the Cirq quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`__ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Cirq <https://github.com/quantumlib/Cirq>`__ is a software library for quantum computing.

.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `<https://pennylane-cirq.readthedocs.io/en/latest/>`__.

Features
========

* Provides access to Cirq's simulator backend via the ``cirq.simulator`` device

* Support for all PennyLane core functionality

.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.6 or above, as well as PennyLane
and Cirq. Installation of this plugin, as well as all dependencies, can be done using ``pip``:

.. code-block:: bash

    $ pip install pennylane-cirq

Alternatively, you can install PennyLane-Cirq from the `source code <https://github.com/XanaduAI/pennylane-cirq>`__
by navigating to the top directory and running:

.. code-block:: bash

	$ python setup.py install

Dependencies
~~~~~~~~~~~~

PennyLane-Cirq requires the following libraries be installed:

* `Python <http://python.org/>`__ >= 3.6

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >= 0.9
* `Cirq <https://cirq.readthedocs.io/>`__ >= 0.7


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed version of Python packaged
for scientific computation.


Tests
~~~~~

To test that the PennyLane-Cirq plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

Documentation
~~~~~~~~~~~~~

To build the HTML documentation, go to the top-level directory and run:

.. code-block:: bash

  $ make docs


The documentation can then be found in the ``doc/_build/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`__ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Authors
=======

PennyLane-Cirq is the work of `many contributors <https://github.com/XanaduAI/pennylane-cirq/graphs/contributors>`__.

If you are doing research using PennyLane and PennyLane-Cirq, please cite `our paper <https://arxiv.org/abs/1811.04968>`__:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane-cirq
- **Issue Tracker:** https://github.com/XanaduAI/pennylane-cirq/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane-Cirq plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__.

.. license-end-inclusion-marker-do-not-remove
