PennyLane Plugin Template
#########################

.. image:: https://img.shields.io/travis/com/XanaduAI/plugin-name/master.svg
    :alt: Travis
    :target: https://travis-ci.com/XanaduAI/plugin-name

.. image:: https://img.shields.io/codecov/c/github/xanaduai/plugin-name/master.svg
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/plugin-name

.. image:: https://img.shields.io/codacy/grade/33d12f7d2d0644968087e33966ed904e.svg
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/plugin-name

.. image:: https://img.shields.io/readthedocs/plugin-name.svg
    :alt: Read the Docs
    :target: https://plugin-name.readthedocs.io

.. image:: https://img.shields.io/pypi/v/plugin-name.svg
    :alt: PyPI
    :target: https://pypi.org/project/plugin-name


This template repository provides the boilerplate and file structure required to easily create your
own PennyLane plugin.

See the `PennyLane Developer API documentation <https://pennylane.readthedocs.io/en/latest/API/overview.html>`_
for more details on developing a PennyLane plugin.

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


Installation
============

Plugin Name requires both PennyLane and Target framework. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install plugin-name


Getting started
===============

Once Plugin Name is installed, the provided Target Framework devices can be accessed straight
away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device('pluginname.devicename', wires=2, additional_options=10)

These devices can then be used just like other devices for the definition and evaluation of
QNodes within PennyLane. For more details, see the
`plugin usage guide <https://plugin-name.readthedocs.io/en/latest/usage.html>`_ and refer
to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the Plugin Name repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to PennyLane-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane and Target Framework.


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


Support
=======

- **Source Code:** https://github.com/XanaduAI/plugin-name
- **Issue Tracker:** https://github.com/XanaduAI/plugin-namesf/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.


License
=======

Plugin Name is **free** and **open source**, released under the Apache License, Version 2.0.
