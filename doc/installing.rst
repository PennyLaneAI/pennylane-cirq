.. _installation:

Setup
#####


Dependencies
============

.. highlight:: bash

Plugin Name requires the following libraries be installed:

* `Python <http://python.org/>`_ >=3.5

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`_ >=0.4
* `Target Framework <https://targetframework.readthedocs.io/>`_ >=0.11


If you currently do not have Python 3 installed, we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`_, a distributed version of Python packaged for scientific computation.


Installation
============

Installation of Plugin Name, as well as all required Python packages mentioned above, can be installed via ``pip``:
::

   	$ python -m pip install plugin-name


Make sure you are using the Python 3 version of pip.

Alternatively, you can install Plugin Name from the source code by navigating to the top directory and running
::

	$ python setup.py install


Software tests
==============

To ensure that Plugin Name is working correctly after installation, the test suite can be run by navigating to the source code folder and running
::

	$ make test


Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

  $ make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
