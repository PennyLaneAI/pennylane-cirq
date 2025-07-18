# Release 0.43.0-dev

### New features since last release

### Improvements 🛠

### Breaking changes 💔

### Deprecations 👋

### Documentation 📝

### Bug fixes 🐛

### Contributors ✍️

This release contains contributions from (in alphabetical order):

---
# Release 0.42.0
 
 ### New features since last release
 
 ### Improvements 🛠
 
 ### Breaking changes 💔

* Upgrade minimum supported version of PennyLane to 0.42.0.
  [(#221)](https://github.com/PennyLaneAI/pennylane-cirq/pull/221)
 
 ### Deprecations 👋

 ### Internal changes ⚙️

 * Bumped `.readthedocs.yml` up to Ubuntu-24.04.
  [(#217)](https://github.com/PennyLaneAI/pennylane-cirq/pull/217)

 * Use new `pennylane.exceptions` module for custom exceptions.
  [(#214)](https://github.com/PennyLaneAI/pennylane-cirq/pull/214)
 
 ### Documentation 📝
 
 ### Bug fixes 🐛
 
 ### Contributors ✍️
 
 This release contains contributions from (in alphabetical order):
 
Runor Agbaire,
Andrija Paurevic

 ---
# Release 0.41.0

### Internal changes ⚙️

* Pinning `setuptools` in the CI to update how the plugin is installed.
  [(#208)](https://github.com/PennyLaneAI/pennylane-cirq/pull/208)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Pietropaolo Frisoni

---
# Release 0.40.0

### Breaking changes 💔

* The ``qml.QubitStateVector`` template has been removed. Instead, use :class:`~pennylane.StatePrep`.
  [(#203)](https://github.com/PennyLaneAI/pennylane-cirq/pull/203)

* Support for the `pennylane.operation.Tensor` observable is removed. This observable was deprecated
  in PennyLane 0.39, and is removed in PennyLane 0.40.
  [(#204)](https://github.com/PennyLaneAI/pennylane-cirq/pull/204)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Andrija Paurevic

---
# Release 0.39.0

### Bug fixes 🐛

* Remove deprecated `qml.operation.Tensor` from codebase in favour of `qml.prod`.
  [(#197)](https://github.com/PennyLaneAI/pennylane-cirq/pull/197)

* Fix deprecated import path for `QubitDevice`.
  [(#194)](https://github.com/PennyLaneAI/pennylane-cirq/pull/194)
  [(#195)](https://github.com/PennyLaneAI/pennylane-cirq/pull/195)
  
### Breaking changes 💔

* Removed support for Python 3.9
  [(#200)](https://github.com/PennyLaneAI/pennylane-cirq/pull/200)

* Upgrade minimum supported version of PennyLane to 0.38.0.
  [(#201)](https://github.com/PennyLaneAI/pennylane-cirq/pull/201)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Astral Cai,
Mudit Pandey,
Alex Preciado

---
# Release 0.36.0

### New features since last release

* Added support for `expval` of `Prod` observables.
  [(#183)](https://github.com/PennyLaneAI/pennylane-cirq/pull/183)

### Bug fixes 🐛

* Fixes a bug where an error is raised from applying `qml.Identity` on multiple wires.
  [(#186)](https://github.com/PennyLaneAI/pennylane-cirq/pull/186)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Astral Cai

---
# Release 0.34.0

### Tests

* The sampling tests no longer set the `_obs_queue` proprerty, as setting it has no effect on the behaviour of the plugin.
  [(#159)](https://github.com/PennyLaneAI/pennylane-cirq/pull/159)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Christina Lee

---
# Release 0.33.0

### Bug fixes 🐛

* Fixes the pasqal device when more than one circuit is executed and adds support
  for specifying `wires` an iterable of wire labels.
  [(#151)](https://github.com/PennyLaneAI/pennylane-cirq/pull/151)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Christina Lee

---
# Release 0.32.0

### Breaking changes 💔

* Python 3.8 support is dropped and Python 3.11 support is added.
  [(#146)](https://github.com/PennyLaneAI/pennylane-cirq/pull/146)

### Bug fixes 🐛

* The plugin is updated to take `qml.StatePrep` operators, the new name for `qml.QubitStateVector`.
  [(#146)](https://github.com/PennyLaneAI/pennylane-cirq/pull/146)

### Contributors ✍️

This release contains contributions from (in alphabetical order):

Christina Lee

---
# Release 0.31.0

### Improvements

* Failing tests corrected to reflect the new shots validation in PennyLane
  [(#138)](https://github.com/PennyLaneAI/pennylane-cirq/pull/138)

### Contributors

This release contains contributions from (in alphabetical order):

Matthew Silverman

---
# Release 0.29.0

### New features since last release

* Support for adjoint operators has been added.
  [(#130)](https://github.com/PennyLaneAI/pennylane-cirq/pull/130)

### Breaking changes

* Support for inverse operators has been removed.
  Note that the `inv()` method and `inverse` property are removed from PennyLane operators as of PennyLane 0.29.
  [(#130)](https://github.com/PennyLaneAI/pennylane-cirq/pull/130)

* Bumps the required PennyLane version to v0.29.0.
  [(#137)](https://github.com/PennyLaneAI/pennylane-cirq/pull/137)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Matthew Silverman

---
# Release 0.28.0

### New features since last release

* Adds support for Python 3.10.
  [(#123)](https://github.com/PennyLaneAI/pennylane-cirq/pull/123)

### Breaking changes

* Removes support for Python 3.7.
  [(#123)](https://github.com/PennyLaneAI/pennylane-cirq/pull/123)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee

---
# Release 0.27.0

### New features since last release

* Support `ISWAP`, `SISWAP`, and their adjoints.
  [(#114)](https://github.com/PennyLaneAI/pennylane-cirq/pull/114)

* Support a variety of operations raised to a power.
  [(#115)](https://github.com/PennyLaneAI/pennylane-cirq/pull/115)

### Breaking changes

* Removed the unnecessary `CPhase` and `ISWAP` operations from the
  plugin's custom support because PennyLane supports them.
  [(#115)](https://github.com/PennyLaneAI/pennylane-cirq/pull/115)

### Improvements

* Pass all the qubits as `qubit_order` parameter to force 
  the simulator not to ignore idle qubits. 
  [(#111)](https://github.com/PennyLaneAI/pennylane-cirq/pull/111) 

### Contributors

This release contains contributions from (in alphabetical order):

Oumarou Oumarou, Matthew Silverman

---
# Release 0.24.0

### Bug fixes

* Defines the missing `returns_state` entry of the
  `capabilities` dictionary for devices.
  [(#107)](https://github.com/PennyLaneAI/pennylane-cirq/pull/107)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.22.0

### Improvements

* Changed to using `cirq_pasqal` instead of `cirq.pasqal` as per a deprecation
  cycle in Cirq.
  [(#90)](https://github.com/PennyLaneAI/pennylane-cirq/pull/90)

* Changed the requirements of PennyLane-Cirq to only contain `cirq-core` and
  `cirq-pasqal`.
  [(#94)](https://github.com/PennyLaneAI/pennylane-cirq/pull/94)

### Contributors

This release contains contributions from (in alphabetical order):

Jay Soni, Antal Száva

---

# Release 0.19.0

### Improvements

* Expectation values are now computed using the `simulate_expectation_values`
  function from Cirq for `cirq.simulator` and `cirq.mixedsimulator`.
  [(#81)](https://github.com/PennyLaneAI/pennylane-cirq/pull/81)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard

---

# Release 0.17.1

### Improvements

We support `Cirq` version `0.12.0`.
[(#77)](https://github.com/PennyLaneAI/pennylane-cirq/pull/77)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard

---

# Release 0.17.0

### Breaking changes

We do not support the new `Cirq` version `0.12.0` because of 
compatibility problems with `qsim`.
[(#73)](https://github.com/PennyLaneAI/pennylane-cirq/pull/73)

### Contributors

This release contains contributions from (in alphabetical order):

Romain Moyard

---

# Release 0.16.0

## New features

* Added support for the new `qml.Projector` observable in
  PennyLane v0.16 to the Cirq devices.
  [(#62)](https://github.com/PennyLaneAI/pennylane-cirq/pull/62)

## Breaking changes

* Deprecated Python 3.6.
  [(#65)](https://github.com/PennyLaneAI/pennylane-cirq/pull/65)

## Bug fixes

* Fix data type bug for mixed simulator when using `QubitStateVector`.
  [(#63)](https://github.com/PennyLaneAI/pennylane-cirq/pull/63)

* Fixed issue when using a subset of wires with `BasisState`.
  [(#61)](https://github.com/PennyLaneAI/pennylane-cirq/pull/61)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Romain Moyard, Vincent Wong

---

# Release 0.15.0

## Breaking changes

* For compatibility with PennyLane v0.15, the `analytic` keyword argument
  has been removed from all devices. Statistics can still be computed analytically
  by setting `shots=None`.
  [(#57)](https://github.com/PennyLaneAI/pennylane-cirq/pull/57)
  [(#58)](https://github.com/PennyLaneAI/pennylane-cirq/pull/58)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Chase Roberts

---

# Release 0.14.0

### New features

* Added support for custom simulator objects for use with the Floq service.
  [(#51)](https://github.com/PennyLaneAI/pennylane-cirq/pull/51)

### Contributors

This release contains contributions from (in alphabetical order):

Chase Roberts

---

# Release 0.13.0

### Improvements

* Added support for iSWAP and CPhase gate operations.
  [(#45)](https://github.com/PennyLaneAI/pennylane-cirq/pull/45)

### Bug fixes

* Removed import of `qsimcirq` from `__init__.py`, allowing other devices in
  this plugin to work without `qsimcirq` being installed.
  [(#46)](https://github.com/PennyLaneAI/pennylane-cirq/pull/46)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson

---

# Release 0.12.1

### New features since last release

* PennyLane integration with the [qsim circuit simulator
  package](https://github.com/quantumlib/qsim) is now available.
  [(#36)](https://github.com/PennyLaneAI/pennylane-cirq/pull/36).

  The new devices include:

  * `cirq.qsim`, a Schrödinger full state-vector simulator

  * `cirq.qsimh`, a hybrid Schrödinger-Feynman simulator. This simulator cuts the qubit lattice into
    two parts; each part is individually simulated using qsim, with Feynman-style path summation used
    to return the final result. Compared to full state-vector simulation, qsimh reduces memory
    requirements, at the expense of an increased runtime.

  After installing the `qsimcirq` package, the qsim and qsimh devices
  can be invoked via the names `"cirq.qsim"` and `"cirq.qsimh"` respectively, e.g.,

  ```python
  dev = qml.device("cirq.qsimh", qsimh_options=qsimh_options, wires=3)
  ```

  These devices can then be used for the evaluation of QNodes within PennyLane. For more details,
  see the [PennyLane qsim
  documentation](https://pennylane-cirq.readthedocs.io/en/latest/devices/qsim.html)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Nathan Killoran, Josh Izaac

---

# Release 0.12.0

### New features since last release

* Devices from Cirq's Pasqal submodule are now available for use.
  [(#40)](https://github.com/PennyLaneAI/pennylane-cirq/pull/40).

  A simulator device compatible with Pasqal's neutral-atom model can be invoked via the name
  `"cirq.pasqal"`, e.g.,

  ```python
  dev = qml.device("cirq.pasqal", control_radius=1.0, wires=3)
  ```

### Contributors

This release contains contributions from (in alphabetical order):

Nathan Killoran, Josh Izaac

---

# Release 0.11.0

### New features

* Made plugin device compatible with new PennyLane wire management.
  [(#37)](https://github.com/PennyLaneAI/pennylane-cirq/pull/37)
  [(#42)](https://github.com/PennyLaneAI/pennylane-cirq/pull/42)

  One can now specify any string or number as a custom wire label,
  and use these labels to address subsystems on the device:

  ```python
  dev = qml.device('cirq.simulator', wires=['q1', 'ancilla', 0, 1])

  def circuit():
    qml.Hadamard(wires='q1')
    qml.CNOT(wires=[1, 'ancilla'])
    ...
  ```

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Nathan Killoran, Maria Schuld

---

# Release 0.9.1

### Improvements

### Contributors

This release contains contributions from (in alphabetical order):

---

# Release 0.9.0

### New features since last release

* Added a new mixedsimulator class to Cirq, which uses Cirq's
  DensityMatrixSimulator as a backend.
  [#27](https://github.com/XanaduAI/pennylane-cirq/pull/27)

### Documentation

* Redesigned the documentation to be consistent with other plugins.
  [#25](https://github.com/XanaduAI/pennylane-cirq/pull/25)

### Bug fixes

* Renamed probability to ``analytic_probability`` to support new
  changes in PennyLane.
  [#24](https://github.com/XanaduAI/pennylane-cirq/pull/24)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Nathan Killoran, Maria Schuld, Antal Száva

---

# Release 0.8.0

### Improvements

* Ported the `CirqDevice` class to use the new `QubitDevice` base class,
  enabling the use of tensor observables.
  [#19](https://github.com/XanaduAI/pennylane-cirq/pull/19)

* Added support for inverse operations by defining the `.inv()` method
  of the `CirqOperation` class which uses the `cirq.inverse` function.
  [#15](https://github.com/XanaduAI/pennylane-cirq/pull/15)

### Bug fixes

* Replaced depreceated Cirq commands.
  [#19](https://github.com/XanaduAI/pennylane-cirq/pull/19)

* Fix a minor bug introduced into the test suite by the release of Cirq 0.7.0.
  [#18](https://github.com/XanaduAI/pennylane-cirq/pull/18)

* Fix bugs introduced into the test suite by the release of Cirq 0.6.0.
  [#13](https://github.com/XanaduAI/pennylane-cirq/pull/13)

### Contributors

This release contains contributions from (in alphabetical order):

Johannes Jakob Meyer, Antal Száva

---

# Release 0.1.0

Initial public release.

### Contributors
This release contains contributions from:

Johannes Jakob Meyer
