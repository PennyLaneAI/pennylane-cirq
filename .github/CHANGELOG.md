# Release 0.9.0-dev

### New features since last release

### Breaking changes

### Improvements

### Documentation

* Redesigned the documentation to be consistent with other plugins
  [#25](https://github.com/XanaduAI/pennylane-cirq/pull/25)

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

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
