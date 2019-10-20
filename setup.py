# Copyright 2019 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3

import sys
import os
from setuptools import setup

with open("pennylane_cirq/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

# Put pip installation requirements here.
# Requirements should be as minimal as possible.
# Avoid pinning, and use minimum version numbers
# only where required.
requirements = ["pennylane", "cirq"]

info = {
    "name": "PennyLane-Cirq",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "http://xanadu.ai",
    "license": "Apache License 2.0",
    "packages": [
        "pennylane_cirq"
    ],
    "entry_points": {
        "pennylane.plugins": [
            "cirq.simulator = pennylane_cirq:SimulatorDevice"
        ]
    },
    # Place a one line description here. This will be shown by pip
    "description": "PennyLane plugin for Cirq",
    "long_description": open("README.rst").read(),
    # The name of the folder containing the plugin
    "provides": ["pennylane_cirq"],
    "install_requires": requirements,
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    # Make sure to specify here the versions of Python supported
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
