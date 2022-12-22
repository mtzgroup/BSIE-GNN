#!/usr/bin/env python

"""The setup script."""

from setuptools import setup
import glob
import os

setup(
    author="Soren Holm",
    author_email="sorenh@stanford.edu",
    python_requires=">=3.5",
    description="GNN for predicting the Basis Set Incompleteness Error of molecules",
    name="BSIE-GNN",
    packages=["NN", "Models"],
    scripts=glob.glob(os.path.join("Scripts", "*.py")),
    install_requires=['matplotlib'],
)
