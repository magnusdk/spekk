#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="spekk",
    version="1.0.9",
    description="Named dimensions of nested arrays, independent of array implementation (Numpy, JAX, etc)",
    author="Magnus Dalen Kvalevåg",
    author_email="magnus.kvalevag@ntnu.no",
    packages=find_packages(),
    install_requires=["numpy"],
)
