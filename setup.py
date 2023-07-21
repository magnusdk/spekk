#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="spekk",
    version="1.0.1",
    description="General speccing library",
    author="Magnus Dalen Kvalevåg",
    author_email="magnus.kvalevag@ntnu.no",
    packages=find_packages(),
    install_requires=["numpy"],
)
