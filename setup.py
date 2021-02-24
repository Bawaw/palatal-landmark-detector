#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='dental-net',
    version='0.0.1',
    description='sparse landmark prediction on dental casts',
    author='Balder Croquet',
    author_email='balder.croquet@kuleuven.be',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Bawaw/3d_auto_landmarking',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
