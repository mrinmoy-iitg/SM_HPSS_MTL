#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:49:36 2021

@author: Mrinmoy Bhattacharjee, Ph.D. Scholar, IIT Guwahati
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('tools.pyx', build_dir='./'), include_dirs=[numpy.get_include()])    


# Running the script: python setup.py build_ext --inplace
