#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
     version='0.0.0',
     description='The 2d_3d_s_meshgenerator package',
     packages=['stanford_dataset'],
     package_dir={'': 'src'}
)

setup(**setup_args)