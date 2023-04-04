# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adarsh Kalikadien                     #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #

from setuptools import setup, find_packages

setup(
    name='obelix-ml-pipeline',
    version='0.1.0',
    packages=find_packages(include=['obelix_ml_pipeline', 'obelix_ml_pipeline.*']),
    url='github.com/epics-group/obelix-ml-pipeline',
    license='GPLv3',
    author='Adarsh Kalikadien',
    author_email='a.v.kalikadien@tudelft.nl',
    description='',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_data={'obelix_ml_pipeline': ['data/']},
    long_description=open('README.md').read(),
)