#!/usr/bin/env python
# -*- coding: utf-8 -*-

'The setup script.'

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author='Tom Martensen',
    author_email='mail@tommartensen.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description=(
        'TIC is a library that acts as a '
        'Toolbox for Interpretability Comparison.'
    ),
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tic',
    name='tic',
    packages=find_packages(include=['tic']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tommartensen/tic',
    version='0.1.0',
    zip_safe=False,
)
