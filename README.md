# Library to the Toolbox for Interpretability Comparison

[![Build Status](https://travis-ci.org/tommartensen/tic.svg?branch=master)](https://travis-ci.org/tommartensen/tic)
![PyPI](https://img.shields.io/pypi/v/tic)

* Free software: MIT license
* Documentation: https://tic.readthedocs.io.
* Examples: [examples/](examples/)

### Features

* Interface to create interpretations as:
  * direct extraction of feature importances, e.g. from coefficients
  * local and global explanations from [Local Interpretable Model Explanations (LIME)](https://github.com/marcotcr/lime)
  * local and global explanations from [SHAP](https://github.com/slundberg/shap)
  * global feature importances from surrogate models
