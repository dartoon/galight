=============================
galight - Galaxy shapes of Light
=============================

.. image:: https://badge.fury.io/py/galight.png
    :target: http://badge.fury.io/py/galight

.. image:: https://travis-ci.org/dartoon/galight.png?branch=master
    :target: https://travis-ci.org/dartoon/galight

A python package that analyze and model the imaging data of galaxies, QSOs and duals.


Installation
------------

.. code-block:: bash

    $ pip install galight --user

Alternatively, the package can be installed through github channel:
https://github.com/dartoon/galight


Requirements
------------
Running ``galight`` requires the following packages to be installed.

 * ``lenstronomy`` `https://github.com/sibirrer/lenstronomy <https://github.com/sibirrer/lenstronomy>`_
 * ``astropy``  `https://github.com/astropy/astropy <https://github.com/astropy/astropy>`_
 * ``photutils`` `https://github.com/astropy/photutils <https://github.com/astropy/photutils>`_
 * ``regions`` `https://github.com/astropy/regions <https://github.com/astropy/regions>`_
and related ones to be installed... 

Example notebooks
-----------------
We have created `notebooks <https://github.com/dartoon/galight_notebooks>`_ to demonstrate how to use ``galight``. These notebooks demonstrate how to model QSOs and galaxies using 2D Sersic profile and scaled point source, based on ``lenstronomy`` `lenstronomy <https://github.com/sibirrer/lenstronomy>`_.

Examples including:

* `Modeling a HSC imaged QSO <https://github.com/dartoon/galight_notebooks/blob/master/galight_HSC_QSO.ipynb>`_
* `Modeling a HSC imaged dual QSO <https://github.com/dartoon/galight_notebooks/blob/master/galight_HSC_dualAGN.ipynb>`_
* `Modeling a HST imaged QSO <https://github.com/dartoon/galight_notebooks/blob/master/galight_HST_QSO.ipynb>`_

Features
--------
The notebook demonstrates the follwing feature/functions:

* Search PSF stars through entire field of view, automatically.
* Cutout the target object and prepare the materials for the modelling.
* Estimate the background noise level from empty regions.
* Estimate the global background light and remove.
* Detecting objects in the cutout stamp and quickly create Sersic keywords (in ``lenstronomy`` type) to model them.
