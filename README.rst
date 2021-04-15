=============================
decomprofile - Decompse the galaxy light profile
=============================

.. image:: https://badge.fury.io/py/decomprofile.png
    :target: http://badge.fury.io/py/decomprofile

.. image:: https://travis-ci.org/dartoon/decomprofile.png?branch=master
    :target: https://travis-ci.org/dartoon/decomprofile

A python package that analyze and model the imaging data of galaxies, QSOs and duals.


Installation
------------

.. code-block:: bash

    $ pip install decomprofile --user

Requirements
------------
Running ``decomprofile`` requires the following packages to be installed.

 * ``lenstronomy`` `https://github.com/sibirrer/lenstronomy <https://github.com/sibirrer/lenstronomy>`_
 * ``astropy``  `https://github.com/astropy/astropy <https://github.com/astropy/astropy>`_
 * ``photutils`` `https://github.com/astropy/photutils <https://github.com/astropy/photutils>`_
 * ``regions`` `https://github.com/astropy/regions <https://github.com/astropy/regions>`_
and related ones to be installed... 

Example notebooks
-----------------
We have create `notebooks <https://github.com/dartoon/decomprofile_notebooks>`_ to demonstrate how to use ``decomprofile``. These notebooks demonstrate how to model QSOs and galaxies using 2D Sersic profile and scaled point source, based on ``lenstronomy`` `lenstronomy <https://github.com/sibirrer/lenstronomy>`_.

Examples including:

* `Modeling a HSC imaged QSO <https://github.com/dartoon/decomprofile_notebooks/blob/master/decomprofile_HSC_QSO.ipynb>`_
* `Modeling a HSC imaged dual QSO <https://github.com/dartoon/decomprofile_notebooks/blob/master/decomprofile_HSC_dualAGN.ipynb>`_
* `Modeling a HST imaged QSO <https://github.com/dartoon/decomprofile_notebooks/blob/master/decomprofile_HST_QSO.ipynb>`_

Features
--------
The notebook demonstrate the follwing feature/functions:

* Search PSF stars through entire field of view, automatically.
* Cutout the target object and prepare the materials for the modelling.
* Estimate the background noise level from empty regions.
* Estimate the global background light and remove.
* Detecting objects in the cutout stamp and quickly create Sersic keywords (in ``lenstronomy`` type) to model them.
