# decomprofile
A python package that can be used to analysis and model the imaging data.

Features
------------
* Search PSF stars in the FOV.
* Estimate the background noise level.
* Cutout the target object galaxies (QSOs) and prepare the materials to model the data.
* Model them QSOs and galaxies using 2D Sersic profile and scaled point source, based on ``lenstronomy <https://github.com/sibirrer/lenstronomy>``

Demonstration:
------------
The notebooks of demonstrating how to use ``decomprofile`` can be found in:
https://github.com/dartoon/decomprofile_notebooks

The data used in the example notebook can be found `here <https://drive.google.com/file/d/1ZO9-HzV8K60ijYWK98jGoSoZHjIGW5Lc/view?usp=sharing>`:


Installation
------------
    $ git clone https://github.com/dartoon/decomprofile <desired location>
    $ cd <desired location>
    $ python setup.py install --user

Requirements
------------
packages including:
``lenstronomy <https://github.com/sibirrer/lenstronomy>``
