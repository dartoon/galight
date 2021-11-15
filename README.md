# galight - Galaxy shapes of Light
A Python-based open-source package that can be used to perform two-dimensional model fitting of optical and near-infrared images to characterize the light distribution of galaxies with components including a disk, bulge, bar and quasar.

![plot](./figs/fitting_result.png)

Installation
------------
    $ git clone https://github.com/dartoon/galight <desired location>
    $ cd <desired location>
    $ python setup.py install --user

Alternatively, ``galight`` can be installed through pip:

    $ pip install galight --user

Getting started:
------------
The [example notebook](https://github.com/dartoon/galight_notebooks/blob/master/galight_HST_QSO.ipynb) allows the user to quickly go through a fitting task on a HST observed AGN at z~1.5.

Some other notebooks of demonstrating how to use ``galight`` with more features can be found in:
https://github.com/dartoon/galight_notebooks

The data used in the example notebook can be found [here](https://drive.google.com/file/d/1ZO9-HzV8K60ijYWK98jGoSoZHjIGW5Lc/view?usp=sharing).

For more information, an online documentation can be found in ReadtheDocs:
https://galight.readthedocs.io/



Additional Features
------------
``galight`` helps the user to execute model fitting with ease based on, but not limited to, automated features as listed as follows:
* Search PSF stars in the FOV.
![plot](./figs/find_PSF.png)
* Automatically Estimate the background noise level.
![plot](./figs/est_bkgstd.png)
* Cutout the target object galaxies (QSOs) and prepare the materials to model the data.
* Detecting objects in the cutout stamp and quickly create Sersic keywords (in ``lenstronomy`` type) to model them.
* Model them QSOs and galaxies using 2D Sersic profile and scaled point source, based on [``lenstronomy``](https://github.com/sibirrer/lenstronomy).
![plot](./figs/fitting_sets.png)


Requirements
------------
packages including:
[``lenstronomy``](https://github.com/sibirrer/lenstronomy);
[``astropy``](https://github.com/astropy/astropy);
[``photutils``](https://github.com/astropy/photutils);
[``regions``](https://github.com/astropy/regions)
and related ones to be installed... 
