========
Usage
========

To use **galight** in your project::

	import galight

To check the current version of **decmoprofile**::

	print(galight.__version__)


Getting started
---------------

**galight** is a package that help to quickly set up the materials and input to **lenstronomy** to achieve the photomertic inference of Galaxy profile with AGN. 
We provide a set of example notebooks to demonstrate the modelling inference, with cases include:

* HST observed AGN:
* HSC SSP observed AGN:
* HSC SSP observed dual AGN:
* Galaxy with multiply components: (Disk, Bulge, Bar):


Besides, some useful tools are also provided to help achieve fast analysis and data reduction, include:

* Help to pick up the PSFs in the field of view
* Estimate the FWHM of the PSF
* Measurement 2D background light (based on Sextractor)
* Deblending images
* Calculate the 1D profiles of a image by drawing a set of apertures


