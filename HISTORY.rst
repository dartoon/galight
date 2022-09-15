.. :changelog:

History
-------

0.0.0 (2021-05-31)
++++++++++++++++++

* First test upload on PyPI.

0.1.0 (2021-11-16)
++++++++++++++++++

* First release on PyPI.

0.1.1 (2021-11-16)
++++++++++++++++++

* Debug the aperture alignment.
* Calculate astrometry information for objs.
* Remove any objs in the original fov image.
* Include the HSC_utils by Connor.

0.1.2 (2021-12-06)
++++++++++++++++++

* Debug the aperture alignment issue. 
* Synchronous upgrade for other package.
* A bridge between galight and statmorph.

0.1.3 (2021-12-15)
++++++++++++++++++

* Add sum_mask feature in mask_obj()
* Update flux_list_2d into dict.
* Maintenance for error.

0.1.4 (2021-12-23)
++++++++++++++++++

* Debug and update the setting for sersic_major_axis following lenstronomy.Conf


0.1.5 (2022-03-09)
++++++++++++++++++

* Debug for no error report in detect_obj function
* A preliminary version for non-parametric CAS calculation
* More functionality with plot tools


0.1.6 (2022-04-18)
++++++++++++++++++

* Debug for 'sersic_model_flux' calculation caused by "sersic_major_axis" setting
* Upgrade with more CAS calculation.
* Improve the MCMC output using 'mcmc_source_result' dict to save all samplers with amp values.


0.1.7 (2022-05-30)
++++++++++++++++++

* Update initial Reff value based on Moments measure.
* Improve the plot to highlight masks.
* Separate a plot_materials function in data_process.


0.1.8 (2022-07-16)
++++++++++++++++++

* Improve reduced chisq calculations.
* Improve stretch of showing image and use bbox_inches='tight'.
* Debug for functions including cal_statmorph() and targets_subtraction().
* Improve PSF selection function.
* Update for lenstronomy > 1.10.3 and new photutils version 1.5.0


0.1.9 (2022-07-18)
++++++++++++++++++

* Add PSF stacking function using photutils.EPSFBuilder 
* Improve the CAS measurement 


0.1.10 (2022-09-15)
++++++++++++++++++

* Add refresh in fit_run.run() to allow refresh the parameters manually
* Add stack_PSF using PSFr
* Update to avoid PSF negative pixel issue for new lenstronomy >= 1.10.4
* Allow linear_solver off to let 'amp' as free parameter
 