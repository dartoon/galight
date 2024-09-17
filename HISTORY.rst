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

* Add refresh in fit_run.run() to allow refresh the parameters manually.
* Add stack_PSF using PSFr.
* Update to avoid PSF negative pixel issue for new lenstronomy >= 1.10.4
* Allow linear_solver off to let 'amp' as free parameter.

0.1.11 (2023-4-07)
++++++++++++++++++

* Improve the final fitting plot.
* Improve the find_PSF() function with better filter.

0.1.12 (2023-6-12)
++++++++++++++++++

* Improve the find_PSF() function with random invalid string.
* Improve modifying kwargs_numerics in FittingSpecify to pass to FittingProcess.

0.1.13 (2023-6-27)
++++++++++++++++++
  
* Adding fit_run_result_to_apertures function to easily get the apertures properties.
* Fix the error when running MCMC in version 0.1.12. 

0.2.0 (2023-09-14)
++++++++++++++++++

* Add point_source_supersampling in prepare_fitting_seq()
* Add dump pickle version free function in fitting_process.
* Debug error for when PS only image.
* Update flux_sersic_model for when obj NO. > 1

0.2.1 (2024-09-06)
++++++++++++++++++

* Match lenstronomy version 1.12.0.
* Match photutils version 1.13.0 (new mask is used for estimating background light).
* Improve generate_target_materials() by allow 'error' input to detect_threshold.
* Improve linear_solver using '_imageModel'.
* Improve the report value for Chisq.
* Allow quick fit galaxy image to fits file with header.
* Allow set the scale bar size in .plot_final_qso/galaxy_fit().



 