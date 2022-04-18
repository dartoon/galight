#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:06:14 2020

@author: Xuheng Ding
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import time
import corner
import pickle
import copy
import matplotlib as matt
matt.rcParams['font.family'] = 'STIXGeneral'
from lenstronomy.Plots.model_plot import ModelPlot
from galight.tools.plot_tools import total_compare
from packaging import version
import lenstronomy
import lenstronomy.Util.param_util as param_util
from galight.tools.measure_tools import model_flux_cal

class FittingProcess(object):
    """
    A class to perform the fitting task and show the result. 
        - define the way to fitting: PSO and MCMC
        - save all the useful fitting materials, if assign save_pkl
        - make plots to show the fittings.
    
    Parameter
    --------
        fitting_level: String.
            Defines the depth of the fitting
                - 'deep', perfer a deep fitting, with more fitting PSO and/or MCMC particles, which takes more time.
                - 'shallow', perfer a quick fitting.
    """
    def __init__(self, fitting_specify_class, savename = 'result', fitting_level='norm'):
        fitting_specify_class.build_fitting_seq()
        self.fitting_specify_class = fitting_specify_class
        self.fitting_seq = fitting_specify_class.fitting_seq
        self.savename = savename
        self.zp = fitting_specify_class.zp
        self.fitting_level = fitting_level
        self.sersic_major_axis = fitting_specify_class.sersic_major_axis
        
    def fitting_kwargs(self, algorithm_list = ['PSO', 'MCMC'], setting_list = [None, None], threadCount=1):
        """
        Define the fitting steps. The 'PSO' and 'MCMC' are defined here. 
        
        Parameter
        --------
            algorithm_list: list.
                Define the steps for the fitting, e.g., ['PSO', 'PSO', 'MCMC'].
            
            setting_list: list of fitting setting.
                Setting are done by 'fitting_setting_temp()' with three options.
        """
        if len(algorithm_list) != len(setting_list):
            raise ValueError("The algorithm_list and setting_list should be in the same length.") 
        fitting_kwargs_list = []
        for i in range(len(algorithm_list)):
            if setting_list[i] is None:
                if isinstance(self.fitting_level, str):
                    setting = fitting_setting_temp(algorithm_list[i], fitting_level = self.fitting_level)
                elif isinstance(self.fitting_level, list):
                    setting = fitting_setting_temp(algorithm_list[i], fitting_level = self.fitting_level[i])
            else:
                setting = setting_list[i]
            if self.fitting_seq._mpi == True:
                setting['threadCount'] = threadCount
            fitting_kwargs_list.append([algorithm_list[i], setting])
        self.fitting_kwargs_list = fitting_kwargs_list
    
    def run(self, algorithm_list = ['PSO', 'MCMC'], setting_list = None, threadCount=1):
        """
        Run the fitting. The algorithm_list and setting_list will be pass to 'fitting_kwargs()'
        """
        if setting_list is None:
            setting_list = [None] * len(algorithm_list)
        self.fitting_kwargs(algorithm_list = algorithm_list, setting_list = setting_list, threadCount=threadCount)
        fitting_specify_class = self.fitting_specify_class
        start_time = time.time()
        chain_list = self.fitting_seq.fit_sequence(self.fitting_kwargs_list)
        kwargs_result = self.fitting_seq.best_fit()
        ps_result = kwargs_result['kwargs_ps']
        source_result = kwargs_result['kwargs_lens_light']
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            self.sampler_type, self.samples_mcmc, self.param_mcmc, self.dist_mcmc  = chain_list[-1]    
        end_time = time.time()
        print(round(end_time - start_time, 3), 'total time taken for the overall fitting (s)')
        print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
        
        from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
        imageLinearFit = ImageLinearFit(data_class=fitting_specify_class.data_class, 
                                        psf_class=fitting_specify_class.psf_class,
                                        lens_light_model_class=fitting_specify_class.lightModel,
                                        point_source_class=fitting_specify_class.pointSource, 
                                        kwargs_numerics=fitting_specify_class.kwargs_numerics)    
        image_reconstructed, error_map, _, _ = imageLinearFit.image_linear_solve(kwargs_lens_light=source_result,
                                                                                 kwargs_ps=ps_result)
        imageModel = fitting_specify_class.imageModel
        image_host_list = []  #The linear_solver before and after LensModelPlot could have different result for very faint sources.
        for i in range(len(source_result)):
            image_host_list.append(imageModel.lens_surface_brightness(source_result, unconvolved=False,k=i))
        
        image_ps_list = []
        for i in range(len(ps_result)):
            image_ps_list.append(imageModel.point_source(ps_result, k = i))
            
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            from lenstronomy.Sampling.parameters import Param
            try:
                kwargs_fixed_source = fitting_specify_class.kwargs_params['lens_light_model'][2]
            except:
                kwargs_fixed_source = None

            try:
                kwargs_fixed_ps=fitting_specify_class.kwargs_params['point_source_model'][2]
            except:
                kwargs_fixed_ps = None
            param = Param(fitting_specify_class.kwargs_model, kwargs_fixed_lens_light=kwargs_fixed_source,
                          kwargs_fixed_ps=kwargs_fixed_ps, **fitting_specify_class.kwargs_constraints)
            mcmc_flux_list = []
            mcmc_source_result = []
            if len(fitting_specify_class.point_source_list) >0 :
                qso_labels_new = ["Quasar_{0} flux".format(i) for i in range(len(fitting_specify_class.point_source_list))]
                galaxy_labels_new = ["Galaxy_{0} flux".format(i) for i in range(len(fitting_specify_class.light_model_list))]
                labels_flux = qso_labels_new + galaxy_labels_new
            else:
                labels_flux = ["Galaxy_{0} flux".format(i) for i in range(len(fitting_specify_class.light_model_list))]
            if len(self.samples_mcmc) > 10000:  #Only save maximum 10000 chain results.
                trans_steps = [len(self.samples_mcmc)-10000, len(self.samples_mcmc)]
            else:
                trans_steps = [0, len(self.samples_mcmc)]
            print("Start transfering the Params to fluxs...")
            for i in range(trans_steps[0], trans_steps[1]):
                kwargs_out = param.args2kwargs(self.samples_mcmc[i])
                kwargs_light_source_out = kwargs_out['kwargs_lens_light']
                kwargs_ps_out =  kwargs_out['kwargs_ps']
                image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(kwargs_lens_light=kwargs_light_source_out,
                                                                                      kwargs_ps=kwargs_ps_out)
                flux_list_quasar = []
                if len(fitting_specify_class.point_source_list) > 0:
                    for j in range(len(fitting_specify_class.point_source_list)):
                        image_ps_j = fitting_specify_class.imageModel.point_source(kwargs_ps_out, k=j)
                        flux_list_quasar.append(np.sum(image_ps_j))
                flux_list_galaxy = []
                mcmc_source_result.append(kwargs_light_source_out)
                for j in range(len(fitting_specify_class.light_model_list)):
                    image_j = fitting_specify_class.imageModel.lens_surface_brightness(kwargs_light_source_out,unconvolved= False, k=j)
                    flux_list_galaxy.append(np.sum(image_j))
                    # _flux_sersic_model = model_flux_cal(kwargs_light_source_out, sersic_major_axis=self.sersic_major_axis)
                    # mcmc_sersic_model_flux.append(_flux_sersic_model)
                mcmc_flux_list.append(flux_list_quasar + flux_list_galaxy )
                if int(i/1000) > int((i-1)/1000) :
                    print(trans_steps[1]-trans_steps[0],
                          "MCMC samplers in total, finished translate:", i-trans_steps[0] )
            self.mcmc_flux_list = np.array(mcmc_flux_list)
            # self.mcmc_sersic_model_flux = mcmc_sersic_model_flux
            self.labels_flux = labels_flux            
            mcmc_source_result_dict = {}
            p_labels = mcmc_source_result[0][0].keys()
            for i in range(len(mcmc_source_result)):
                for j in range(len(mcmc_source_result[0])):
                    for label in p_labels:
                        if i == 0:
                            mcmc_source_result_dict[label+'_'+ str(j)] = []
                        mcmc_source_result_dict[label+'_'+ str(j)].append(mcmc_source_result[i][j][label])
            for key in mcmc_source_result_dict.keys():
                mcmc_source_result_dict[key] = np.array(mcmc_source_result_dict[key])
            self.mcmc_source_result = mcmc_source_result_dict
        self.chain_list = chain_list
        self.kwargs_result = kwargs_result
        self.ps_result = ps_result
        self.source_result = source_result
        self.imageLinearFit = imageLinearFit
        self.reduced_Chisq =  imageLinearFit.reduced_chi2(image_reconstructed, error_map)
        self.image_host_list = image_host_list
        self.image_ps_list = image_ps_list
        self.translate_result()

    def run_diag(self, diag_list = None, show_plot = True):
        """
        Plot the fitting particles and show how they converge. 
        
        Parameter
        --------
            diag_list: None or list of int, e.g., [0, 1]
                Defines which chains to show?
        """         
        from lenstronomy.Plots import chain_plot
        if diag_list is None:
            for i in range(len(self.chain_list)):
                f, axes = chain_plot.plot_chain_list(self.chain_list,i)
        else:
            for i in diag_list:
                f, axes = chain_plot.plot_chain_list(self.chain_list,i)
        if show_plot == True:
            plt.show()
        else:
            plt.close()

    def model_plot(self, save_plot = False, show_plot = True):
        """
        Show the fitting plot based on lenstronomy.Plots.model_plot.ModelPlot
        """
        # this is the linear inversion. The kwargs will be updated afterwards
        modelPlot = ModelPlot(self.fitting_specify_class.kwargs_data_joint['multi_band_list'],
                              self.fitting_specify_class.kwargs_model, self.kwargs_result,
                              arrow_size=0.02, cmap_string="gist_heat", 
                              likelihood_mask_list=self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'] )    
        
        f, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=False, sharey=False)
        modelPlot.data_plot(ax=axes[0,0], text="Data")
        modelPlot.model_plot(ax=axes[0,1])
        modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
        
        modelPlot.decomposition_plot(ax=axes[1,0], text='Host galaxy', lens_light_add=True, unconvolved=True)
        modelPlot.decomposition_plot(ax=axes[1,1], text='Host galaxy convolved', lens_light_add=True)
        modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
        
        modelPlot.subtract_from_data_plot(ax=axes[2,0], text='Data - Point Source', point_source_add=True)
        modelPlot.subtract_from_data_plot(ax=axes[2,1], text='Data - host galaxy', lens_light_add=True)
        modelPlot.subtract_from_data_plot(ax=axes[2,2], text='Data - host galaxy - Point Source', lens_light_add=True, point_source_add=True)
        f.tight_layout()
        if save_plot == True:
            plt.savefig('{0}_model.pdf'.format(self.savename))  
        if show_plot == True:
            plt.show()
        else:
            plt.close()

    def plot_params_corner(self, save_plot = False, show_plot = True):
        """
        Show the MCMC parametere corner plots.
        """        
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            samples_mcmc = self.samples_mcmc
            n, num_param = np.shape(samples_mcmc)
            plot = corner.corner(samples_mcmc, labels=self.param_mcmc, show_titles=True)
            if show_plot == True:
                plt.show()
            else:
                plt.close()
        else:
            print("Please make sure MCMC has been performed in the last-fitting step rather than", self.fitting_kwargs_list[-1][0])
        if save_plot == True:
            savename = self.savename
            plot.savefig('{0}_params_corner.pdf'.format(savename)) 
            
    def plot_flux_corner(self, save_plot = False, show_plot = True):
        """
        After translate the MCMC parameter values into flux. This function plots the corner plot.
        """          
        plot = corner.corner(self.mcmc_flux_list, labels=self.labels_flux, show_titles=True)
        if save_plot == True:
            savename = self.savename
            plot.savefig('{0}_flux_corner.pdf'.format(savename))
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    def plot_final_qso_fit(self, if_annuli=False, show_plot = True, arrows=False, save_plot = False, target_ID = None):
        """
        Plot the compact fitting result, if a QSO is fitted.
        """ 
        data = self.fitting_specify_class.kwargs_data['image_data']
        if 'psf_error_map' in self.fitting_specify_class.kwargs_psf.keys():
            modelPlot = ModelPlot(self.fitting_specify_class.kwargs_data_joint['multi_band_list'],
                                  self.fitting_specify_class.kwargs_model, self.kwargs_result,
                                  arrow_size=0.02, cmap_string="gist_heat", 
                                  likelihood_mask_list=self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'] )    
            _, psf_error_map, _, _ = modelPlot._imageModel.image_linear_solve(inv_bool=True, **self.kwargs_result)
            noise = np.sqrt(self.fitting_specify_class.kwargs_data['noise_map']**2+np.abs(psf_error_map[0]))
        else:
            noise = self.fitting_specify_class.kwargs_data['noise_map']
        
        ps_list = self.image_ps_list
        ps_image = np.zeros_like(ps_list[0])
        if target_ID is None:
            target_ID = 'target_ID'
        for i in range(len(ps_list)):
            ps_image = ps_image+ps_list[i]
        galaxy_list = self.image_host_list
        galaxy_image = np.zeros_like(data)
        for i in range(len(galaxy_list)):
            galaxy_image = galaxy_image+galaxy_list[i]
        model = ps_image + galaxy_image
        data_removePSF = data - ps_image
        norm_residual = (data - model)/noise
        flux_dict_2d = {'data':data, 'model':model, 'data-Point Source':data_removePSF, 'normalized residual':norm_residual}
        self.flux_2d_out = flux_dict_2d
        flux_dict_1d = {'data':data, 'model':model, 'Point Source':ps_image, '{0} galaxy(s)'.format(len(galaxy_list)):galaxy_image}
        self.flux_1d_out = flux_dict_1d
        fig = total_compare(list(flux_dict_2d.values()), list(flux_dict_2d.keys()), list(flux_dict_1d.values()), list(flux_dict_1d.keys()), deltaPix = self.fitting_specify_class.deltaPix,
                      zp=self.zp, if_annuli=if_annuli, arrows= arrows, show_plot = show_plot,
                      mask_image = self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'][0],
                      target_ID = target_ID)
        if save_plot == True:
            savename = self.savename
            fig.savefig(savename+"_qso_final_plot.pdf")   
        if show_plot == True:
            plt.show()
        else:
            plt.close()

    def plot_final_galaxy_fit(self, if_annuli=False, show_plot = True, arrows=False, save_plot = False, target_ID = None):
        """
        Plot the compact fitting result, if galaxies is fitted (i.e., no point source).
        """         
        data = self.fitting_specify_class.kwargs_data['image_data']
        noise = self.fitting_specify_class.kwargs_data['noise_map']
        galaxy_list = self.image_host_list
        galaxy_image = np.zeros_like(galaxy_list[0])
        if target_ID is None:
            target_ID = 'target_ID'
        for i in range(len(galaxy_list)):
            galaxy_image = galaxy_image+galaxy_list[i]
        model = galaxy_image
        norm_residual = (data - model)/noise
        flux_dict_2d = {'data':data, 'model':model, 'normalized residual':norm_residual}
        self.flux_2d_out = flux_dict_2d
        flux_dict_1d = {'data':data, 'model ({0} galaxy(s))'.format(len(galaxy_list)):model}
        self.flux_1d_out = flux_dict_1d
        fig = total_compare(list(flux_dict_2d.values()), list(flux_dict_2d.keys()), list(flux_dict_1d.values()), list(flux_dict_1d.keys()), deltaPix = self.fitting_specify_class.deltaPix,
                      zp=self.zp, if_annuli=if_annuli, arrows= arrows, show_plot = show_plot,
                      mask_image = self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'][0],
                      target_ID = target_ID)
        if save_plot == True:
            savename = self.savename
            fig.savefig(savename+"_galaxy_final_plot.pdf")   
        if show_plot == True:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, target_ID=None):
        """
        Plot everyting, including:
            -run_diag()
            -model_plot()
            -plot_params_corner()
            -plot_flux_corner()
            -plot_final_qso_fit() or plot_final_galaxy_fit(), based on if point source is included or not.
            
            
        """          
        self.run_diag()
        self.model_plot()
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            self.plot_params_corner()
            self.plot_flux_corner()  
        if self.image_ps_list != []:
            self.plot_final_qso_fit(target_ID=target_ID)
        else:
            self.plot_final_galaxy_fit(target_ID=target_ID)

    def translate_result(self):
        """
        Translate some parameter results to make the fitting more readable, including the flux value, and the elliptical.
        """
        self.final_result_galaxy = copy.deepcopy(self.source_result)
        flux_sersic_model = model_flux_cal(self.final_result_galaxy, sersic_major_axis=self.sersic_major_axis)
        for i in range(len(self.final_result_galaxy)):
            source = self.final_result_galaxy[i]
            source['phi_G'], source['q'] = param_util.ellipticity2phi_q(source['e1'], source['e2'])
            source['flux_sersic_model'] = flux_sersic_model[i]
            source['flux_within_frame'] = np.sum(self.image_host_list[i])
            source['magnitude'] = -2.5*np.log10(source['flux_within_frame']) + self.zp
            self.final_result_galaxy[i] = source
        
        self.final_result_ps = copy.deepcopy(self.ps_result)
        for i in range(len(self.final_result_ps)):
            ps = self.final_result_ps[i]
            ps['flux_within_frame'] = np.sum(self.image_ps_list[i])
            ps['magnitude'] = -2.5*np.log10(ps['flux_within_frame']) + self.zp  
            self.final_result_ps[i] = ps
        
    def cal_astrometry(self):
        from astropy.wcs import WCS
        header = self.fitting_specify_class.header
        wcs = WCS(header)
        # pos = wcs.all_world2pix([[ra, dec]], 1)[0]
        target_pos = self.fitting_specify_class.target_pos
        wcs.all_pix2world([target_pos], 1)[0]
        deltaPix = self.fitting_specify_class.deltaPix
        for i in range(len(self.final_result_ps)):
            x, y = -self.final_result_ps[i]['ra_image'][0]/deltaPix, self.final_result_ps[i]['dec_image'][0]/deltaPix
            x_orgframe = x + target_pos[0] + 1
            y_orgframe = y + target_pos[1] + 1
            target_ra, target_dec = wcs.all_pix2world([[x_orgframe, y_orgframe]], 1)[0]
            self.final_result_ps[i]['wcs_RaDec'] = target_ra, target_dec
            self.final_result_ps[i]['position_xy'] = x, y
            
        for i in range(len(self.final_result_galaxy)):
            x, y = -self.final_result_galaxy[i]['center_x']/deltaPix, self.final_result_galaxy[i]['center_y']/deltaPix
            x_orgframe = x + target_pos[0] + 1
            y_orgframe = y + target_pos[1] + 1
            target_ra, target_dec = wcs.all_pix2world([[x_orgframe, y_orgframe]], 1)[0]
            self.final_result_galaxy[i]['wcs_RaDec'] = target_ra, target_dec
            self.final_result_galaxy[i]['position_xy'] = x, y
    
    def targets_subtraction(self, sub_gal_list = [], sub_qso_list = [], org_fov_data = None, 
                           save_fitsfile = False):
        """
        Subtract the target from the FOV image, based on the infernece. 
        
        Parameter
        --------
            sub_gal_list: list of int number.
                A list of galaxies will be removed.
                
            sub_qso_list: list of int number.
                A list of qso/stars to be removed.
                
            org_fov_data: 2D array.
                Input a the original FOV data, whose pixel grides should be the same as original input.
            
        Return
        --------
            A FOV image that remove a certain of fitted targets.
        """
        if org_fov_data is None:
            target_removed_fov_data = copy.deepcopy(self.fitting_specify_class.data_process_class.fov_image)
        else:
            target_removed_fov_data = org_fov_data
        header = self.fitting_specify_class.data_process_class.header
        target_pos = self.fitting_specify_class.data_process_class.target_pos
        fmr = int(len(self.fitting_specify_class.kwargs_data['image_data'])/2)
        x_range = target_pos[0]-fmr, target_pos[0]+fmr
        y_range = target_pos[1]-fmr, target_pos[1]+fmr
        remove_ = target_removed_fov_data[y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1 ]
        for i in sub_gal_list:
            remove_  -= self.image_host_list[i]
        for i in sub_qso_list:
            remove_  -= self.image_ps_list[i]
        self.fov_image_targets_sub = target_removed_fov_data
        if save_fitsfile == True:
            pyfits.PrimaryHDU(self.fov_image_targets_sub,header=header).writeto(self.savename+'_target_removed_fov.fits',overwrite=True)

    def cal_statmorph(self, obj_id=0, segm = None, if_plot=False):
        import statmorph
        if segm is None:
            segm = self.fitting_specify_class.segm_deblend
        if isinstance(self.fitting_specify_class.segm_deblend, (np.ndarray)):
            data = segm
        else:
            data = segm.data
            
        obj_id = obj_id
        apertures = self.fitting_specify_class.apertures
        pix_pos = np.intc(apertures[obj_id].positions)
        seg_idx = data[pix_pos[1], pix_pos[0]]
        segmap = data == seg_idx
        import scipy.ndimage as ndi
        segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
        segmap_ = segmap_float > 0.5
        if np.sum(segmap_)>10:
            segmap = segmap_
        mask = np.zeros_like(data, dtype=np.bool)
        for i in range(1,data.max()+1):
            if i != seg_idx:
                mask_  = data == i
                mask = mask + mask_
        feeddata = copy.deepcopy(self.fitting_specify_class.kwargs_data['image_data'])
        for i in range(len(self.image_host_list)):
            if i != obj_id:
                feeddata -= self.image_host_list[i]
        for i in range(len(self.image_ps_list)):
            feeddata -= self.image_ps_list[i]
        if if_plot:
            from matplotlib.colors import LogNorm
            fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(14, 10))
            im1 = ax1.imshow(feeddata, origin='lower', norm=LogNorm(vmax = feeddata.max(), vmin = 1.e-4))
            ax1.set_title('Data input to starmorph', fontsize=25)
            fig.colorbar(im1, ax=ax1, pad=0.01,  orientation="horizontal")
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False) 
            im2 = ax2.imshow(segmap, origin='lower')
            ax2.set_title('Segmap', fontsize=25)
            fig.colorbar(im2, ax=ax2, pad=0.01,  orientation="horizontal")
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False) 
            im3 = ax3.imshow(feeddata * (1-mask), origin='lower', norm=LogNorm(vmax = feeddata.max(), vmin = 1.e-4))
            ax3.set_title('data * mask', fontsize=25)
            fig.colorbar(im3, ax=ax3, pad=0.01,  orientation="horizontal")
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False) 
            plt.show()         
        
        source_morphs = statmorph.source_morphology(feeddata, segmap, 
                                                    weightmap=self.fitting_specify_class.kwargs_data['noise_map'], 
                                                    psf=self.fitting_specify_class.kwargs_psf['kernel_point_source'],mask = mask)
        return source_morphs[0]
        
    def mcmc_result_range(self, chain=None, param=None):
        """
        Quick checkout the MCMC fitting 1-sigma range.
        """ 
        if chain is None:
            chain = self.samples_mcmc
            param = self.param_mcmc
        for i in range(len(param)):
            print(i, ':', param[i])
        checkid = int(input("Which parameter to checkout?\n"))
        print("Low {0:.3f}, Mid {1:.3f}, High: {2:.3f}".format(np.percentile(chain[:, checkid],16),
                                                                np.percentile(chain[:, checkid],50), 
                                                                np.percentile(chain[:, checkid],84)) )
    def dump_result(self, savedata= False):
        """
        Save all the fitting materials as pickle for the future use. To save space, the data_process_class() will be removed, since it usually includes FOV image which can be huge.
        """        
        savename = self.savename
        dump_class = copy.deepcopy(self)
        if hasattr(dump_class.fitting_specify_class, 'data_process_class') and savedata==False:
            del dump_class.fitting_specify_class.data_process_class
        if dump_class.fitting_specify_class.kwargs_likelihood['custom_logL_addition'] != None:
            dump_class.prior = str(dump_class.fitting_specify_class.kwargs_likelihood['custom_logL_addition'])
            del dump_class.fitting_specify_class.kwargs_likelihood['custom_logL_addition']
        pickle.dump(dump_class, open(savename+'.pkl', 'wb'))    
    
def fitting_setting_temp(algorithm, fitting_level = 'norm'):
    """
    Quick setting up the fitting particles for the 'PSO' and 'MCMC'.
    
    Parameter
    --------
        fill_value_list: 
            A list of values to fill in to the settings.
            
    Return
    --------
        Fitting particle settings.
    """    
    if algorithm == 'PSO':
        if fitting_level == 'deep':
            setting = {'sigma_scale': 0.8, 'n_particles': 100, 'n_iterations': 150}
        else:
            setting = {'sigma_scale': 0.8, 'n_particles': 50, 'n_iterations': 50}
    elif algorithm == 'MCMC':     
        if fitting_level == 'deep':            
            setting = {'n_burn': 100, 'n_run': 200, 'walkerRatio': 10, 'sigma_scale': .1}
        else:
            setting = {'n_burn': 100, 'n_run': 30, 'walkerRatio': 10, 'sigma_scale': .1}
    return setting

