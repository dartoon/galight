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

class FittingProcess(object):
    """
    A class to perform the fitting task:
        - define the way to fitting: PSO and MCMC
        - save all the useful fitting materials, if assign save_pkl
    """
    def __init__(self, fitting_specify_class, savename = 'result', zp = 27.0):
        self.fitting_specify_class = fitting_specify_class
        self.fitting_seq = fitting_specify_class.fitting_seq
        self.savename = savename
        self.zp = fitting_specify_class.zp
        
    def fitting_kwargs(self, algorithm_list = ['PSO', 'MCMC'], setting_list = [None, None]):
        if len(algorithm_list) != len(setting_list):
            raise ValueError("The algorithm_list and setting_list should be in the same length.") 
        fitting_kwargs_list = []
        for i in range(len(algorithm_list)):
            if setting_list[i] is None:
                setting = fitting_setting_temp(algorithm_list[i])
            else:
                setting = setting_list[i]
            fitting_kwargs_list.append([algorithm_list[i], setting])
        self.fitting_kwargs_list = fitting_kwargs_list
    
    def run(self):
        self.fitting_kwargs(algorithm_list = ['PSO', 'MCMC'], setting_list = [None, None])
        fitting_specify_class = self.fitting_specify_class
        start_time = time.time()
        chain_list = self.fitting_seq.fit_sequence(self.fitting_kwargs_list)
        kwargs_result = self.fitting_seq.best_fit()
        ps_result = kwargs_result['kwargs_ps']
        source_result = kwargs_result['kwargs_source']
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            self.sampler_type, self.samples_mcmc, self.param_mcmc, self.dist_mcmc  = chain_list[-1]    
        end_time = time.time()
        print(round(end_time - start_time, 3), 'total time taken for the overall fitting (s)')
        print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
        
        from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
        imageLinearFit = ImageLinearFit(data_class=fitting_specify_class.data_class, psf_class=fitting_specify_class.psf_class,
                                        source_model_class=fitting_specify_class.lightModel,
                                        point_source_class=fitting_specify_class.pointSource, 
                                        kwargs_numerics=fitting_specify_class.kwargs_numerics)    
        image_reconstructed, error_map, _, _ = imageLinearFit.image_linear_solve(kwargs_source=source_result,
                                                                                 kwargs_ps=ps_result)
        from lenstronomy.Plots.model_plot import ModelPlot
        # this is the linear inversion. The kwargs will be updated afterwards
        modelPlot = ModelPlot(fitting_specify_class.kwargs_data_joint['multi_band_list'],
                              fitting_specify_class.kwargs_model, kwargs_result,
                              arrow_size=0.02, cmap_string="gist_heat", 
                              likelihood_mask_list=fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'] )    

        imageModel = fitting_specify_class.imageModel
        image_host_list = []  #The linear_solver before and after LensModelPlot could have different result for very faint sources.
        for i in range(len(source_result)):
            image_host_list.append(imageModel.source_surface_brightness(source_result, de_lensed=True,unconvolved=False,k=i))
        
        image_ps_list = []
        for i in range(len(ps_result)):
            image_ps_list.append(imageModel.point_source(ps_result, k = i))
            
        if self.fitting_kwargs_list[-1][0] == 'MCMC':
            from lenstronomy.Sampling.parameters import Param
            param = Param(fitting_specify_class.kwargs_model, kwargs_fixed_source=fitting_specify_class.source_params[2],
                          kwargs_fixed_ps=fitting_specify_class.ps_params[2], **fitting_specify_class.kwargs_constraints)
            mcmc_flux_list = []
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
                kwargs_light_source_out = kwargs_out['kwargs_source']
                kwargs_ps_out =  kwargs_out['kwargs_ps']
                image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(kwargs_source=kwargs_light_source_out,
                                                                                      kwargs_ps=kwargs_ps_out)
                flux_list_quasar = []
                if len(fitting_specify_class.point_source_list) > 0:
                    for j in range(len(fitting_specify_class.point_source_list)):
                        image_ps_j = fitting_specify_class.imageModel.point_source(kwargs_ps_out, k=j)
                        flux_list_quasar.append(np.sum(image_ps_j))
                flux_list_galaxy = []
                for j in range(len(fitting_specify_class.light_model_list)):
                    image_j = fitting_specify_class.imageModel.source_surface_brightness(kwargs_light_source_out,unconvolved= False, k=j)
                    flux_list_galaxy.append(np.sum(image_j))
                mcmc_flux_list.append(flux_list_quasar + flux_list_galaxy )
                if int(i/1000) > int((i-1)/1000) :
                    print(trans_steps[1]-trans_steps[0],
                          "MCMC samplers in total, finished translate:", i-trans_steps[0] )
            self.mcmc_flux_list = mcmc_flux_list
            self.labels_flux = labels_flux            
        self.chain_list = chain_list
        self.kwargs_result = kwargs_result
        self.ps_result = ps_result
        self.source_result = source_result
        self.modelPlot = modelPlot
        self.imageLinearFit = imageLinearFit
        self.reduced_Chisq =  imageLinearFit.reduced_chi2(image_reconstructed, error_map)
        self.image_host_list = image_host_list
        self.image_ps_list = image_ps_list
        self.translate_result()

    def run_diag(self, diag_list = None, show_plot = True):
        """
        The purpose of this def
        
        Parameter
        --------
            diag_list: list of int
            which chains to show?
            
        Return
        --------
            A sth sth
            
        #TODO: Add save plot.
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
        f, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=False, sharey=False)
        self.modelPlot.data_plot(ax=axes[0,0], text="Data")
        self.modelPlot.model_plot(ax=axes[0,1])
        self.modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
        
        self.modelPlot.decomposition_plot(ax=axes[1,0], text='Host galaxy', source_add=True, unconvolved=True)
        self.modelPlot.decomposition_plot(ax=axes[1,1], text='Host galaxy convolved', source_add=True)
        self.modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
        
        self.modelPlot.subtract_from_data_plot(ax=axes[2,0], text='Data - Point Source', point_source_add=True)
        self.modelPlot.subtract_from_data_plot(ax=axes[2,1], text='Data - host galaxy', source_add=True)
        self.modelPlot.subtract_from_data_plot(ax=axes[2,2], text='Data - host galaxy - Point Source', source_add=True, point_source_add=True)
        f.tight_layout()
        if save_plot == True:
            plt.savefig('{0}_model.pdf'.format(self.savename))  
        if show_plot == True:
            plt.show()
        else:
            plt.close()

    def plot_params_corner(self, save_plot = False, show_plot = True):
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
        plot = corner.corner(self.mcmc_flux_list, labels=self.labels_flux, show_titles=True)
        if save_plot == True:
            savename = self.savename
            plot.savefig('{0}_flux_corner.pdf'.format(savename))
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    def plot_final_qso_fit(self, if_annuli=False, show_plot = True, arrows=False, save_plot = False):
        from decomprofile.tools.plot_tools import total_compare
        data = self.fitting_specify_class.kwargs_data['image_data']
        noise = self.fitting_specify_class.kwargs_data['noise_map']
        ps_list = self.image_ps_list
        ps_image = np.zeros_like(ps_list[0])
        for i in range(len(ps_list)):
            ps_image = ps_image+ps_list[i]
        galaxy_list = self.image_host_list
        galaxy_image = np.zeros_like(galaxy_list[0])
        for i in range(len(galaxy_list)):
            galaxy_image = galaxy_image+galaxy_list[i]
        model = ps_image + galaxy_image
        data_removePSF = data - ps_image
        norm_residual = (data - model)/noise
        flux_list_2d = [data, model, data_removePSF, norm_residual]
        label_list_2d = ['data', 'model', 'data-Point Source', 'normalized residual']
        flux_list_1d = [data, model, ps_image, galaxy_image]
        label_list_1d = ['data', 'model', 'Point Source', '{0} galaxy(s)'.format(len(galaxy_list))]
        fig = total_compare(flux_list_2d, label_list_2d, flux_list_1d, label_list_1d, deltaPix = self.fitting_specify_class.deltaPix,
                      zp=self.zp, if_annuli=if_annuli, arrows= arrows, show_plot = show_plot,
                      mask_image = self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'][0])
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        if save_plot == True:
            savename = self.savename
            fig.savefig(savename+"_qso_final_plot.pdf")   

    def plot_final_galaxy_fit(self, if_annuli=False, show_plot = True, arrows=False, save_plot = False):
        from decomprofile.tools.plot_tools import total_compare
        data = self.fitting_specify_class.kwargs_data['image_data']
        noise = self.fitting_specify_class.kwargs_data['noise_map']
        galaxy_list = self.image_host_list
        galaxy_image = np.zeros_like(galaxy_list[0])
        for i in range(len(galaxy_list)):
            galaxy_image = galaxy_image+galaxy_list[i]
        model = galaxy_image
        norm_residual = (data - model)/noise
        flux_list_2d = [data, model, norm_residual]
        label_list_2d = ['data', 'model', 'normalized residual']
        flux_list_1d = [data, model]
        label_list_1d = ['data', 'model ({0} galaxy(s))'.format(len(galaxy_list))]
        fig = total_compare(flux_list_2d, label_list_2d, flux_list_1d, label_list_1d, deltaPix = self.fitting_specify_class.deltaPix,
                      zp=self.zp, if_annuli=if_annuli, arrows= arrows, show_plot = show_plot,
                      mask_image = self.fitting_specify_class.kwargs_likelihood['image_likelihood_mask_list'][0])
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        if save_plot == True:
            savename = self.savename
            fig.savefig(savename+"_qso_final_plot.pdf")   
    
    def plot_all(self):
        self.run_diag()
        self.model_plot()
        self.plot_params_corner()
        self.plot_flux_corner()  
        if self.image_ps_list != []:
            self.plot_final_qso_fit()
        else:
            self.plot_final_galaxy_fit()

    def translate_result(self):
        import lenstronomy.Util.param_util as param_util
        from decomprofile.tools.measure_tools import model_flux_cal
        self.final_galaxy_result = copy.deepcopy(self.source_result)
        flux_sersic_model = model_flux_cal(self.final_galaxy_result)
        for i in range(len(self.final_galaxy_result)):
            source = self.final_galaxy_result[i]
            source['phi_G'], source['q'] = param_util.ellipticity2phi_q(source['e1'], source['e2'])
            source['flux_sersic_model'] = flux_sersic_model[i]
            source['flux_within_frame'] = np.sum(self.image_host_list[i])
            source['magnitude'] = -2.5*np.log10(source['flux_within_frame']) + self.zp
            self.final_galaxy_result[i] = source
        self.final_ps_result = copy.deepcopy(self.ps_result)
        for i in range(len(self.final_ps_result)):
            ps = self.final_ps_result[i]
            ps['flux_within_frame'] = np.sum(self.image_ps_list[i])
            ps['magnitude'] = -2.5*np.log10(ps['flux_within_frame']) + self.zp      
            
    def mcmc_result_range(self, chain=None, param=None):
        if chain is None:
            chain = self.samples_mcmc
            param = self.param_mcmc
        for i in range(len(param)):
            print(i, ':', param[i])
        checkid = int(input("Which parameter to checkout?\n"))
        print("Low {0:.3f}, Mid {1:.3f}, High: {2:.3f}".format(np.percentile(chain[:, checkid],16),
                                                                np.percentile(chain[:, checkid],50), 
                                                                np.percentile(chain[:, checkid],84)) )
        #TODO: Add q and theta range 
        

    def dump_result(self):
        savename = self.savename
        dump_class = copy.deepcopy(self)
        if hasattr(dump_class.fitting_specify_class, 'data_process_class'):
            del dump_class.fitting_specify_class.data_process_class
        pickle.dump(dump_class, open(savename+'.pkl', 'wb'))     
    
def fitting_setting_temp(algorithm, fill_value_list = None):
    if algorithm == 'PSO':
        if fill_value_list is None:
            # setting = {'sigma_scale': 0.8, 'n_particles': 150, 'n_iterations': 150}
            setting = {'sigma_scale': 0.8, 'n_particles': 50, 'n_iterations': 50}
        else:
            setting = {'sigma_scale': fill_value_list[0], 'n_particles': fill_value_list[1], 'n_iterations': fill_value_list[2]}
    elif algorithm == 'MCMC':     
        if fill_value_list is None:        
            # setting = {'n_burn': 100, 'n_run': 200, 'walkerRatio': 10, 'sigma_scale': .1}
            setting = {'n_burn': 50, 'n_run': 100, 'walkerRatio': 10, 'sigma_scale': .1}
        else:
            setting = {'n_burn': fill_value_list[0], 'n_run': fill_value_list[1],
                       'walkerRatio': fill_value_list[2], 'sigma_scale': fill_value_list[3]}
    return setting

#TODO: add other priors? i.e., the q? host flux ratio?
#TODO: Translate MCMC Chains to 1 sigma.