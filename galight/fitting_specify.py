#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:16:43 2020

@author: Xuheng Ding
"""

import numpy as np
import copy
import lenstronomy
from packaging import version
import warnings

class FittingSpecify(object):
    """
    A class to generate the materials for the 'FittingSequence', defined by 'lenstronomy'
    key materials include the following, which are prepared by 'prepare_fitting_seq()':
        - kwargs_data_joint: data materils
        - kwargs_model: a list of class 
        - kwargs_constraints
        - kwargs_likelihood
        - kwargs_params
        - imageModel 
    """    
    def __init__(self, data_process_class, sersic_major_axis=True):
        self.data_process_class = data_process_class
        self.deltaPix = data_process_class.deltaPix
        self.numPix = len(self.data_process_class.target_stamp)
        self.zp = data_process_class.zp
        self.apertures = copy.deepcopy(data_process_class.apertures)
        self.header = copy.deepcopy(data_process_class.header)
        self.target_pos = copy.deepcopy(data_process_class.target_pos)
        self.segm_deblend = np.array(data_process_class.segm_deblend)
        if sersic_major_axis is None:
            if version.parse(lenstronomy.__version__) >= version.parse("1.9.0"):
                from lenstronomy.Conf import config_loader
                convention_conf = config_loader.conventions_conf()
                self.sersic_major_axis = convention_conf['sersic_major_axis']  #If sersic_major_axis == None, the sersic_major_axis follows Lenstronomy.
        else:
            self.sersic_major_axis = sersic_major_axis

    def sepc_kwargs_data(self, supersampling_factor = 2, psf_data = None, psf_error_map = None):
        import lenstronomy.Util.simulation_util as sim_util
        kwargs_data = sim_util.data_configure_simple(self.numPix, self.deltaPix,
                                                     inverse=True)  #inverse: if True, coordinate system is ra to the left, if False, to the right
        kwargs_data['image_data'] = self.data_process_class.target_stamp
        kwargs_data['noise_map'] = self.data_process_class.noise_map
        
        if psf_data is None:
            psf_data = self.data_process_class.PSF_list[self.data_process_class.psf_id_for_fitting]
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf_data,'pixel_size': self.deltaPix}
        if psf_error_map is not None:
            kwargs_psf['psf_error_map']  = psf_error_map
        
        # here we super-sample the resolution of some of the pixels where the surface brightness profile has a high gradient 
        supersampled_indexes = np.zeros((self.numPix, self.numPix), dtype=bool)
        supersampled_indexes[int(self.numPix/2)-int(self.numPix/10):int(self.numPix/2)+int(self.numPix/10), 
                             int(self.numPix/2)-int(self.numPix/10):int(self.numPix/2)+int(self.numPix/10)] = True
        kwargs_numerics = {'supersampling_factor': supersampling_factor, 
                           'compute_mode': 'adaptive',
                           'supersampled_indexes': supersampled_indexes}
        
        # kwargs_numerics = {'supersampling_factor': supersampling_factor} 
        image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
        multi_band_list = [image_band]
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.kwargs_numerics = kwargs_numerics
        self.kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'single-band', 'multi-linear', 'joint-linear'
    
    def sepc_kwargs_model(self, extend_source_model = ['SERSIC_ELLIPSE'] * 1, point_source_num = 1):
        point_source_list = ['UNLENSED'] * point_source_num
        kwargs_model = {'point_source_model_list': point_source_list}
        if extend_source_model != None and extend_source_model != []:
            light_model_list = extend_source_model
            kwargs_model['lens_light_model_list'] = light_model_list
        else:
            light_model_list = []
        self.point_source_list = point_source_list
        self.light_model_list = light_model_list
        kwargs_model['sersic_major_axis'] = self.sersic_major_axis
        self.kwargs_model = kwargs_model
        
    def sepc_kwargs_constraints(self, fix_center_list = None):
        """
        Prepare the 'kwargs_constraints' for the fitting.
        
        Parameter
        --------
            fix_center_list: list.
                -if not None, describe how to fix the center [[0,0]] for example.
                This list defines how to 'joint_lens_light_with_point_source' definied by lenstronomy:
                    [[i_point_source, k_lens_light], [...], ...], see 
                    https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/Sampling/parameters.html?highlight=joint_lens_light_with_point_source#
                    for example [[0, 1]], joint first (0) point source with the second extend source (1).
        """
        kwargs_constraints = {'num_point_source_list': [1] * len(self.point_source_list)  #kwargs_constraints also generated here
                              }
        if fix_center_list is not None:
            kwargs_constraints['joint_lens_light_with_point_source'] =  fix_center_list
        
        self.kwargs_constraints = kwargs_constraints 
       
    def sepc_kwargs_likelihood(self, condition=None):
        """
        Prepare the 'kwargs_likelihood' for the fitting.
        
        Most default values will be assigned. 
        
        Parameter
        --------
            condition: input as a defination.
                Set up extra prior. For example if one want the first component have lower
                Sersic index, it can be set by first define a condition:
    
                    def condition_def(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special, kwargs_extinction):
                        logL = 0
                        cond_0 = (kwargs_source[0]['n_sersic'] > kwargs_source[1]['n_sersic'])
                        if cond_0:
                            logL -= 10**15
                        return logL
                Then assign to condition:
    
                    fit_sepc.prepare_fitting_seq(**, condition = condition_def)                
        """        
        kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                             'image_likelihood_mask_list': [self.data_process_class.target_mask],
                             'custom_logL_addition': condition
                             }
        if self.light_model_list != []:
            kwargs_likelihood['source_marg'] = False #In likelihood_module.LikelihoodModule -- whether to fully invert the covariance matrix for marginalization
            kwargs_likelihood['check_positive_flux'] = True #penalty is any component's flux is 'negative'.
        self.kwargs_likelihood = kwargs_likelihood
        
    def sepc_kwargs_params(self, source_params = None, fix_n_list = None, fix_Re_list = None, ps_params = None, ps_pix_center_list= None,
                           neighborhood_size = 4, threshold = 5, apertures_center_focus = False):
        """
        Setting up the 'kwargs_params' (i.e., the parameters) for the fitting. If 'source_params' or 'ps_params'
        are given, rather then setting as None, then, the input settings will be used.
        
        Parameter
        --------
            fix_n_list: list.
                Describe a prior if want to fix the Sersic index.
                e.g., fix_n_list= [[0,4], [1,1]], means the first (i.e., 0) fix n = 4; the second (i.e., 1) fix n = 1.
            
            fix_Re_list: list.
                Describe a prior if want to fix the Sersic effective radius.
                e.g., fix_n_list= [[0,0.4], [1,1]], means the first (i.e., 0) fix Reff value as 0.4.
            
            apertures_center_focus: bool.
                If true, the default parameters will have strong prior so that the center of the fitted Sersic will 
                be closer to the apertures.
            
        """
        kwargs_params = {}
        if self.light_model_list != []:
            if source_params is None:
                source_params = source_params_generator(frame_size = self.numPix, 
                                                        apertures = self.apertures,
                                                        deltaPix = self.deltaPix,
                                                        fix_n_list = fix_n_list,
                                                        fix_Re_list = fix_Re_list,
                                                        apertures_center_focus = apertures_center_focus)
            else:
                source_params = source_params
            kwargs_params['lens_light_model'] = source_params
            
        if ps_params is None and len(self.point_source_list) > 0:
            if ps_pix_center_list is None:
                from galight.tools.measure_tools import find_loc_max
                x, y = find_loc_max(self.data_process_class.target_stamp, neighborhood_size = neighborhood_size, threshold = threshold)  #Automaticlly find the local max as PS center.
                # if x == []:
                if len(x) < len(self.point_source_list):
                    x, y = find_loc_max(self.data_process_class.target_stamp, neighborhood_size = neighborhood_size, threshold = threshold/2)  #Automaticlly find the local max as PS center.
                    # raise ValueError("Warning: could not find the enough number of local max to match the PS numbers. Thus,\
                    #                  the ps_params must input manually or change the neighborhood_size and threshold values")
                    if len(x) < len(self.point_source_list):
                        warnings.warn("\nWarning: could not find the enough number of local max to match the PS numbers. Thus, all the initial PS set the same initial parameters.")
                        if x == []:
                            x, y = [self.numPix/2], [self.numPix/2]                
                        else:
                            x = x * len(self.point_source_list)
                            y = y * len(self.point_source_list)
                flux_ = []
                for i in range(len(x)):
                    flux_.append(self.data_process_class.target_stamp[int(x[i]), int(y[i])])
                _id = np.flipud(np.argsort(flux_))
                arr_x = np.array(x)
                arr_y = np.array(y)
                ps_x = - 1 * ((arr_x - int(self.numPix/2) ) )
                ps_y = (arr_y - int(self.numPix/2) )
                center_list = []
                flux_list = []
                for i in range(len(self.point_source_list)):
                    center_list.append([ps_x[_id[i]], ps_y[_id[i]]])
                    flux_list.append(flux_[_id[i]] * 10 )
            elif ps_pix_center_list is not None:
                if len(ps_pix_center_list) != len(self.point_source_list):
                    raise ValueError("Point source number mismatch between ps_pix_center_list and point_source_num")
                center_list = ps_pix_center_list
                for i in range(len(center_list)):
                    center_list[i][0] = -center_list[i][0] 
            ps_params = ps_params_generator(centers = center_list,
                                            deltaPix = self.deltaPix)
        else:
            ps_params = ps_params            
        kwargs_params['point_source_model'] = ps_params
            
        center_pix_pos = []
        if len(self.point_source_list) > 0:
            for i in range(len(ps_params[0])):
                x = -1 * ps_params[0][i]['ra_image'][0]/self.deltaPix
                y = ps_params[0][i]['dec_image'][0]/self.deltaPix
                center_pix_pos.append([x, y])
            center_pix_pos = np.array(center_pix_pos)
            center_pix_pos = center_pix_pos + int(self.numPix/2)
        self.center_pix_pos = center_pix_pos
        self.kwargs_params = kwargs_params
        
    def sepc_imageModel(self, sersic_major_axis):
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        data_class = ImageData(**self.kwargs_data)
        
        from lenstronomy.PointSource.point_source import PointSource
        pointSource = PointSource(point_source_type_list=self.point_source_list)
        psf_class = PSF(**self.kwargs_psf) 
        
        from lenstronomy.LightModel.light_model import LightModel
        try:
            lightModel = LightModel(light_model_list=self.light_model_list, sersic_major_axis=sersic_major_axis)  # By this setting: fit_sepc.lightModel.func_list[1]._sersic_major_axis
        except:
            lightModel = LightModel(light_model_list=self.light_model_list)
            if version.parse(lenstronomy.__version__) >= version.parse("1.9.0"):
                warnings.warn("\nWarning: The current Lenstronomy Version doesn't not allow for sersic_major_axis=True. Please update you Lenstrnomy version or change you Lenstronomy configure file.")
        if self.light_model_list is None:
            imageModel = ImageModel(data_class, psf_class, point_source_class=pointSource, kwargs_numerics=self.kwargs_numerics)  
        else:
            imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lightModel,
                                    point_source_class=pointSource, kwargs_numerics=self.kwargs_numerics)   
        self.data_class = data_class
        self.psf_class = psf_class
        self.lightModel = lightModel
        self.imageModel = imageModel
        self.pointSource = pointSource
        
    def plot_fitting_sets(self, savename = None, show_plot=True):
        """
        To make a plot show how the data will be fitted. The extend source will be shown using aperture, point source will be show as point source.
        
        Parameter
        --------
            savename: None or string. 
            -Defining the saving name.
            
            show_plot: bool.
            -Plot or not plot. Note that figure can be saved without shown.
        """
        from galight.tools.measure_tools import plot_data_apertures_point
        plot_data_apertures_point(self.kwargs_data['image_data'] * self.kwargs_likelihood['image_likelihood_mask_list'][0], 
                                  self.apertures, self.center_pix_pos, savename = savename, show_plot=show_plot)

    def prepare_fitting_seq(self, supersampling_factor = 2, psf_data = None,
                          extend_source_model = None,
                          point_source_num = 0, ps_pix_center_list = None, 
                          fix_center_list = None, source_params = None,
                          fix_n_list = None, fix_Re_list = None, ps_params = None, condition = None,
                          neighborhood_size = 4, threshold = 5, apertures_center_focus = False,
                          psf_error_map = None, mpi = False):
        """
        Key function used to prepared for the fitting. Parameters will be passed to the corresponding functions.
        """
        self.mpi = mpi
        if extend_source_model is None:
            extend_source_model = ['SERSIC_ELLIPSE'] * len(self.apertures)
        self.sepc_kwargs_data(supersampling_factor = supersampling_factor, psf_data = psf_data, psf_error_map = psf_error_map)
        self.sepc_kwargs_model(extend_source_model = extend_source_model, point_source_num = point_source_num)
        self.sepc_kwargs_constraints(fix_center_list = fix_center_list)
        self.sepc_kwargs_likelihood(condition)
        self.sepc_kwargs_params(source_params = source_params, fix_n_list = fix_n_list, fix_Re_list = fix_Re_list, 
                                ps_params = ps_params, neighborhood_size = neighborhood_size, threshold = threshold,
                                apertures_center_focus = apertures_center_focus, ps_pix_center_list = ps_pix_center_list)
        if point_source_num == 0 or point_source_num == None:
            del self.kwargs_params['point_source_model']
            del self.kwargs_constraints['num_point_source_list']
            del self.kwargs_model['point_source_model_list']
            
        self.sepc_imageModel(sersic_major_axis = self.sersic_major_axis)
        print("The settings for the fitting is done. Ready to pass to FittingProcess. \n  However, please make updates manullay if needed.")
    
    def build_fitting_seq(self):
        from lenstronomy.Workflow.fitting_sequence import FittingSequence
        self.fitting_seq = FittingSequence(self.kwargs_data_joint, self.kwargs_model, 
                                      self.kwargs_constraints, self.kwargs_likelihood, 
                                      self.kwargs_params, mpi=self.mpi)
        # return fitting_seq, self.imageModel
    
def source_params_generator(frame_size, apertures = [], deltaPix = 1, fix_n_list = None, fix_Re_list = None,
                            apertures_center_focus = False):
    """
    Quickly generate a source parameters for the fitting.
    
    Parameter
    --------
        frame_size: int.
            The frame size, to define the center of the frame
            
        apertures: 
            The apertures of the targets
        
        deltaPix: 
            The pixel size of the data
        
        fix_n_list: 
            A list to define how to fix the sersic index, default = []
            -for example: fix_n_list = [[0,1],[1,4]], fix first and disk and second as bulge.
            
        apertures_center_focus:
            If True, the prior of the Sersic postion will be most limited to the center of the aperture. 
        
    Return
    --------
        A Params list for the fitting.
    """
    import lenstronomy.Util.param_util as param_util
    fixed_source = []
    kwargs_source_init = []
    kwargs_source_sigma = []
    kwargs_lower_source = []
    kwargs_upper_source = []
    center = int(frame_size/2)
    
    for i in range(len(apertures)):
        aper = apertures[i]
        Reff = np.sqrt((aper.a**2 + aper.b**2)/2) * deltaPix
        q = aper.b/aper.a
        phi = - aper.theta # since data_configure_simple(inverse=True), aperture is anti-clock-wise, and inverse=True means lenstronomy is clock-wise
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        
        if isinstance(apertures[0].positions[0],float): 
            pos_x, pos_y = aper.positions[0], aper.positions[1]
        elif isinstance(apertures[0].positions[0],np.ndarray):
            pos_x, pos_y = aper.positions[0]
        c_x = -(pos_x - center) * deltaPix  #Lenstronomy defines x flipped, (i.e., East on the left.)
        c_y = (pos_y - center) * deltaPix
        if fix_n_list is not None:
            fix_n_list = np.array(fix_n_list)
            if i in fix_n_list[:,0]:
                fix_n_value = (fix_n_list[:,1])[fix_n_list[:,0]==i]
                if len(fix_n_value) != 1:
                    raise ValueError("fix_n are not assigned correctly - {0} component have two assigned values.".format(i))
                else:
                    fix_n_value = fix_n_value[0] #extract the fix n value from the list
                fixed_source.append({'n_sersic': fix_n_value})
                kwargs_source_init.append({'R_sersic': Reff, 'n_sersic': fix_n_value,
                                           'e1': e1, 'e2': e2, 'center_x': c_x, 'center_y': c_y})
            else:
                fixed_source.append({})  
                kwargs_source_init.append({'R_sersic': Reff, 'n_sersic': 2., 'e1': e1, 'e2': e2, 'center_x': c_x, 'center_y': c_y})
        else:
            fixed_source.append({})  
            kwargs_source_init.append({'R_sersic': Reff, 'n_sersic': 2., 'e1': e1, 'e2': e2, 'center_x': c_x, 'center_y': c_y})
       
        if fix_Re_list is not None:
            fix_Re_list = np.array(fix_Re_list)
            if i in fix_Re_list[:,0]:
                fix_Re_value = (fix_Re_list[:,1])[fix_Re_list[:,0]==i]
                if len(fix_Re_value) != 1:
                    raise ValueError("fix_Re are not assigned correctly - {0} component have two assigned values.".format(i))
                else:
                    fix_Re_value = fix_Re_value[0] #extract the fix Re value from the list
                fixed_source[-1]['R_sersic'] = fix_Re_value
                kwargs_source_init[-1]['R_sersic'] = fix_Re_value
        
        kwargs_source_sigma.append({'n_sersic': 0.3, 'R_sersic': 0.2*deltaPix, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1*deltaPix, 'center_y': 0.1*deltaPix})
        if apertures_center_focus == False:
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': deltaPix*0.05, 'n_sersic': 0.3, 'center_x': c_x-10*deltaPix, 'center_y': c_y-10*deltaPix})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': Reff*30, 'n_sersic': 9., 'center_x': c_x+10*deltaPix, 'center_y': c_y+10*deltaPix})        
        elif apertures_center_focus == True:
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': deltaPix*0.05, 'n_sersic': 0.3, 'center_x': c_x-2*deltaPix, 'center_y': c_y-2*deltaPix})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': Reff*30, 'n_sersic': 9., 'center_x': c_x+2*deltaPix, 'center_y': c_y+2*deltaPix})        
    source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    return source_params

def ps_params_generator(centers, deltaPix = 1):
    """
    Quickly generate a point source parameters for the fitting.
    """    
    fixed_ps = []
    kwargs_ps_init = []
    kwargs_ps_sigma = []
    kwargs_lower_ps = []
    kwargs_upper_ps = []    
    for i in range(len(centers)):
        center_x = centers[i][0] * deltaPix
        center_y = centers[i][1] * deltaPix
        # point_amp = flux_list[i] 
        fixed_ps.append({})
        kwargs_ps_init.append({'ra_image': [center_x], 'dec_image': [center_y]}) # , 'point_amp': [point_amp]})
        kwargs_ps_sigma.append({'ra_image': [0.5*deltaPix], 'dec_image': [0.5*deltaPix]})
        kwargs_lower_ps.append({'ra_image': [center_x-2*deltaPix], 'dec_image': [center_y-2*deltaPix] } )
        kwargs_upper_ps.append({'ra_image': [center_x+2*deltaPix], 'dec_image': [center_y+2*deltaPix] } )
    ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    return ps_params
    
