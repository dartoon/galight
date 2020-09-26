#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:16:43 2020

@author: Xuheng Ding
"""

import numpy as np

class FittingSpeficy(object):
    """
    A class to generate the direct materials for the 'FittingSequence', including:
        - kwargs_data_joint: data materils
        - kwargs_model: a list of class 
        - kwargs_constraints
        - kwargs_likelihood
        - kwargs_params
        - imageModel: 
    """    
    def __init__(self, data_process_class):
        self.data_process_class = data_process_class
        self.deltaPix = data_process_class.deltaPix
        self.numPix = len(self.data_process_class.target_stamp)
        self.zp = data_process_class.zp
    
    def sepc_kwargs_data(self, supersampling_factor = 2, psf_data = None):
        import lenstronomy.Util.simulation_util as sim_util
        kwargs_data = sim_util.data_configure_simple(self.numPix, self.deltaPix,
                                                     inverse=True)
        kwargs_data['image_data'] = self.data_process_class.target_stamp
        kwargs_data['noise_map'] = self.data_process_class.noise_map
        
        if psf_data is None:
            psf_data = self.data_process_class.PSF_list[self.data_process_class.psf_id_4_fitting]
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf_data}
        kwargs_numerics = {'supersampling_factor': supersampling_factor, 'supersampling_convolution': False} 
        image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
        multi_band_list = [image_band]
        self.kwargs_data = kwargs_data
        self.kwargs_psf = kwargs_psf
        self.kwargs_numerics = kwargs_numerics
        self.kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'single-band', 'multi-linear', 'joint-linear'
    
    def sepc_kwargs_model(self, extend_source_model = ['SERSIC_ELLIPSE'] * 1, point_source_num = 1):
        point_source_list = ['UNLENSED'] * point_source_num
        kwargs_model = { 'point_source_model_list': point_source_list
                }
        if extend_source_model is not None:
            light_model_list = extend_source_model
            kwargs_model['source_light_model_list'] = light_model_list
        else:
            light_model_list = []
        self.point_source_list = point_source_list
        self.light_model_list = light_model_list
        self.kwargs_model = kwargs_model
        
    def sepc_kwargs_constraints(self, fix_center_list = None):
        """
        generate the kwargs_constraints for the fitting.
        
        Parameter
        --------
            fix_center_list: list 
            if not None, describe how to fix the center [[0,0]] for example.
            
            Define how to 'joint_source_with_point_source':
                for example [[0, 1]], joint first extend source with second ps.
        """
        kwargs_constraints = {'num_point_source_list': [1] * len(self.point_source_list)  #kwargs_constraints also generated here
                              }
        if fix_center_list is not None:
            kwargs_constraints['joint_source_with_point_source'] =  fix_center_list
        
        self.kwargs_constraints = kwargs_constraints 
       
    def sepc_kwargs_likelihood(self):
        kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                     'image_likelihood_mask_list': [self.data_process_class.target_mask]
             }
        if self.light_model_list != []:
            kwargs_likelihood['source_marg'] = False #In likelihood_module.LikelihoodModule -- whether to fully invert the covariance matrix for marginalization
            kwargs_likelihood['check_positive_flux'] = True #penalty is any component's flux is 'negative'.
        self.kwargs_likelihood = kwargs_likelihood
        
    def sepc_kwargs_params(self, source_params = None, fix_n_list = None, ps_params = None, neighborhood_size = 4, threshold = 5):
        kwargs_params = {}
        if self.light_model_list != []:
            if source_params is None:
                source_params = source_params_generator(frame_size = self.numPix, 
                                                        apertures = self.data_process_class.apertures,
                                                        deltaPix = self.deltaPix,
                                                        fix_n_list = fix_n_list)
            else:
                source_params = source_params
            kwargs_params['source_model'] = source_params
            
        if ps_params is None:
            from decomprofile.tools.measure_tools import find_loc_max
            x, y = find_loc_max(self.data_process_class.target_stamp, neighborhood_size = neighborhood_size, threshold = threshold)  #Automaticlly find the local max as PS center.
            if len(x) < len(self.point_source_list):
                raise ValueError("Warning: could not find the enough number of local max to match the PS numbers. Thus,\
                                 the ps_params must input manually or change the neighborhood_size and threshold values")
            flux_ = []
            for i in range(len(x)):
                flux_.append(self.data_process_class.target_stamp[int(x[i]), int(y[i])])
            _id = np.flipud(np.argsort(flux_))
            print("_id", _id)
            arr_x = np.array(x)
            arr_y = np.array(y)
            ps_x = -1 * ((arr_x - self.numPix/2) * self.deltaPix)
            ps_y = (arr_y - self.numPix/2) * self.deltaPix
            center_list = []
            flux_list = []
            for i in range(len(self.point_source_list)):
                center_list.append([ps_x[_id[i]], ps_y[_id[i]]])
                flux_list.append(flux_[_id[i]] * 10 )
            ps_params = ps_params_generator(centers = center_list,
                                            flux_list = flux_list,
                                            deltaPix = self.deltaPix)
        else:
            ps_params = ps_params            
        kwargs_params['point_source_model'] = ps_params
        
        self.kwargs_params = kwargs_params
        self.source_params = source_params
        self.ps_params = ps_params
        
    def sepc_imageModel(self):
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        
        data_class = ImageData(**self.kwargs_data)
        
        from lenstronomy.PointSource.point_source import PointSource
        pointSource = PointSource(point_source_type_list=self.point_source_list)
        psf_class = PSF(**self.kwargs_psf) 
        
        from lenstronomy.LightModel.light_model import LightModel
        lightModel = LightModel(light_model_list=self.light_model_list)
        if self.light_model_list is None:
            imageModel = ImageModel(data_class, psf_class, point_source_class=pointSource, kwargs_numerics=self.kwargs_numerics)  
        else:
            imageModel = ImageModel(data_class, psf_class, source_model_class=lightModel,
                                    point_source_class=pointSource, kwargs_numerics=self.kwargs_numerics)   
        self.data_class = data_class
        self.psf_class = psf_class
        self.lightModel = lightModel
        self.imageModel = imageModel
        self.pointSource = pointSource
    
    def prepare_fitting_seq(self, supersampling_factor = 2, psf_data = None,
                          extend_source_model = None,
                          point_source_num = 1, fix_center_list = None, source_params = None,
                          fix_n_list = None, ps_params = None, neighborhood_size = 4, threshold = 5):
        if extend_source_model is None:
            extend_source_model = ['SERSIC_ELLIPSE'] * len(self.data_process_class.apertures)
        self.sepc_kwargs_data(supersampling_factor = supersampling_factor, psf_data = psf_data)
        self.sepc_kwargs_model(extend_source_model = extend_source_model, point_source_num = point_source_num)
        self.sepc_kwargs_constraints(fix_center_list = fix_center_list)
        self.sepc_kwargs_likelihood()
        self.sepc_kwargs_params(source_params = None, fix_n_list = fix_n_list, ps_params = None,
                                neighborhood_size = neighborhood_size, threshold = threshold)
        self.sepc_imageModel()
        print("The settings for the fitting is done. Ready to pass to FittingProcess. \n\tHowever, please update self.settings manullay if needed.")
    
    def build_fitting_seq(self):
        from lenstronomy.Workflow.fitting_sequence import FittingSequence
        self.fitting_seq = FittingSequence(self.kwargs_data_joint, self.kwargs_model, 
                                      self.kwargs_constraints, self.kwargs_likelihood, 
                                      self.kwargs_params)
        # return fitting_seq, self.imageModel
    
def source_params_generator(frame_size, apertures = [], deltaPix = 1, fix_n_list = None):
    """
    Quickly generate a source parameters for the fitting
    
    Parameter
    --------
        frame_size: int
        The frame size, to define the center of the frame
            
        apertures: The apertures of the targets
        
        deltaPix: The pixel size of the data
        
        fix_n_list: A list to define how to fix the sersic index, default = []
        use example: fix_n_list = [[0,1],[1,4]], fix first and disk and second as bulge.
        
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
        phi = aper.theta
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
                fixed_source.append({})  # we fix the Sersic index to n=1 (exponential)
                kwargs_source_init.append({'R_sersic': Reff, 'n_sersic': 2., 'e1': e1, 'e2': e2, 'center_x': c_x, 'center_y': c_y})

        else:
            fixed_source.append({})  # we fix the Sersic index to n=1 (exponential)
            kwargs_source_init.append({'R_sersic': Reff, 'n_sersic': 2., 'e1': e1, 'e2': e2, 'center_x': c_x, 'center_y': c_y})
        kwargs_source_sigma.append({'n_sersic': 0.3, 'R_sersic': 0.5*deltaPix, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1*deltaPix, 'center_y': 0.1*deltaPix})
        kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': Reff*0.1*deltaPix, 'n_sersic': 0.3, 'center_x': c_x-10*deltaPix, 'center_y': c_y-10*deltaPix})
        kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': Reff*30*deltaPix, 'n_sersic': 9., 'center_x': c_x+10*deltaPix, 'center_y': c_y+10*deltaPix})        
    source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    return source_params

def ps_params_generator(centers, flux_list, deltaPix = 1):
    fixed_ps = []
    kwargs_ps_init = []
    kwargs_ps_sigma = []
    kwargs_lower_ps = []
    kwargs_upper_ps = []    
    for i in range(len(centers)):
        center_x = centers[i][0] * deltaPix
        center_y = centers[i][1] * deltaPix
        point_amp = flux_list[i] 
        fixed_ps.append({})
        kwargs_ps_init.append({'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]})
        kwargs_ps_sigma.append({'ra_image': [0.5*deltaPix], 'dec_image': [0.5*deltaPix]})
        kwargs_lower_ps.append({'ra_image': [center_x-2*deltaPix], 'dec_image': [center_y-2*deltaPix] } )
        kwargs_upper_ps.append({'ra_image': [center_x+2*deltaPix], 'dec_image': [center_y+2*deltaPix] } )
    ps_params = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    return ps_params
    
#TODO: Test if double PSF. i.e., dual AGN
