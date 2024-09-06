#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:10:39 2020

@author: Xuheng Ding

A class to process the data
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from galight.tools.measure_tools import measure_bkg
from galight.tools.cutout_tools import cut_center_auto, cutout
from copy import deepcopy
from matplotlib.colors import LogNorm
from galight.tools.astro_tools import plt_fits, read_pixel_scale
import photutils
import sys
from packaging import version
from galight.tools.measure_tools import search_local_max, measure_FWHM
from galight.tools.astro_tools import plt_many_fits
import lenstronomy 
from galight.tools.measure_tools import detect_obj, mask_obj
class DataProcess(object):
    """
    A class to Process the data, including the following feature:
        - automaticlly estimate and remove background light.
        - cutout the target photo stamp.
        - search all the avaiable PSF in the field.
        - creat mask for the objects.
        - measure the target surface brightness profile, PSF FWHM, background.
    Parameter
    --------
        fov_image: 2D array.
            The field of view image of the data.
        
        target_pos: list or tuple or array, length = 2.
            The position of the target.
        
        pos_type: string.
            'pixel' or 'wcs'
            Define the position of the target, i.e., if the position is in 'pixel' or 'wcs'.
            
        header: io.fits.header.Header.
            -The header information given by the fits file. 
            Note: should including the exposure time and WCS information.
        
        exptime: float / 2D array.
            The exposure time of the data in (s) a the exptime_map
        
        fov_noise_map: 2D array.
            The field of view noise map, should have same shape as the 'fov_image'.
        
        rm_bkglight: bool. 
            If 'True', the FOV background light will be modeled and removed. 
        
        if_plot: bool.
            If 'True', the plots will made during the data processing.
        
        zp: float.
            The zeropoint of the telescope. To calcualte the magnitude. If not provided will assign as 27.0
                
    """
    def __init__(self, fov_image=None, target_pos = None, pos_type = 'pixel', header=None, 
                 exptime = None, fov_noise_map = None,rm_bkglight = False, if_plot = False, 
                 zp = None, **kwargs):
        if target_pos is not None:
            if pos_type == 'pixel':
                self.target_pos = target_pos
            elif pos_type == 'wcs':
                wcs = WCS(header)
                if isinstance(target_pos[0], str) or isinstance(target_pos[1], str):
                    from astropy.coordinates import SkyCoord
                    from astropy import units as u
                    pos = SkyCoord('{0} {1}'.format(target_pos[0], target_pos[1]), unit=(u.hourangle, u.deg))
                    target_pos = pos.ra.degree, pos.dec.degree
                self.target_pos = wcs.all_world2pix([[target_pos[0], target_pos[1]]], 1)[0]
            else:
                raise ValueError("'pos_type' is should be either 'pixel' or 'wcs'.")
            self.target_pos = np.int0(self.target_pos)
        else:
            raise ValueError("'target_pos' must be assigned.")

        self.exptime = exptime
        self.if_plot = if_plot    
        self.header = header
        if header is not None:
            self.deltaPix = read_pixel_scale(header)
            if self.deltaPix == 3600.:
                print("WARNING: pixel size could not read from the header, thus the value assigend as 1! ")
                self.deltaPix = 1.
        if fov_image is not None and rm_bkglight == True:
            bkglight = measure_bkg(fov_image, if_plot=if_plot, **kwargs)
            _mask = (fov_image!=0)
            fov_image[_mask] = fov_image[_mask]-bkglight[_mask]
        self.fov_image = fov_image
        self.fov_noise_map = fov_noise_map
        self.psf_id_for_fitting = 0 #The psf id in the PSF_list that would be used in the fitting.
        if zp is None:
            print("Zeropoint value is not provided, use 27.0 to calculate magnitude.")
            self.zp = 27.0
        else:
            self.zp = zp

    def generate_target_materials(self, cut_kernel = None,  radius=None, radius_list = None,
                                  bkg_std = None, if_select_obj = False, create_mask = False, 
                                  if_plot=None, use_moments = True, show_materials=None,
                                  skip = False, **kwargs):
        """
        Prepare the fitting materials to used for the fitting, including the image cutout, noise map and masks (optional).
        More important, the apertures that used to define the fitting settings are also generated.
        
        Parameter
        --------
            cut_kernel: string or 'None'.
                The args will be input as kernel into galight.tools.cutout_tools.cut_center_auto()
            
            radius: int or float
                The radius to cutout the image data. The final framesize will be 2*radius+1
                
            cut_kernel: None or 'center_gaussian' or 'center_bright'.
                - if 'None', directly cut.
                - if 'center_gaussian', fit central as 2D Gaussian and cut at the Gaussian center.
                - if 'center_bright', cut the brightest pixel in the center
                
            bkg_std: float
                To input the background noise level.
                
            if_select_obj:
                - if 'True', only selected obj will be modelled. 
            
            create_mask: bool.
                'True' if to define a mask based on the apertures. Note that the corresponding aperture 
                will de removed automaticlly. 

            if_plot: bool.
                If 'True', the plots will made during the cut out.
                
            **kwargs:
                Arguments can also passed to detect_obj()
                        
        """
        if if_plot == None:
            if_plot = self.if_plot
        if show_materials == None:
            show_materials = if_plot 
        
        self.bkg_std = bkg_std
        
        if radius == 'nocut':
            target_stamp = self.fov_image
        else:
            if radius == None:
                if radius_list == None:
                    radius_list = [30, 35, 40, 45, 50, 60, 70]
                for rad in radius_list:
                    from galight.tools.measure_tools import fit_data_oneD_gaussian
                    _cut_data = cutout(image = self.fov_image, center = self.target_pos, radius=rad)
                    edge_data = np.concatenate([_cut_data[0,:],_cut_data[-1,:],_cut_data[:,0], _cut_data[:,-1]])
                    try:
                        gauss_mean, gauss_1sig = fit_data_oneD_gaussian(edge_data, ifplot=False)
                    except:
                        gauss_mean, gauss_1sig = np.mean(edge_data), np.std(edge_data)
                    up_limit = gauss_mean + 2 * gauss_1sig
                    percent = np.sum(edge_data>up_limit)/float(len(edge_data))
                    if percent<0.03:
                        break
                radius = rad
            elif isinstance(radius, str):
                _cut_data = cutout(image = self.fov_image, center = self.target_pos, radius=100)
                _, segm_deblend, _, tbl = detect_obj(_cut_data,  nsigma=1, npixels = 30, if_plot=False)
                # info = tbl[tbl['kron_flux'] == np.max(tbl['kron_flux'])]
                info = tbl[tbl['label'] == 0]
                size = info['semimajor_sigma'].value
                rad = int(size*float(radius))
                radius = rad
                
            if cut_kernel is not None:
                if if_plot == True:
                    print("Plot target cut out zoom in:")
                target_stamp, self.target_pos = cut_center_auto(image=self.fov_image, center= self.target_pos, 
                                                  kernel = cut_kernel, radius=radius,
                                                  return_center=True, if_plot=if_plot, **kwargs)
            else:
                target_stamp = cutout(image = self.fov_image, center = self.target_pos, radius=radius)
        self.radius = int(len(target_stamp)/2)
        
        
        if self.fov_noise_map is not None:
            if radius == 'nocut':
                self.noise_map = self.fov_noise_map
            else:
                self.noise_map = cutout(image = self.fov_noise_map, center = self.target_pos, radius=radius)
        else:
            if bkg_std == None:
                from galight.tools.measure_tools import esti_bgkstd
                cut_rad = np.min([self.radius*2, len(self.fov_image)/2])
                target_2xlarger_stamp = cutout(image=self.fov_image, center= self.target_pos, radius=cut_rad)
                self.bkg_std = esti_bgkstd(target_2xlarger_stamp, if_plot=if_plot, npixels = 20)
            _exptime = deepcopy(self.exptime)
            if _exptime is None:
                if 'EXPTIME' in self.header.keys():
                    _exptime = self.header['EXPTIME']
                else:
                    raise ValueError("No Exposure time information in the header, should input a value.")
            if isinstance(_exptime, np.ndarray):
                _exptime = cutout(image=self.exptime, center= self.target_pos, radius=self.radius)
            noise_map = np.sqrt(abs(target_stamp/_exptime) + self.bkg_std**2)
            self.noise_map = noise_map
        
        
        if skip == False:
            target_mask = np.ones_like(target_stamp)
            apertures, segm_deblend, mask_apertures, tbl = detect_obj(target_stamp, if_plot= create_mask or if_select_obj or if_plot, 
                                                      err=self.noise_map, use_moments=use_moments, **kwargs)
            self.segm_deblend = segm_deblend
            self.tbl  = tbl
            _apertures_select = []
            if if_select_obj == True:
                select_idx = str(input('Input directly the a obj idx to MODEL, use space between each id:\n'))
                if select_idx != '':
                    if sys.version_info.major > 2:
                        select_idx = [int(obj) for obj in select_idx.split(' ') if obj.isnumeric()]
                    else:
                        select_idx = [int(obj) for obj in select_idx.split(' ') if obj.isdigit()]
                    _apertures_select = [i for i in select_idx]  
                else:
                    _apertures_select = [i for i in range(len(apertures))]  
            _restof_i = []
            if create_mask == True:
                select_idx = str(input('Input directly the a obj that used to create MASK, use space between each id:\n'))
                select_idx_list = [int(s) for s in select_idx.split() if s.isdigit()]
                
                if '!' not in select_idx:
                    apertures_ = [mask_apertures[i] for i in select_idx_list]
                    _restof_i = [i for i in range(len(apertures)) if i not in select_idx_list]
                else:
                    apertures_ = [mask_apertures[i] for i in range(len(apertures)) if i not in select_idx_list]                            
                    _restof_i = [i for i in range(len(apertures)) if i in select_idx_list]      
                mask_list = mask_obj(target_stamp, apertures_, if_plot=False)
                for i in range(len(mask_list)):
                    target_mask *= mask_list[i]
            if _restof_i+_apertures_select != []:
                if _restof_i == [] or _apertures_select == []:
                    _common = _restof_i+_apertures_select
                else:
                    _common = list(set(_restof_i).intersection(_apertures_select))
                apertures = [apertures[i] for i in range(len(apertures)) if i in _common]
                mask_apertures = [mask_apertures[i] for i in range(len(apertures)) if i in _common]
            self.apertures = apertures
            self.mask_apertures = mask_apertures
            self.target_stamp = target_stamp
            self.target_mask = target_mask
            if show_materials:
                self.plot_materials()
        elif skip == True:
            self.segm_deblend = None
            self.tbl  = None
            from photutils import EllipticalAperture
            self.apertures = [EllipticalAperture([len(target_stamp)/2]*2, a=5, b=5, theta=0)]
            self.mask_apertures = []
            self.target_stamp = target_stamp
            self.target_mask = np.ones_like(target_stamp)
            
    def plot_materials(self):
        fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(14, 10))
        im1 = ax1.imshow(self.target_stamp, origin='lower', norm=LogNorm(vmax = self.target_stamp.max(), vmin = 1.e-4))
        ax1.set_title('Cutout target', fontsize=25)
        fig.colorbar(im1, ax=ax1, pad=0.01,  orientation="horizontal")
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False) 
        im2 = ax2.imshow(self.noise_map, origin='lower', norm=LogNorm())
        ax2.set_title('Noise map', fontsize=25)
        fig.colorbar(im2, ax=ax2, pad=0.01,  orientation="horizontal")
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False) 
        im3 = ax3.imshow(self.target_stamp * self.target_mask + (self.target_mask==0)*1.e6, 
                         origin='lower', norm=LogNorm(vmax = self.target_stamp.max(), vmin = 1.e-4))
        ax3.set_title('data * mask', fontsize=25)
        fig.colorbar(im3, ax=ax3, pad=0.01,  orientation="horizontal")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False) 
        plt.show() 
    
    def plot_aperture(self, figsize=(8,6)):
        from galight.tools.plot_tools import plot_data_apertures_point
        plot_data_apertures_point(self.target_stamp * self.target_mask, # + (self.kwargs_likelihood['image_likelihood_mask_list'][0]==0)*1.e6 , 
                                  self.apertures, figsize=figsize)
        
    
    def clean_aperture(self):
        from galight.tools.cutout_tools import clean_aperture_list
        cl_list = clean_aperture_list(self.target_stamp, self.apertures)
        self.apertures = [self.apertures[i] for i in range(len(self.apertures)) if i not in cl_list]
        self.mask_apertures = [self.mask_apertures[i] for i in range(len(self.mask_apertures)) if i not in cl_list]
        cl_list = list(dict.fromkeys(cl_list)) 
        cl_list.reverse()
        for i in cl_list:
            self.segm_deblend[self.segm_deblend>i]  = self.segm_deblend[self.segm_deblend>i] - 1
            self.tbl.remove_row(i)
            
    def find_PSF(self, radius = 50, PSF_pos_list = None, pos_type = 'pixel', psf_edge=120, FWHM_sort=True,
                 if_filter=False, FWHM_filer = None,flux_filter=None, user_option= False, select_all=False,
                 nearyby_obj_filter = False , kernel= 'center_bright', norm = LogNorm(), **kwargs):
        """
        Find all the available PSF candidates in the field of view.
        
        Parameter
        --------
            radius: int/float.
                The radius of the cutout frames of the PSF. PSF size = 2*radius + 1
            
            PSF_pos_list: None or list of position.
                Input a list if PSF star position has decided.
            
            pos_type: string.
                'pixel' or 'wcs'
                Define the position of the target
            
            user_option: bool.
                Only works when PSF_pos_list = None. 
                
            psf_edge: int/float.
                The PSF should be avoid at the edge by how many pixels.
                
            neighborhood_size (**kwargs): digit.
                Define the region size to filter the local minima.
            
            threshold {**kwargs}: digit.
                Define the significance (flux value) of the maximazation point. The lower, the more would be found.
        
        """
        if PSF_pos_list is None:
            init_PSF_locs_ = search_local_max(self.fov_image, radius = psf_edge,**kwargs)
            init_PSF_locs, FWHMs, fluxs, PSF_cutouts = [], [], [], []
            for i in range(len(init_PSF_locs_)):
                cut_image = cut_center_auto(self.fov_image, center = init_PSF_locs_[i],
                                            radius=radius)
                try:
                    _fwhms = measure_FWHM(cut_image , radius = int(radius/5))
                except:
                    _fwhms = 100
                if np.std(_fwhms)/np.mean(_fwhms) < 0.1 :  #Remove the deteced "PSFs" at the edge.
                    init_PSF_locs.append(init_PSF_locs_[i])
                    FWHMs.append(np.mean(_fwhms))
                    fluxs.append(np.sum(cut_image))
                    PSF_cutouts.append(cut_image)
            init_PSF_locs = np.array(init_PSF_locs)
            FWHMs = np.array(FWHMs)
            fluxs = np.array(fluxs)
            PSF_cutouts = np.array(PSF_cutouts)
            if FWHM_sort == True:
                PSF_cutouts = PSF_cutouts[FWHMs.argsort()[::-1]]
                init_PSF_locs = init_PSF_locs[FWHMs.argsort()[::-1]]
                fluxs = fluxs[FWHMs.argsort()[::-1]]
                FWHMs = FWHMs[FWHMs.argsort()[::-1]]
            
            if if_filter:
                if hasattr(self, 'target_stamp'):
                    target_flux = np.sum(self.target_stamp)
                    dis = np.sqrt( np.sum( (init_PSF_locs - self.target_pos)**2  , axis=1) )
                    if FWHM_filer is None:
                        FWHM_filer = np.median(FWHMs)*1.5
                    select_bool = (FWHMs<FWHM_filer)*(fluxs<target_flux*10)*(fluxs>target_flux/10) * (dis>5)
                else:
                    if FWHM_filer is None:
                        select_bool = (FWHMs<np.median(FWHMs)*1.5)
                    else:
                        select_bool = (FWHMs<FWHM_filer)
                    if flux_filter is not None:
                        select_bool *= (fluxs<flux_filter[1])*(fluxs>flux_filter[0])
                    
                if nearyby_obj_filter:
                    near_bools = []
                    for i in range(len(PSF_cutouts)):
                        _detect= detect_obj(PSF_cutouts[i],if_plot=False, nsigma=5)
                        tbl = _detect[3]
                        kron_fluxes = [float(tbl[tbl['label']==j]['kron_flux']) for j in range(len(tbl))]
                        fluxes_ratios = np.array(kron_fluxes)/kron_fluxes[0]
                        if len(fluxes_ratios) == 1:
                            near_bools.append(True)
                        elif np.max(fluxes_ratios[fluxes_ratios!=1]) > 0.05:
                            near_bools.append(False)
                        else:
                            near_bools.append(True)
                    select_bool = select_bool * np.array(near_bools)
                PSF_locs = init_PSF_locs[select_bool]    
                FWHMs = FWHMs[select_bool]
                fluxs = fluxs[select_bool]
                PSF_cutouts = PSF_cutouts[select_bool]
            else:
                PSF_locs = init_PSF_locs
                
            if user_option == False:
                select_idx = [np.where(FWHMs == FWHMs.min())[0][0]]
            else:
                plt_many_fits(PSF_cutouts, FWHMs, 'FWHM', norm = norm)
                if select_all is not True:
                    if sys.version_info[0] == 2:
                        select_idx = raw_input('Input directly the PSF inital id to select, use space between each id:\n (press Enter to selet all)\n')
                    elif sys.version_info[0] == 3:
                        select_idx = input('Input directly the PSF inital id to select, use space between each id:\n (press Enter to selet all)\n')
                else:
                    select_idx = ''
                if  select_idx == '' or select_idx == 'a':
                    select_idx = [i for i in range(len(PSF_cutouts))]
                else:
                    select_idx = select_idx.split(" ")       
                    if sys.version_info.major > 2:
                        select_idx = [int(select_idx[i]) for i in range(len(select_idx)) if select_idx[i].isnumeric()]
                    else:
                        select_idx = [int(select_idx[i]) for i in range(len(select_idx)) if select_idx[i].isdigit()]                    
            
            # removing invalid index
            select_idx_valid = []
            for idx in select_idx:
                try:
                    t = PSF_locs[idx]
                    select_idx_valid.append(idx)
                except TypeError as e:
                    print(f"Removing an invalid index, {idx}: {e}")
                except IndexError as e:
                    print(f"Removing an invalid index, {idx}: {e}")
            #
            self.PSF_pos_list = [PSF_locs[i] for i in select_idx_valid]
            self.PSF_FWHM_list = [FWHMs[i] for i in select_idx_valid]
            self.PSF_flux_list = [fluxs[i] for i in select_idx_valid]
        else:
            if pos_type == 'pixel':
                self.PSF_pos_list = PSF_pos_list
            elif pos_type == 'wcs':
                wcs = WCS(self.header)
                self.PSF_pos_list = [wcs.all_world2pix([[PSF_pos_list[i][0], PSF_pos_list[i][1]]], 1)[0] for i in range(len(PSF_pos_list))]
        self.PSF_list = [cut_center_auto(self.fov_image, center = self.PSF_pos_list[i],
                                          kernel = 'center_bright', radius=radius) for i in range(len(self.PSF_pos_list))]
        if hasattr(self, 'stack_PSF_done'):
            del self.stack_PSF_done

    def profiles_compare(self, **kargs):
        """
        Use galight.tools.measure_tools.profiles_compare to plot the profiles of data and PSFs (when prepared).
        """    
        from galight.tools.measure_tools import profiles_compare    
        profiles_compare([self.target_stamp] + self.PSF_list, **kargs)


    def stack_PSF(self,  oversampling=1, maxiters=10, tool = 'photutils',if_plot=False):
        if not hasattr(self, 'PSF_FWHM_list'):
            self.PSF_FWHM_list = []
            for i in range(len(self.PSF_list)):
                self.PSF_FWHM_list.append(np.mean(measure_FWHM(self.PSF_list[i])) )
        if hasattr(self, 'stack_PSF_done'):
            print("WARNING: PSF has stacked already! Let's just show the plot.")
        elif tool == 'photutils':
            from galight.tools.measure_tools import stack_PSF
            stack_PSF = stack_PSF(self.fov_image, self.PSF_pos_list, psf_size=len(self.PSF_list[0]),
                             oversampling=oversampling, maxiters=maxiters, tool = tool)
        elif tool == 'psfr':
            print("Stacking PSF using psfr ...")
            from psfr.psfr import stack_psf
            stack_PSF, center_list_psfr, mask_list = stack_psf(self.PSF_list, oversampling=1, 
                                                              saturation_limit=None, num_iteration=50, 
                                                              n_recenter=20)
            
        self.PSF_FWHM_list.append(np.mean(measure_FWHM(stack_PSF))) 
        self.PSF_list.append(stack_PSF)
        self.psf_id_for_fitting = -1
        # print('The stack PSF in the last:')
        if if_plot==True:
            labels = ['PSF{0}'.format(i) for i in range(len(self.PSF_list))]
            labels[-1] = 'stacked PSF'
            plt_many_fits(self.PSF_list[:-1]+[self.PSF_list[-1]*np.sum(self.PSF_list[0])], 
                          self.PSF_FWHM_list, 'FWHM', labels = labels)
            print('Plot residual:')
            labels = ['stacked - PSF{0}'.format(i) for i in range(len(self.PSF_list))]
            plt_many_fits([(self.PSF_list[-1]/np.sum(self.PSF_list[-1]) - 
                            self.PSF_list[i]/np.sum(self.PSF_list[i]) ) for i in range(len(self.PSF_list)-1)], 
                          labels = labels[:-1], norm = None)
        self.stack_PSF_done = True


        
    def plot_overview(self, **kargs):
        """
        Use galight.tools.cutout_tools.plot_overview to plot image overview.
        """
        from galight.tools.cutout_tools import plot_overview
        if hasattr(self, 'PSF_pos_list'):
            PSF_pos_list = self.PSF_pos_list
        else:
            PSF_pos_list = None
        plot_overview(self.fov_image, center_target= self.target_pos,
                      c_psf_list=PSF_pos_list, **kargs)
    
    def checkout(self):
        """
        Check out if everything is prepared to pass to galight.fitting_process().
        """        
        checklist = ['deltaPix', 'target_stamp', 'noise_map',  'target_mask', 'psf_id_for_fitting']
        ct = 0
        for name in checklist:
            if not hasattr(self, name):
                print('The keyword of {0} is missing.'.format(name))
                ct = ct+1
        if hasattr(self, 'PSF_list'):
            if len(self.PSF_list[self.psf_id_for_fitting]) != 0 and self.PSF_list[self.psf_id_for_fitting].shape[0] != self.PSF_list[self.psf_id_for_fitting].shape[1]:
                print("The PSF is not a box size, will cut it to a box size automatically.")
                cut = int((self.PSF_list[self.psf_id_for_fitting].shape[0] - self.PSF_list[self.psf_id_for_fitting].shape[1])/2)
                if cut>0:
                    self.PSF_list[self.psf_id_for_fitting] = self.PSF_list[self.psf_id_for_fitting][cut:-cut,:]
                elif cut<0:
                    self.PSF_list[self.psf_id_for_fitting] = self.PSF_list[self.psf_id_for_fitting][:,-cut:cut]
                self.PSF_list[self.psf_id_for_fitting] /= self.PSF_list[self.psf_id_for_fitting].sum()
                if self.PSF_list[self.psf_id_for_fitting].shape[0] != self.PSF_list[self.psf_id_for_fitting].shape[1]:
                    raise ValueError("PSF shape is not a square.")
            if version.parse(lenstronomy.__version__) >= version.parse("1.10.4"):
                print("The negative PSF values are corrected as 0 values.")
                bool_ = (self.PSF_list[self.psf_id_for_fitting]<0)
                if np.sum(bool_) >0:
                    self.PSF_list[self.psf_id_for_fitting][bool_] = 0
            
        else:
            print('The PSF has not been assigned yet. For a direct asymmetry measurement, it is OK.')
            #Manually input a mock PSF:
            PSF = np.zeros([3,3])
            PSF[1,1] = 1
            self.PSF_list = [PSF]

        if ct == 0:
            print('The data_process is ready to go to pass to FittingSpecify!')
        
        
    
