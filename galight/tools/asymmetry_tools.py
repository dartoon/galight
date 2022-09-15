#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:38:43 2022

@author: Dartoon
"""

import numpy as np
from scipy import ndimage
import scipy.optimize as op
from galight.tools.astro_tools import plt_fits
import matplotlib.pyplot as plt
import copy
from galight.tools.measure_tools import detect_obj, mask_obj
from photutils import EllipticalAperture
from scipy.ndimage.interpolation import shift
from matplotlib.ticker import AutoMinorLocator
import photutils
import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import Tophat2DKernel
from scipy.signal import convolve as scipy_convolve

def shift_img(img, shift_pix, order=1):
    shift_pix = shift_pix[::-1]  #uniform the yx to xy
    shifted_digit_image=shift(img, shift_pix, order = order)
    return shifted_digit_image
def rotate_image(img, rotate_pix, order =1):
    shift_pix = [-rotate_pix[0]*2, -rotate_pix[1]*2]
    shift_ = shift_img(img, shift_pix, order=order)
    rotate = np.flip(shift_)
    return rotate

def cal_r_petrosian(image, center, eta=0.2,radius=None, mask=None, q=None, theta = None, 
                    if_plot=False, x_gridspace=None):
    """
    Use a set of apertures to measure the petrosian radius.
    
    Parameter
    --------
        image: 2D array.
            The 2D image to measure the petrosian radius.
       
        center: list or tuple or array, length = 2.
            The center position to be used for measure the petrosian.
            
        eta: float, should be <1.
            The ratio to definion the petrosian radius.
            
        mask: 2D array, same shape as image. 
            The mask to block the nearby galaxy.
        
        radius: None or float value.
            The upper limit for plotting the aperture.
        
        q and theta: two float values.
            The q and theta values to for ellipse aperture.
    Return
    --------
        The calcualted value of petrosian radius.
    """
    from galight.tools.measure_tools import SB_profile
    if mask is None:
        mask = np.ones_like(image)
    if radius is None:
        radius = len(image)/2*0.95
    seeding_num = np.min([int(radius*2), 100])
    r_SB, r_grids  =  SB_profile(image*mask, center = center, radius = radius, q=q, theta=theta, mask_image = mask,
                                 if_plot=False, fits_plot = if_plot, if_annuli= False, grids=seeding_num )
    r_SB_annu, _  =  SB_profile(image*mask, center = center, radius = radius, q=q, theta=theta, mask_image = mask,
                                 if_plot=False, fits_plot = False, if_annuli= True, grids=seeding_num )
    
    ratio_bl_eta = r_SB_annu/r_SB<eta
    r_p_list = r_grids[ratio_bl_eta]
    r_SB_list = r_SB[ratio_bl_eta]
    try:
        r_p = r_p_list[0]
        r_SB_p = r_SB_list[0]
    except:
        r_p = r_grids[-1]
        r_SB_p = r_SB[-1]
        warnings.warn("Couldn't find the SB_annu/SB_rad below eta, and use the last annu instead...",
                      AstropyUserWarning)
    if if_plot == True:
        print("Plot the measure of petrosian radius:")
        minorLocator = AutoMinorLocator()
        fig, ax = plt.subplots()
        plt.plot(r_grids, r_SB, 'x-', label = 'Ave SB in radius')
        plt.plot(r_grids, r_SB_annu, 'x--', label = 'SB in annuli')
        plt.scatter(r_p, r_SB_p*eta,s=100,c = 'r', marker='o',
                    label='Ave SB times eta ({0})'.format(eta))
        ax.xaxis.set_minor_locator(minorLocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='r')
        plt.grid()
        ax.set_ylabel("Surface Brightness")
        ax.set_xlabel("Pixels")
        if x_gridspace == 'log':
            ax.set_xscale('log')
            plt.xlim(1.5*0.7, )   #1.5 is the starting pixel radius to measure.
        plt.grid(which="minor")
        plt.legend()
        plt.show()
    return r_p
    
def pass_bkg(data_process, num_pix, rotate_pix, ini_pix):
    """
    The function to make re-cut and expend the size of the stampe to prepare for the background asy measure.
    """
    data_process = copy.deepcopy(data_process)
    ini_pix = np.asarray(ini_pix)
    rotate_pix = rotate_pix - ini_pix
    data_process.target_pos = data_process.target_pos + ini_pix
    for boost_list in [28, 30, 35, 40, 50, 60]: #times to boost the size (by pixel)
        radius = np.sqrt(num_pix*boost_list)/2
        data_process.generate_target_materials(radius=radius)
        img = data_process.target_stamp
        _, _, mask_apertures, _ = detect_obj(img, nsigma=1, exp_sz=1.6, npixels = 10)
        obj_mask = mask_obj(img, mask_apertures, if_plot=False, sum_mask=True)
        image_masked = img*obj_mask
        obj_mask_ = rotate_image(obj_mask, rotate_pix,order =1)
        obj_masks = obj_mask*obj_mask_
        if np.sum(obj_masks) > num_pix*25:
            break
    return image_masked, obj_masks

class Measure_asy(object):
    """
    A class to measure the asymmetry value.
    Some key parameters to put into the init function.
    Parameter
    --------
        fitting_process_class: a python class.
            The 'fit_run' class by galight.fitting_process 
        
        obj_id: int.
            The aperture id to used for measure asy.
        
        interp_order: int.
            The order of the interpotation when shift the image. 
        
        seg_cal_reg: 'or' / 'and'.
            'or' to use all the segm rotation region for the asy value.
            'and' to use the common part in the rotation segm.
        
        consider_petrosian: bool.
            if True, use the pertrosian radius aperture as target segm map.
        
        rm_ps: bool.
            if True, the point source(s) in the image will be removed. 

        rm_model: bool.
            if True, the model will be removed. This is for to get A of the residual.
        
        rm_obj: bool.
            if True, the nearby objects will be removed using fitted Sersics.
        
        
    """
    def __init__(self, fitting_process_class, obj_id=0, interp_order=3, seg_cal_reg = 'or', 
                 consider_petrosian=False, eta = 0.2, rm_ps = False, rm_model=False, rm_obj=False):
        self.fitting_process_class = fitting_process_class
        self.interp_order = interp_order
        self.seg_cal_reg = seg_cal_reg
        self.obj_id = obj_id
        self.interp_order = interp_order
        self.img = copy.deepcopy(self.fitting_process_class.fitting_specify_class.kwargs_data['image_data'])
        if rm_ps == True: 
            self.img -= np.sum(self.fitting_process_class.image_ps_list, axis = 0)
        elif rm_model == True:
            self.img = self.img - np.sum(self.fitting_process_class.image_ps_list, axis = 0) - np.sum(self.fitting_process_class.image_host_list, axis = 0)
        if rm_obj == True:
            self.img = self.img - np.sum(self.fitting_process_class.image_host_list[:obj_id] + self.fitting_process_class.image_host_list[obj_id+1:], axis = 0)
        self.consider_petrosian = consider_petrosian
        self.eta = eta
    def asy_segm(self, segm = None, mask_type = 'segm', extend=1.):
        """
        To produce the segmentation map and get the segm_id of the object for the measurement.
       
        Parameter
        --------
            segm: None or a 2D array of segm.
                If None, the segm will be adopted using either the aperture or the original segm by galight's measure.
            
            mask_type: 'segm' or 'aper'. Only applied whtn segm is None.
                The type of mask to define the region to measure.
            
            extend: float or int.
                The level to expand the aperture for the segm.
        """
        self.extend = extend
        obj_id = self.obj_id
        if segm is None:
            if mask_type == 'segm':
                segm_deblend = self.fitting_process_class.fitting_specify_class.segm_deblend
                apertures = self.fitting_process_class.fitting_specify_class.apertures
            elif mask_type == 'aper': #!!!
                segm_deblend = np.zeros_like(self.img)
                apertures = self.fitting_process_class.fitting_specify_class.mask_apertures
                for i in range(len(apertures)):
                    apertures[obj_id].a = apertures[obj_id].a * self.extend
                    apertures[obj_id].b = apertures[obj_id].b * self.extend
                    segm_deblend  = segm_deblend + (1-mask_obj(self.img, [apertures[i]])[0]) * (i+1)
        else:
            segm_deblend = segm
            apertures = self.fitting_process_class.fitting_specify_class.apertures
        if isinstance(segm_deblend, (np.ndarray)):
            self.segm = segm_deblend
        else:
            self.segm = segm_deblend.data
        pix_pos = np.intc(apertures[obj_id].positions)
        self.segm_id = self.segm[pix_pos[1], pix_pos[0]]
        if self.segm_id == 0:
            self.segm_id = np.max(self.segm[pix_pos[1]-2:pix_pos[1]+2, pix_pos[0]-2:pix_pos[0]+2])
        self.ini_pix = [pix_pos[0]-len(self.img)/2., pix_pos[1]-len(self.img)/2.]
        self.apertures = apertures
        
    def abs_res(self, rotate_pix, if_plot=False):
        """
        Calculate the absolute value of the difference between image-image_180 at a given position.
        """
        cal_areas, masks, punish = self._segm_to_mask(rotate_pix)
        rotate_ = rotate_image(self.img, rotate_pix, order = self.interp_order)
        res_ = self.img - rotate_
        if if_plot == True:
            print("Plot the minimized abs residual:")
            plt_fits(abs(res_*cal_areas),norm='log')
        if punish == False:
            return np.sum(abs(res_*cal_areas))
        else:
            return 10**6
        
    def find_pos(self):
        """
        Use minimazation to find the position for asymmetry. 
        """
        ini_pix = self.ini_pix
        print('Measuring the position for minimized asy...')
        result = op.minimize(self.abs_res, ini_pix, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True})
        return result
    
    def _segm_to_mask(self, rotate_pix, segm_id = None):
        """
        function to derve the region for calculating the asy, object mask
        if rotate_pix is exceed the cal_areas, punish will return. For the make_bkg function.
        """
        if segm_id is None:
            segm_id = self.segm_id
        _segm = copy.deepcopy(self.segm)
        if self.consider_petrosian == True:
            rotate_pix = rotate_pix #+ np.array([len(self.img)/2]*2)
            radius = int(np.sqrt(np.sum(self.segm==segm_id)))*2
            rotate_pix_center = rotate_pix +  np.array([len(self.img)/2]*2)
            r_p = cal_r_petrosian(self.img, center=rotate_pix_center, eta=self.eta, mask= self.segm == segm_id,
                                  radius=radius)
            apr = EllipticalAperture(rotate_pix+np.array([len(self.img)/2]*2), r_p*self.extend, r_p*self.extend)
            petro_mask = (1-mask_obj(self.img, [apr])[0])
            _segm = self.segm*(self.segm!=segm_id) + petro_mask*(self.segm==segm_id) *segm_id
            self.r_p = r_p
            self.petro_mask = petro_mask
            self._segm = _segm

        mask = (_segm != segm_id) * (_segm != 0)
        cal_area = _segm == segm_id
        rotate_pix = np.around(rotate_pix)
        cal_area_ = rotate_image(cal_area, rotate_pix,order =1)
        mask_ = rotate_image(mask, rotate_pix,order =1)
        punish = False
        if self.seg_cal_reg == 'and':
            cal_areas = cal_area * cal_area_
            mask_areas = mask * mask_
        elif self.seg_cal_reg == 'or':
            cal_areas = cal_area + cal_area_
            mask_areas = mask + mask_
            cal_areas = cal_areas*(1-mask_areas)
        if np.sum(cal_area * cal_area_) < np.sum(cal_area)/2:  
            punish = True
        # if np.sum(cal_areas) < 10:
            # punish = True
        return cal_areas, mask_areas, punish

    def make_bkg(self, rotate_pix, if_remeasure_bkg=False):
        """
        To collect the background pixels, and create a rotation image. Get also the masks for nearby objects.

        Parameters
        ----------
        rotate_pix : list or tuple or array, length = 2.
            The position used to make bkg.
            
        if_remeasure_bkg : bool, optional
            If True, will expand the cutout size to collect more pixels (25 times the segm map) for the background.

        Returns
        -------
        None.

        """
        self.cal_areas, self.asy_masks, _ = self._segm_to_mask(rotate_pix)
        if if_remeasure_bkg == False:
            light_masks = self.cal_areas + self.asy_masks
            light_masks = light_masks == False
            img_bkg = self.img * light_masks
            img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix), order =1)
            rot_mask = img_bkg_!=0
            light_masks = light_masks * rot_mask
        elif hasattr(self.fitting_process_class.fitting_specify_class, 'data_process_class'):
                # data_process_class = self.fitting_process_class.fitting_specify_class.data_process_class,
                img_bkg, light_masks = pass_bkg(data_process=self.fitting_process_class.fitting_specify_class.data_process_class, 
                                              num_pix=np.sum(self.cal_areas),
                                              rotate_pix=rotate_pix,
                                              ini_pix = self.ini_pix)
                img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix-self.ini_pix), order =1)
        else:
            raise ValueError("data_process_class has been removed and should be re-assigned to fitting_specify_class.") 
        self.img_bkg = img_bkg
        self.img_bkg_ = img_bkg_
        self.light_masks = light_masks
    
    def _sky_asymmetry(self, if_plot_bkg = False, bkg_asy_dens=None):
        if bkg_asy_dens is None:
            bkg_asy_2d = abs(self.img_bkg - self.img_bkg_) * self.light_masks
            bkg_asy = np.sum(bkg_asy_2d)
            self.bkg_asy_dens = bkg_asy/np.sum(self.light_masks) #The density of the background asymmetry.
        else:
            assert 0 < bkg_asy_dens < 1.0
            self.bkg_asy_dens = bkg_asy_dens
        if if_plot_bkg == True:
            print("Plot the region to estiamte the background asymmetry:")
            plt_fits(bkg_asy_2d,norm=None)
        return self.bkg_asy_dens

    def cal_asymmetry(self, rotate_pix, obj_flux = None, if_plot = True, bkg_asy_dens=None, if_plot_bkg = False):
        '''
        Parameters
        ----------
        rotate_pix : array
            center of rotation.
        bkg_asy_dens : float between 0 and 1, optional
            bkg asymmetry per pixel, if given, use this value directly. The default is None.
        if_plot : boolean, optional
            Plot the minimized abs residual. The default is True.
        Returns
        -------
        float
            asymmetry value.
        '''
        asy = self.abs_res(rotate_pix, if_plot=if_plot)
        cal_areas = self.cal_areas
        
        if obj_flux is None:
            self.obj_flux = np.sum(self.img * cal_areas)
        else:
            self.obj_flux = obj_flux

        sky_asymmetry = self._sky_asymmetry(bkg_asy_dens=bkg_asy_dens,if_plot_bkg=if_plot_bkg)
        return asy/self.obj_flux - sky_asymmetry * np.sum(cal_areas)/self.obj_flux  
    

#%%

# def rff(image, varmap, residual, pflag, sexseg):
#     #obtain mask for rff calculation
#     #image is image
#     #varmap is hsc varmap
#     #sexseg is segmentation map obtained by statmorph
#     sexsegimg=photutils.SegmentationImage(sexseg)
#     #pflag is object flag number
#     s = sexsegimg.slices[sexsegimg.get_index(pflag)]
#     xmin, xmax = s[1].start, s[1].stop - 1
#     ymin, ymax = s[0].start, s[0].stop - 1
#     #dx, dy = xmax + 1 - xmin, ymax + 1 - ymin
#     #xc, yc = xmin + dx//2, ymin + dy//2
#     #ny, nx = sexseg.shape
#     rff_slice_stamp = (slice(ymin, ymax),slice(xmin, xmax))
#     mask=None
#     segmap_stamp_rff = sexsegimg.data[rff_slice_stamp]
#     rff_mask_stamp = (segmap_stamp_rff != 0) & (segmap_stamp_rff != pflag)
#     if mask is not None:
#             rff_mask_stamp |= mask[rff_slice_stamp]
#     locs_invalid = ~np.isfinite(image[rff_slice_stamp])
#     if varmap is not None:
#         locs_invalid |= ~np.isfinite(varmap[rff_slice_stamp])
#     mask_stamp_nan=locs_invalid
#     num_badpixels = -1
#     badpixels = np.zeros((image[rff_slice_stamp].shape[0], image[rff_slice_stamp][1]), dtype=np.bool8)
#     badpixels = _get_badpixels(image[rff_slice_stamp])
#     num_badpixels = np.sum(badpixels)
#     mask_stamp_badpixels  = badpixels
#     rff_mask_stamp |= mask_stamp_nan
#     rff_mask_stamp |= mask_stamp_badpixels

#     #find flux of image
#     immask4rff=np.where(~rff_mask_stamp,image[rff_slice_stamp], 0.0)
#     flux_iso=np.sum(immask4rff)

#     #find flux of residual and correct errors with variancemap
#     res4rff = np.where(~rff_mask_stamp, residual[rff_slice_stamp], 0.0)
#     var4rff = np.where(~rff_mask_stamp, varmap[rff_slice_stamp],0.0)

#     res4rffsum = np.sum(res4rff)
#     var4rffsum= np.sum(0.8*var4rff)

#     rffdenom=np.abs(res4rffsum-var4rffsum)

#     #rff
#     rffvalue = rffdenom / flux_iso

#     return rffvalue

# def _get_badpixels(image):
#         """
#         Detect outliers (bad pixels) as described in Lotz et al. (2004).

#         Notes
#         -----
#         ndi.generic_filter(image, np.std, ...) is too slow,
#         so we do a workaround using ndi.convolve.
#         """
#         # Pixel weights, excluding central pixel.
#         w = np.array([
#             [1, 1, 1],
#             [1, 0, 1],
#             [1, 1, 1]], dtype=np.float64)
#         w = w / np.sum(w)

#         # Use the fact that var(x) = <x^2> - <x>^2.
#         local_mean = convolve(image, w, boundary='extend',
#                               normalize_kernel=False)
#         local_mean2 = convolve(image**2, w, boundary='extend',
#                                normalize_kernel=False)
#         local_std = np.sqrt(local_mean2 - local_mean**2)

#         # Get "bad pixels"
#         badpixels = (np.abs(image - local_mean) >
#                      10 * local_std)

#         return badpixels


#%%
class CAS(Measure_asy):
    """
    Inherit Measure_asy and calculate the other CAS parameters including smoothness, concentration and gini.
    """
    def __init__(self, fitting_process_class, **kwargs):
        Measure_asy.__init__(self, fitting_process_class=fitting_process_class, **kwargs)

    def cal_CAS(self, mask_type='segm', if_remeasure_bkg = False, bkg_asy_dens=None, skysmooth=None, extend=1, 
                if_plot = False, if_plot_bkg=False, segm = None, if_residual=False, image_org=None, radius=None):
        self.asy_segm(mask_type=mask_type, segm=segm, extend=extend)
        self.find_pos_res = self.find_pos()
        self.make_bkg(rotate_pix = self.find_pos_res["x"], if_remeasure_bkg=if_remeasure_bkg)
        obj_flux = None
        if if_residual == True and image_org is not None:
            obj_flux = np.sum(image_org[self.cal_areas == True])
        self.asy = self.cal_asymmetry(rotate_pix = self.find_pos_res["x"], bkg_asy_dens=bkg_asy_dens,
                                      if_plot=if_plot, if_plot_bkg=if_plot_bkg, obj_flux = obj_flux)
        segm_id = self.segm_id
        if radius == None:
            radius = len(self.img)/2*0.95
        center =  np.array([len(self.img)/2]*2) + self.find_pos_res["x"]
        self._mask = (self.segm == segm_id) +  (self.segm == 0)  #A mask for the object.
        
        try:
            q = self.fitting_process_class.final_result_galaxy[self.obj_id]['q']
            theta = - self.fitting_process_class.final_result_galaxy[self.obj_id]['phi_G']  #Galight and apr's theta is reversed.
            xc, yc = np.array(self.fitting_process_class.final_result_galaxy[self.obj_id]['position_xy']) + int(len(self.img)/2)
        except:
            apr = self.fitting_process_class.fitting_specify_class.apertures[self.obj_id]
            q = apr.b/apr.a
            theta = apr.theta
            xc, yc = apr.positions
        if if_residual == False:
            self.r_p_c = cal_r_petrosian(self.img, center=center, eta=self.eta, mask= self._mask ,
                                    radius=radius, if_plot=if_plot)
            self.r_p_e = cal_r_petrosian(self.img, center=center, eta=self.eta, mask= self._mask,
                                    radius=radius, q=q, theta = theta, if_plot=False)
        if if_residual == True:
            if image_org is None:
                image_org = copy.deepcopy(self.fitting_process_class.fitting_specify_class.kwargs_data['image_data'])
            self.image_org = image_org
            self.r_p_c = cal_r_petrosian(image_org, center=center, eta=self.eta, mask= self._mask ,
                                         radius=radius, if_plot=if_plot) 
            
            self.r_p_e = cal_r_petrosian(image_org, center=center, eta=self.eta, mask= self._mask,
                                    radius=radius, q=q, theta = theta, if_plot=False)
        
        
        skysmooth = self._skysmoothness(bkg=self.img_bkg,r_p_c=self.r_p_c,skysmooth=skysmooth)
        self.smoothness, self.S_flag = self.cal_smoothness(image= self.img,
                                    center=center, r_p_c=self.r_p_c,skysmooth=skysmooth, image_org= image_org,
                                    if_residual=if_residual)

        self.concentration = self.cal_concentration(image = self.img ,
                                                mask = self._mask,
                                                center=center, radius = radius, if_plot=if_plot)
        self.gini = self.cal_gini(self.img * self.cal_areas, self.r_p_e, theta, q, xc, yc)
        return self.asy, self.smoothness, self.concentration, self.gini


    def cal_concentration(self, image, mask, center, radius, tot_flux=None, if_plot = False):
        from galight.tools.measure_tools import flux_profile
        seeding_num = np.min([int(radius*2), 100])
        r_flux, r_grids, _  =  flux_profile(image, center = center, radius = radius, mask_image = mask, 
                                            x_gridspace = 'log', start_p=1,
                                            if_plot=False, fits_plot = if_plot, grids=seeding_num )
        if tot_flux is None:
            tot_flux = r_flux[-1]
        r_80 = r_grids[r_flux/tot_flux>0.8][0]
        r_20 = r_grids[r_flux/tot_flux>0.2][0]
        if if_plot == True:
            minorLocator = AutoMinorLocator()
            fig, ax = plt.subplots()
            plt.plot(r_grids, r_flux, 'x-', label = 'Flux')
            plt.scatter(r_80, r_flux[r_grids==r_80],s=50,c = 'r', marker='o',
                        label='r80')
            plt.scatter(r_20, r_flux[r_grids==r_20],s=50,c = 'b', marker='o',
                        label='r20')
            ax.xaxis.set_minor_locator(minorLocator)
            plt.tick_params(which='both', width=2)
            plt.tick_params(which='major', length=7)
            plt.tick_params(which='minor', length=4, color='r')
            plt.grid()
            ax.set_ylabel("Fluxs inside")
            ax.set_xlabel("Pixels")
            plt.grid(which="minor")
            plt.legend()
            plt.show()
        C = 5.0 * np.log10(r_80/r_20)
        return C
    
    def _skysmoothness(self, bkg, r_p_c, skysmooth):
        if skysmooth is None:
            if bkg.size == 0:
                skysmooth= -99.0
            # boxcar_size = 0.25 * r_p_c
            kernel = Tophat2DKernel(0.25 * r_p_c)
            bkg_smooth = scipy_convolve(bkg, kernel, mode='same', method='direct')
            bkg_diff = bkg - bkg_smooth
            bkg_diff_abs = abs(bkg_diff)
            skysmooth_abs= np.sum(bkg_diff_abs) / float(bkg.size)
            bkg_diff_pos = copy.deepcopy(bkg_diff)
            bkg_diff_pos[bkg_diff_pos < 0] = 0.0
            skysmooth_pos= np.sum(bkg_diff_pos) / float(bkg.size)
            self.skysmooth  = [skysmooth_abs, skysmooth_pos]
        else:
            self.skysmooth = skysmooth
        return self.skysmooth
    
    def cal_smoothness(self, image, center, r_p_c, skysmooth, if_plot=False, if_residual = False, image_org=None):
        """
        Calculate smoothness (a.k.a. clumpiness) as defined in eq. (11)
        from Lotz et al. (2004). Note that the original definition by
        Conselice (2003) includes an additional factor of 10.
        
        Return value inlcuding abs and positive abs value. positive means the
        smoothness is only account the positive fluxes.
        """
        r_in = 0.25 * r_p_c
        r_out = 1.5 * r_p_c
        ap = photutils.CircularAnnulus((center), r_in, r_out)
    
        kernel = Tophat2DKernel(0.25 * r_p_c)
        image_smooth = scipy_convolve(image, kernel, mode='same', method='direct')
        image_diff = image - image_smooth
        image_diff_abs = abs(image_diff)
        image_diff_pos = copy.deepcopy(image_diff)
        image_diff_pos[image_diff_pos < 0] = 0.0
        if if_plot==True:
            from galight.tools.astro_tools import plt_many_fits
            plt_many_fits([image, image_smooth, image_diff_abs], labels = ['image', 'image_smooth', 'image_diff'])
        if if_residual == False:
            ap_flux = ap.do_photometry(image, method='exact')[0][0]
        elif if_residual == True:
            ap_flux = ap.do_photometry(image_org * self.cal_areas, method='exact')[0][0]
            
        ap_diff_abs = ap.do_photometry(image_diff_abs, method='exact')[0][0]
        ap_diff_pos = ap.do_photometry(image_diff_pos, method='exact')[0][0]
        flag = 0
        if ap_flux <= 0:
            warnings.warn('[smoothness] Nonpositive total flux.',
                          AstropyUserWarning)
            flag = 1
            S_abs, S_pos = -99.0, -99.0
    
        if skysmooth[0] == -99.0 or skysmooth[1] == -99.0:  # invalid skybox
            S_abs = ap_diff_abs / ap_flux
            S_pos = ap_diff_pos / ap_flux
        else:
            S_abs = (ap_diff_abs - ap.area*skysmooth[0]) / ap_flux
            S_pos = (ap_diff_pos - ap.area*skysmooth[1]) / ap_flux
    
        if not np.isfinite(S_abs) and not np.isfinite(S_abs):
            warnings.warn('Invalid smoothness.', AstropyUserWarning)
            flag = 1
            S_abs, S_pos = -99.0, -99.0
        return (S_abs, S_pos), flag
    
    def cal_gini(self, image,r_p_e, theta, q, xc, yc):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        """
        segmap = self. _segmap_gini(image,r_p_e,q,theta, xc, yc).flatten()
        imagenomaskforgini=image.flatten()
        
        sorted_pixelvals = np.sort(np.abs(imagenomaskforgini[segmap]))
        n = len(sorted_pixelvals)
        if n <= 1 or np.sum(sorted_pixelvals) == 0:
            warnings.warn('[gini] Not enough data for Gini calculation.',
                          AstropyUserWarning)
            return -99.0  # invalid
        
        indices = np.arange(1, n+1)  # start at i=1
        gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
                (float(n-1) * np.sum(sorted_pixelvals)))
        return gini
    
    def _segmap_gini(self, image, r_p_e, q,theta, xc, yc):
        """
        Create a new segmentation map (relative to the "postage stamp")
        based on the elliptical Petrosian radius.
        """
        petro_sigma = 0.2 * r_p_e #fractiongini=0.2, 
        cutout_smooth = ndimage.gaussian_filter(image, petro_sigma)
        a_in = r_p_e - 0.5 * 1.0
        a_out = r_p_e + 0.5 * 1.0
        b_out = a_out / q
        theta = theta
        ellip_annulus = photutils.EllipticalAnnulus(
            (xc, yc), a_in, a_out, b_out, theta=theta)
        ellip_annulus_mean_flux = self._aperture_mean_nomask(
            ellip_annulus, cutout_smooth, method='exact')
        above_threshold = cutout_smooth >= ellip_annulus_mean_flux
        s = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(above_threshold, structure=s)
        if num_features == 0:
            warnings.warn('[segmap_gini] Empty Gini segmap!',
                          AstropyUserWarning)
            return above_threshold
        if np.sum(above_threshold) == cutout_smooth.size:
            warnings.warn('[segmap_gini] Full Gini segmap!',
                          AstropyUserWarning)
            return above_threshold
        if num_features > 1:
            warnings.warn('[segmap_gini] Disjoint features in Gini segmap.',
                          AstropyUserWarning)
            ic, jc = np.argwhere(cutout_smooth == np.max(cutout_smooth))[0]
            assert labeled_array[ic, jc] != 0
            segmap = labeled_array == labeled_array[ic, jc]
        else:
            segmap = above_threshold
    
        return segmap
    
    def _aperture_mean_nomask(self, ap, image, **kwargs):
        """
        Calculate the mean flux of an image for a given photutils
        aperture object. Note that we do not use ``_aperture_area``
        here. Instead, we divide by the full area of the
        aperture, regardless of masked and out-of-range pixels.
        This avoids problems when the aperture is larger than the
        region of interest.
        """
        return ap.do_photometry(image, **kwargs)[0][0] / ap.area

    def cal_rff(self):
        noise_map = self.fitting_process_class.fitting_specify_class.kwargs_data['noise_map']
        res = self.fitting_process_class.fitting_specify_class.kwargs_data['image_data'] - np.sum(self.fitting_process_class.image_ps_list, axis = 0) - np.sum(self.fitting_process_class.image_host_list, axis = 0)
        #mask non-source and background
        rffmask=(self.segm!=0)*(self.segm!=1)
        #sum image flux
        rffdenom=np.nansum(self.img*(self.segm==1)*(1-rffmask)*(1-np.isinf(noise_map))) 
        #sum residual flux
        resflux = np.nansum(np.abs(res*(self.segm==1)*(1-rffmask)*(1-np.isinf(noise_map))))
        #sum noise map flux
        varflux = 0.8*np.nansum(noise_map*(self.segm==1)*(1-rffmask)*(1-np.isinf(noise_map)))
        rff=(resflux - varflux) /rffdenom
        # print(rff)
        return rff
        
    
    def cal_M20(self):
        """
        Calculate the M_20 coefficient as described in Lotz et al. (2004).
        """
        if np.sum(self._segmap_gini) == 0:
            return -99.0  # invalid

        # # Use the same region as in the Gini calculation
        image = np.where(self._segmap_gini, self.img*self._mask, 0.0)  #!!! This is needed to make sure.
        image = np.float64(image)  # skimage wants double
        # image = self.img
        import skimage.measure
        # Calculate centroid
        M = skimage.measure.moments(image, order=1)
        if M[0, 0] <= 0:
            warnings.warn('[deviation] Nonpositive flux within Gini segmap.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid
        yc = M[1, 0] / M[0, 0]
        xc = M[0, 1] / M[0, 0]

        # Calculate second total central moment
        Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
        second_moment_tot = Mc[0, 2] + Mc[2, 0]

        # Calculate threshold pixel value
        sorted_pixelvals = np.sort(image.flatten())
        flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
        sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
        if len(sorted_pixelvals_20) == 0:
            # This can happen when there are very few pixels.
            warnings.warn('[m20] Not enough data for M20 calculation.',
                          AstropyUserWarning)
            self.flag = 1
            return -99.0  # invalid
        threshold = sorted_pixelvals_20[0]

        # Calculate second moment of the brightest pixels
        image_20 = np.where(image >= threshold, image, 0.0)
        Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
        second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

        if (second_moment_20 <= 0) | (second_moment_tot <= 0):
            warnings.warn('[m20] Negative second moment(s).',
                          AstropyUserWarning)
            self.flag = 1
            m20 = -99.0  # invalid
        else:
            m20 = np.log10(second_moment_20 / second_moment_tot)

        return m20


        
        
