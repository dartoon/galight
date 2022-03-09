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
def shift_img(img, shift_pix, order=1):
    shift_pix = shift_pix[::-1]  #uniform the yx to xy
    shifted_digit_image=shift(img, shift_pix, order = order)
    return shifted_digit_image
def rotate_image(img, rotate_pix, order =1):
    shift_pix = [-rotate_pix[0]*2, -rotate_pix[1]*2]
    shift_ = shift_img(img, shift_pix, order=order)
    rotate = np.flip(shift_)
    return rotate

def cal_r_petrosian(image, center, eta=0.2, mask=None, if_plot=False, x_gridspace=None, radius=None,
                    q=None, theta = None):
    from galight.tools.measure_tools import SB_profile
    if mask is None:
        mask = np.ones_like(image)
    # center = center + np.array([len(image)/2]*2)
    if radius is None:
        radius = len(image)/2*0.95
    seeding_num = np.min([int(radius*2), 100])
    # if radius > len(image)/2:
    #     radius = len(image)/2-1
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
    # try:
    #     idx = np.sum(r_SB_annu/r_SB>eta)
    #     r_p = r_grids[idx]
    # except:
    #     idx = -1
    #     r_p = r_grids[-1]
    if if_plot == True:
        minorLocator = AutoMinorLocator()
        fig, ax = plt.subplots()
        plt.plot(r_grids, r_SB, 'x-', label = 'Ave SB in radius')
        plt.plot(r_grids, r_SB_annu, 'x--', label = 'SB in annuli')
        plt.scatter(r_p, r_SB_p*eta,s=100,c = 'r', marker='o',
                    label='Ave SB times eta ({0})'.format(eta))
        # plt.scatter(r_p, r_SB[idx]*eta,s=100,c = 'r', marker='o',
        #             label='Ave SB times eta ({0})'.format(eta))
        ax.xaxis.set_minor_locator(minorLocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='r')
        plt.grid()
        ax.set_ylabel("Surface Brightness")
        ax.set_xlabel("Pixels")
        if x_gridspace == 'log':
            ax.set_xscale('log')
            plt.xlim(1.5*0.7, )   #1.5 is the start pixel radius to measure.
        plt.grid(which="minor")
        plt.legend()
        plt.show()
    return r_p
    
def pass_bkg(data_process, num_pix, rotate_pix, ini_pix):# **kwargs):
    data_process = copy.deepcopy(data_process)
    ini_pix = np.asarray(ini_pix)
    rotate_pix = rotate_pix - ini_pix
    data_process.target_pos = data_process.target_pos + ini_pix
    for boost_list in [28, 30, 35, 40, 50, 60]: #times to boost the size (by pixel)
        radius = np.sqrt(num_pix*boost_list)/2
        data_process.generate_target_materials(radius=radius)
        img = data_process.target_stamp
        apertures = detect_obj(img, nsigma=1, exp_sz=1.6, npixels = 10)
        # apertures = detect_obj(img, detect_tool = 'sep',exp_sz=2.5,err=data_process.noise_map)
        obj_mask = mask_obj(img, apertures, if_plot=False, sum_mask=True)
        image_masked = img*obj_mask
        obj_mask_ = rotate_image(obj_mask, rotate_pix,order =1)
        obj_masks = obj_mask*obj_mask_
        if np.sum(obj_masks) > num_pix*25:
            break
    return image_masked, obj_masks

class Measure_asy(object):
    def __init__(self, fitting_process_class, obj_id=0, interp_order=3, seg_cal_reg = 'or', 
                 consider_petrosian=False, extend=1., eta = 0.2):
        self.fitting_process_class = fitting_process_class
        self.interp_order = interp_order
        self.seg_cal_reg = seg_cal_reg
        self.obj_id = obj_id
        self.interp_order = interp_order
        self.img = self.fitting_process_class.fitting_specify_class.kwargs_data['image_data']
        self.consider_petrosian = consider_petrosian
        self.extend = extend
        self.eta = eta
    def asy_segm(self, segm = None, mask_type = 'segm'):
        obj_id = self.obj_id
        apertures = self.fitting_process_class.fitting_specify_class.apertures
        if segm is None:
            if mask_type == 'segm':
                segm_deblend = self.fitting_process_class.fitting_specify_class.segm_deblend
            elif mask_type == 'aper': #!!!
                segm_deblend = np.zeros_like(self.img)
                for i in range(len(apertures)):
                    apertures[obj_id].a = apertures[obj_id].a * self.extend
                    apertures[obj_id].b = apertures[obj_id].b * self.extend
                    segm_deblend  = segm_deblend + (1-mask_obj(self.img, [apertures[i]])[0]) * (i+1)
        else:
            segm_deblend = segm
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
        cal_areas, masks, punish = self.segm_to_mask(rotate_pix)
        rotate_ = rotate_image(self.img, rotate_pix, order = self.interp_order)
        res_ = self.img - rotate_  #Consider resdiual as data-model, where model is the rotation.
        # self.cal_areas, self.masks, self.punish = cal_areas, masks, punish
        if if_plot == True:
            print("Plot the minimized abs residual:")
            plt_fits(abs(res_*cal_areas),norm='log')
        if punish == False:
            return np.sum(abs(res_*cal_areas))
        else:
            return 10**6
        
    def find_pos(self):
        ini_pix = self.ini_pix
        print('Measuring the position for minimized asy...')
        result = op.minimize(self.abs_res, ini_pix, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True})
        return result
    
    def segm_to_mask(self, rotate_pix, segm_id = None):
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
            if np.sum(cal_areas) < np.sum(cal_area)/3:
                punish = True
        elif self.seg_cal_reg == 'or':
            cal_areas = cal_area + cal_area_
            mask_areas = mask + mask_
            cal_areas = cal_areas*(1-mask_areas)
        return cal_areas, mask_areas, punish

    def run_bkg(self, rotate_pix, img_bkg=None, if_remeasure_bkg=False):
        self.cal_areas, self.masks, _ = self.segm_to_mask(rotate_pix)
        if if_remeasure_bkg == False:
            obj_masks = self.cal_areas + self.masks
            obj_masks = obj_masks == False
            if img_bkg is not None:
                self.img_bkg = img_bkg
            else:
                img_bkg = self.img * obj_masks
            img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix), order =1)
            rot_mask = img_bkg_!=0
            obj_masks = obj_masks * rot_mask
        elif hasattr(self.fitting_process_class.fitting_specify_class, 'data_process_class'):
                # data_process_class = self.fitting_process_class.fitting_specify_class.data_process_class,
                img_bkg, obj_masks = pass_bkg(data_process=self.fitting_process_class.fitting_specify_class.data_process_class, 
                                              num_pix=np.sum(self.cal_areas),
                                              rotate_pix=rotate_pix,
                                              ini_pix = self.ini_pix)
                img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix-self.ini_pix), order =1)
        else:
            raise ValueError("data_process_class has been removed and should be re-assigned to fitting_specify_class.") 
        self.img_bkg = img_bkg
        self.img_bkg_ = img_bkg_
        self.obj_masks = obj_masks
    
    def cal_asymmetry(self, rotate_pix, obj_flux = None,
                      if_plot = True, if_plot_bkg = False):
        '''
        Parameters
        ----------
        rotate_pix : array
            center of rotation.
        bkg_asy_dens : float between 0 and 1, optional
            bkg asymmetry per pixel, if given, use this value directly. The default is None.
        if_remeasure_bkg : boolean, optional
            if True, use a larger area up to 25 * obj pixels to calculate the bkg asymmetry. The default is False.
        if_plot : boolean, optional
            Plot the minimized abs residual. The default is True.
        if_plot_bkg : boolean, optional
            Plot the region to estiamte the background asymmetry. The default is False.

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
        
        bkg_asy_2d = abs(self.img_bkg - self.img_bkg_) * self.obj_masks
        bkg_asy = np.sum(bkg_asy_2d)
        self.bkg_asy_dens = bkg_asy/np.sum(self.obj_masks) #The density of the background asymmetry.
        # else:
        #     assert 0 < bkg_asy_dens < 1.0
        #     self.bkg_asy_dens = bkg_asy_dens
            
        if if_plot_bkg == True:
            print("Plot the region to estiamte the background asymmetry:")
            plt_fits(bkg_asy_2d,norm='linear')
        return asy/self.obj_flux - self.bkg_asy_dens * np.sum(cal_areas)/self.obj_flux  
    

#    #%%     
# import pickle
# fit_run_pkl = pickle.load(open('./HSC_QSO.pkl','rb'))
# fit_run_pkl.fitting_specify_class.plot_fitting_sets()
# data_process = fit_run_pkl.fitting_specify_class.data_process_class

# asy_class = Measure_asy(fit_run_pkl, seg_cal_reg = 'or', obj_id=0)
# asy_class.asy_segm(mask_type='aper')
# result = asy_class.find_pos()
# print(result["x"])
# plt_fits(asy_class.img,colorbar=True)
# asy_class.run_bkg(rotate_pix = result["x"], if_remeasure_bkg=False ,)
# asy = asy_class.cal_asymmetry(rotate_pix = result["x"], if_plot=True, if_plot_bkg=True)
# print('asymmetry :', asy)

#%%
def skysmoothness(bkg, r_p_c):
#bkg refers to background image
    if bkg.size == 0:
        skysmooth= -99.0

    # If the smoothing "boxcar" is larger than the skybox itself,
    # this just sets all values equal to the mean:
    # boxcar_size = np.max([int(0.25 * r_p_c),2])#circular petrosian radius goes here
    boxcar_size = 0.25 * r_p_c
    bkg_smooth = ndimage.uniform_filter(bkg, size=boxcar_size)
    # print('boxcar_size, r_p_c',boxcar_size, r_p_c)
    # plt_fits(bkg)
    # plt_fits(bkg_smooth)

    bkg_diff = bkg - bkg_smooth
    bkg_diff[bkg_diff < 0] = 0.0  # set negative pixels to zero

    skysmooth= np.sum(bkg_diff) / float(bkg.size)
    return skysmooth

def cal_smoothness(image, center, r_p_c, skysmooth, if_plot=False):
    """
    Calculate smoothness (a.k.a. clumpiness) as defined in eq. (11)
    from Lotz et al. (2004). Note that the original definition by
    Conselice (2003) includes an additional factor of 10.
    """


    # Exclude central region during smoothness calculation:
    r_in = 0.25 * r_p_c#circular petrosian radius goes here
    r_out = 1.5 * r_p_c#circular petrosian radius goes here
    ap = photutils.CircularAnnulus((center), r_in, r_out)

    # boxcar_size = 0.25 * r_p_c
    boxcar_size = np.max([int(0.25 * r_p_c),2])
    image_smooth = ndimage.uniform_filter(image, size=boxcar_size)
    #with image sliced
    image_diff = image - image_smooth
    # image_diff[image_diff < 0] = 0.0  # set negative pixels to zero  #!!!
    image_diff = abs(image_diff)
    if if_plot==True:
        from galight.tools.astro_tools import plt_many_fits
        plt_many_fits([image, image_smooth, image_diff], labels = ['image', 'image_smooth', 'image_diff'])
    #     plt_fits(image)
    #     plt_fits(image_smooth)
    #     plt_fits(image_diff)
    ap_flux = ap.do_photometry(image, method='exact')[0][0]
    ap_diff = ap.do_photometry(image_diff, method='exact')[0][0]
    flag = 0
    if ap_flux <= 0:
        warnings.warn('[smoothness] Nonpositive total flux.',
                      AstropyUserWarning)
        flag = 1
        S= -99.0  # invalid

    if skysmooth == -99.0:  # invalid skybox
        S = ap_diff / ap_flux
    else:
        # print('boxcar_size, r_p_c',boxcar_size, r_p_c)
        # # print(ap)
        # print(ap_diff)
        # print(ap.area)
        # print(skysmooth)
        # print(ap_flux)
        S = (ap_diff - ap.area*skysmooth) / ap_flux

    if not np.isfinite(S):
        warnings.warn('Invalid smoothness.', AstropyUserWarning)
        flag = 1
        S= -99.0  # invalid
    return S, flag

#%%
def cal_gini(image,r_p_e, theta, q, xc, yc):
    """
    Calculate the Gini coefficient as described in Lotz et al. (2004).
    """
    #mask_stamp is masking everything that is background or is not source
    #image[slice_stamp] is image sliced appropriately#
    #image = np.where(~mask_stamp, image[slice_stamp], 0.0)
    
    # ny, nx=image.shape
    # r_outer = np.sqrt(ny**2+nx**2)
    #petroellip=_rpetro_ellip_generic(imagenomask, sepa, center, sepelongation, septheta, r_outer)
    #use petrosian elliptical radius for petroellip
    
    # segmapraw=_segmap_gini(r_p_e, image, q,theta)
    segmap = _segmap_gini(image,r_p_e,q,theta, xc, yc).flatten()
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

def _segmap_gini(image, r_p_e, q,theta, xc, yc):
        """
        Create a new segmentation map (relative to the "postage stamp")
        based on the elliptical Petrosian radius.
        """
        # Smooth image
        petro_sigma = 0.2 * r_p_e #fractiongini=0.2, 
        # print('petro_sigma',petro_sigma)
        cutout_smooth = ndimage.gaussian_filter(image, petro_sigma)
        #xc = morph.xc_centroid - morph.xmin_stamp
        #yc = morph.yc_centroid - morph.ymin_stamp 
        
        # Use mean flux at the Petrosian "radius" as threshold
        a_in = r_p_e - 0.5 * 1.0
        a_out = r_p_e + 0.5 * 1.0
        b_out = a_out / q
        theta = theta
        #xc, yc is centroid 
        ellip_annulus = photutils.EllipticalAnnulus(
            (xc, yc), a_in, a_out, b_out, theta=theta)
        ellip_annulus_mean_flux = _aperture_mean_nomask(
            ellip_annulus, cutout_smooth, method='exact')

        above_threshold = cutout_smooth >= ellip_annulus_mean_flux

        # Grow regions with 8-connected neighbor "footprint"
        s = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(above_threshold, structure=s)

        # In some rare cases (e.g., Pan-STARRS J020218.5+672123_g.fits.gz),
        # this results in an empty segmap, so there is nothing to do.
        if num_features == 0:
            warnings.warn('[segmap_gini] Empty Gini segmap!',
                          AstropyUserWarning)
            #self.flag = 1
            return above_threshold

        # In other cases (e.g., object 110 from CANDELS/GOODS-S WFC/F160W),
        # the Gini segmap occupies the entire image, which is also not OK.
        if np.sum(above_threshold) == cutout_smooth.size:
            warnings.warn('[segmap_gini] Full Gini segmap!',
                          AstropyUserWarning)
            #self.flag = 1
            return above_threshold

        # If more than one region, activate the "bad measurement" flag
        # and only keep segment that contains the brightest pixel.
        if num_features > 1:
            warnings.warn('[segmap_gini] Disjoint features in Gini segmap.',
                          AstropyUserWarning)
            #self.flag = 1
            ic, jc = np.argwhere(cutout_smooth == np.max(cutout_smooth))[0]
            assert labeled_array[ic, jc] != 0
            segmap = labeled_array == labeled_array[ic, jc]
        else:
            segmap = above_threshold

        return segmap


def _aperture_mean_nomask(ap, image, **kwargs):
    """
    Calculate the mean flux of an image for a given photutils
    aperture object. Note that we do not use ``_aperture_area``
    here. Instead, we divide by the full area of the
    aperture, regardless of masked and out-of-range pixels.
    This avoids problems when the aperture is larger than the
    region of interest.
    """
    return ap.do_photometry(image, **kwargs)[0][0] / ap.area


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
    def __init__(self, fitting_process_class, obj_id=0, interp_order=3, seg_cal_reg = 'or', 
                  consider_petrosian=False, extend=1.5, eta = 0.2):
        Measure_asy.__init__(self, fitting_process_class=fitting_process_class, obj_id=obj_id, 
                              interp_order=interp_order, seg_cal_reg = seg_cal_reg, 
                              consider_petrosian=consider_petrosian, 
                              extend=extend, eta =eta)

    def cal_CAS(self, mask_type='segm', if_remeasure_bkg = False, if_plot = False, if_plot_bkg=False,
                img_bkg=None):
        self.asy_segm(mask_type=mask_type)
        self.find_pos = self.find_pos()
        self.run_bkg(rotate_pix = self.find_pos["x"], if_remeasure_bkg=if_remeasure_bkg, img_bkg = img_bkg)
        self.asy = self.cal_asymmetry(rotate_pix = self.find_pos["x"], 
                                      if_plot=if_plot, if_plot_bkg=if_plot_bkg)
        segm_id = self.segm_id
        # radius = np.max([int(np.sqrt(np.sum(self.segm==segm_id)))*2 * 1.5, int(len(self.img)/2) ])
        radius = len(self.img)/2*0.95
        center =  np.array([len(self.img)/2]*2) + self.find_pos["x"]
        self.r_p_c = cal_r_petrosian(self.img, center=center, eta=self.eta, mask= (self.segm == segm_id) +  (self.segm == 0) ,
                                radius=radius, if_plot=if_plot)
        try:
            q = self.fitting_process_class.final_result_galaxy[self.obj_id]['q']
            theta = - self.fitting_process_class.final_result_galaxy[self.obj_id]['phi_G']  #Galight and apr's theta is reversed.
            xc, yc = np.array(self.fitting_process_class.final_result_galaxy[self.obj_id]['position_xy']) + int(len(self.img)/2)
        except:
            apr = self.fitting_process_class.fitting_specify_class.apertures[self.obj_id]
            q = apr.b/apr.a
            theta = apr.theta
            xc, yc = apr.positions
        
        self.r_p_e = cal_r_petrosian(self.img, center=center, eta=self.eta, mask= (self.segm == segm_id) +  (self.segm == 0),
                                radius=radius, q=q, theta = theta, if_plot=if_plot)
        
        # guess_rms = np.std(self.img)
        # mask = (self.img>guess_rms)
        # import sep
        # bkg = sep.Background(self.img, mask=mask, bw=32, bh=32, fw=7, fh=7)
        # skysmooth = skysmoothness(bkg,self.r_p_c)
        skysmooth = skysmoothness(self.img_bkg,self.r_p_c)
        self.smoothness, self.S_flag = cal_smoothness(image= self.img * self.cal_areas, 
                                    center=center, r_p_c=self.r_p_c,skysmooth=skysmooth)

        self.concentration = self.cal_concentration(image = self.img ,#* self.cal_areas,
                                                # mask = self.cal_areas,
                                                mask = (1-self.masks),
                                                center=center, radius = radius,if_plot=if_plot)
        # print(theta, q, xc, yc)
        self.gini = cal_gini(self.img * self.cal_areas, self.r_p_e, theta, q, xc, yc)
        return self.asy, self.smoothness, self.concentration, self.gini


    def cal_concentration(self, image, mask, center, radius, tot_flux=None, if_plot = False):
        from galight.tools.measure_tools import flux_profile
        seeding_num = np.min([int(radius*2), 100])
        r_flux, r_grids, _  =  flux_profile(image, center = center, radius = radius, mask_image = mask,
                                      if_plot=if_plot, fits_plot = if_plot, grids=seeding_num )
        if tot_flux is None:
            tot_flux = r_flux[-1]
        r_80 = r_grids[r_flux/tot_flux>0.8][0]
        r_20 = r_grids[r_flux/tot_flux>0.2][0]
        C = 5.0 * np.log10(r_80/r_20)#_radius_at_fraction_of_total_cas(0.8)#_radius_at_fraction_of_total_cas(0.2)
        return C

#%%
# import pickle
# #links of file https://drive.google.com/file/d/1jE_6pZeDTHgXwmd2GW28fCRuPaQo8I61/view?usp=sharing
# fit_run_pkl = pickle.load(open('./HSC_QSO.pkl','rb'))
# CAS_class = CAS(fit_run_pkl, seg_cal_reg = 'or', obj_id=0, extend=1)
# # CAS_class.asy_segm(mask_type='aper')
# # result = CAS_class.find_pos()
# # asy = CAS_class.cal_asymmetry(rotate_pix = result["x"], if_remeasure_bkg=False ,if_plot=False, if_plot_bkg=False)
# # print(asy)
# # plt_fits(CAS_class.img,colorbar=True)
# cas = CAS_class.cal_CAS(mask_type='aper', if_plot=False)
# print(cas)