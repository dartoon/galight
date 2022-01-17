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

def shift_img(img, shift_pix, order=1):
    shift_pix = shift_pix[::-1]  #uniform the yx to xy
    from scipy.ndimage.interpolation import shift
    shifted_digit_image=shift(img, shift_pix, order = order)
    return shifted_digit_image
def rotate_image(img, rotate_pix, order =1):
    shift_pix = [-rotate_pix[0]*2, -rotate_pix[1]*2]
    shift_ = shift_img(img, shift_pix, order=order)
    rotate = np.flip(shift_)
    return rotate

class Measure_asy:
    def __init__(self, fitting_process_class, obj_id=0, interp_order=3, seg_cal_reg = 'or'):
        self.fitting_process_class = fitting_process_class
        self.interp_order = interp_order
        self.seg_cal_reg = seg_cal_reg
        self.obj_id = obj_id
        self.interp_order = interp_order
        self.img = self.fitting_process_class.fitting_specify_class.kwargs_data['image_data']
    def asy_segm(self, segm = None, mask_type = 'segm'):
        obj_id = self.obj_id
        apertures = self.fitting_process_class.fitting_specify_class.apertures
        if segm is None:
            if mask_type == 'segm':
                segm_deblend = self.fitting_process_class.fitting_specify_class.segm_deblend
            elif mask_type == 'aper': #!!!
                from galight.tools.measure_tools import mask_obj
                segm_deblend = np.zeros_like(self.img)
                for i in range(len(apertures)):
                    segm_deblend  = segm_deblend + (1-mask_obj(self.img, [apertures[i]])[0]) * (i+1)
        else:
            segm_deblend = segm
        if isinstance(segm_deblend, (np.ndarray)):
            self.segm = segm_deblend
        else:
            self.segm = segm_deblend.data
        pix_pos = np.intc(apertures[obj_id].positions)
        self.segm_id = self.segm[pix_pos[1], pix_pos[0]]
        self.ini_pix = [pix_pos[0]-len(self.img)/2., pix_pos[1]-len(self.img)/2.]
        
    def abs_res(self, rotate_pix, if_plot=False):
        cal_areas, _, punish = self.segm_to_mask(rotate_pix)
        rotate_ = rotate_image(self.img, rotate_pix, order = self.interp_order)
        res_ = self.img - rotate_  #Consider resdiual as data-model, where model is the rotation.
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
        cal_area = self.segm == segm_id
        mask = (self.segm != segm_id) * (self.segm != 0)
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
        return cal_areas, mask_areas, punish
        
    def cal_asymmetry(self, rotate_pix, bkg_asy_dens=None, obj_flux = None, if_remeasure_bkg=False, if_plot = True, if_plot_bkg = False):
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
        cal_areas, masks, _ = self.segm_to_mask(rotate_pix)
        
        if obj_flux is None:
            self.obj_flux = np.sum(self.img * cal_areas)
        else:
            self.obj_flux = obj_flux
        if bkg_asy_dens is None:
            if if_remeasure_bkg == False:
                obj_masks = cal_areas + masks
                obj_masks = obj_masks == False
                img_bkg = self.img * obj_masks
                img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix), order =1)
                rot_mask = img_bkg_!=0
                obj_masks = obj_masks * rot_mask
            elif hasattr(self.fitting_process_class.fitting_specify_class, 'data_process_class'):
                    # data_process_class = self.fitting_process_class.fitting_specify_class.data_process_class,
                    img_bkg, obj_masks = pass_bkg(data_process=self.fitting_process_class.fitting_specify_class.data_process_class, 
                                                  num_pix=np.sum(cal_areas),
                                                  rotate_pix=rotate_pix,
                                                  ini_pix = self.ini_pix)
                    img_bkg_ = rotate_image(img_bkg, np.around(rotate_pix-self.ini_pix), order =1)
            else:
                raise ValueError("data_process_class has been removed and should be re-assigned to fitting_specify_class.") 
            bkg_asy_2d = abs(img_bkg - img_bkg_) * obj_masks
            bkg_asy = np.sum(bkg_asy_2d)
            self.bkg_asy_dens = bkg_asy/np.sum(obj_masks) #The density of the background asymmetry.
        else:
            assert 0 < bkg_asy_dens < 1.0
            self.bkg_asy_dens = bkg_asy_dens
        if if_plot_bkg == True:
            print("Plot the region to estiamte the background asymmetry:")
            plt_fits(bkg_asy_2d,norm='linear')
        return asy/self.obj_flux - self.bkg_asy_dens * np.sum(cal_areas)/self.obj_flux  
    
from galight.tools.measure_tools import detect_obj, mask_obj
def pass_bkg(data_process, num_pix, rotate_pix, ini_pix):# **kwargs):
    ini_pix = np.asarray(ini_pix)
    rotate_pix = rotate_pix - ini_pix
    data_process.target_pos = data_process.target_pos + ini_pix
    for boost_list in [28, 30, 35, 40, 50, 60]:
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
    
# import pickle
# fit_run_pkl = pickle.load(open('/Users/Dartoon/Astro/Projects/my_code/package_code/galight_example/HSC_QSO.pkl','rb'))
# fit_run_pkl.fitting_specify_class.plot_fitting_sets()
# data_process = fit_run_pkl.fitting_specify_class.data_process_class

# asy_class = Measure_asy(fit_run_pkl, seg_cal_reg = 'or', obj_id=0)
# asy_class.asy_segm(mask_type='aper')
# result = asy_class.find_pos()
# print(result["x"])
# plt_fits(asy_class.img,colorbar=True)
# asy = asy_class.cal_asymmetry(rotate_pix = result["x"], if_remeasure_bkg=False ,if_plot=True, if_plot_bkg=True)
# print('asymmetry :', asy)

