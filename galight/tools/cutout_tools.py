#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:12:15 2020

@author: Xuheng Ding
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from regions import PixCoord, CirclePixelRegion, EllipsePixelRegion
from matplotlib.colors import LogNorm
from astropy.coordinates import Angle
import copy

def pix_region(center=[49.0,49.0], radius=5, q = None, theta = None):
    """
    Creat a region file, in pixel units.
    
    Parameter
    --------
        center: 
            The center of the region, with [reg_x, reg_y].
            
        radius: 
            The radius of the region.
    Return
    --------
        A region which is ds9-like.
    """
    center= PixCoord(x=center[0],y=center[1])
    if q is not None and theta is not None:
        angle = Angle(theta/np.pi*180, 'deg')
        region = EllipsePixelRegion(center, radius*2, radius*2*q, angle=angle )  #Input are width and height, i.e., *2
    else:
        region = CirclePixelRegion(center, radius)
    #TODO: Add function and use the EllipsePixelRegion
    return region

def cutout(image, center, radius):
    """
    Cutout a stamp image from a large frame image.
    
    Parameter
    --------
        image: 
            Large frame 2D image data.
            
        center: 
            The center position to cutout.
            
        radius: 
            The cutout box size.
    Return
    --------
        A cutout image stamp, frame size in odd number.
    """
    region = pix_region(center, radius=radius)
    cut = region.to_mask(mode='exact')
    cut_image = cut.cutout(image)
    return cut_image

def cut_center_auto(image, center, radius, kernel = 'center_bright', return_center=False,
                      if_plot=False, **kwargs):
    """
    Automaticlly cutout out a image, so that the central pixel is either the "center_bright" or "center_gaussian".
    
    Parameter
    --------
        image: 
            Large frame 2D image data.
            
        center: 
            The center position to cutout.
            
        kernel: the way to define the central pixel, with choices:
            -'center_bright': Cutout at the brightest pixel as center
            -'center_gaussian': Cutout at the Gaussian center
            
        radius: 
            The cutout box size.
            
        return_center:
            If return the finally used center value.
            
        if_plot: 
            If plot the zoom in center of the cutout stamp.
    
    Warning  
    --------
    Frame size shouldn't be too larger that exceed the target's frame otherwise the Gaussion fitting and the max pixel could miss targets.
        
    Return
    --------
        A cutout image; (if return_center= True, a central pixel position used for the cutout would be returned)
    """
    try:
        from photutils.centroids import centroid_2dg
    except:
        from photutils import centroid_2dg
        import warnings
        warnings.warn("\nThe photuils are updated to 0.1.4.")
    temp_center = np.asarray(center)
#    print temp_center.astype(int)
    radius = radius
    img_cut_int = cutout(image=image, center=temp_center.astype(int), radius=radius)
    frm_q = int(len(img_cut_int)/2.5)  #A quarter scale of the frame
    ms, mew = 30, 2.
    if kernel == 'center_bright':
        test_center =  np.asarray(np.where(img_cut_int == img_cut_int[frm_q:-frm_q,frm_q:-frm_q].max()))[:,0]
        center_shift = np.array((test_center- radius))[::-1]
        center_pos = (temp_center.astype(int) + np.round(center_shift))
        cutout_image = cutout(image=image, center=center_pos, radius=radius)
    elif kernel == 'center_gaussian':
        # test_center = frm_q + centroid_2dg(img_cut_int[frm_q:-frm_q,frm_q:-frm_q])
        gauss_center = centroid_2dg(img_cut_int)
        center_shift = gauss_center - radius
        center = np.asarray(center)
        center_pos = center.astype(int) + center_shift
        cutout_image = cutout(image=image, center=center_pos, radius=radius)
    elif kernel == 'nearest_obj_center':
        from galight.tools.measure_tools import detect_obj
        apertures, segm_deblend, mask_apertures, tbl = detect_obj(img_cut_int, if_plot=False, 
                                                                  auto_sort_center = True,
                                                                  **kwargs)
        pos = apertures[0].positions
        frame_c = np.array([len(img_cut_int)/2, len(img_cut_int)/2])
        center_shift = pos - frame_c
        center_pos = center.astype(int) + np.round(center_shift)
        cutout_image = cutout(image=image, center=center_pos, radius=radius)
        # print("Check:")
        # print(cutout_image.shape, center_pos)
    else:
        raise ValueError("kernel is not defined")
    if if_plot==True :
        fig, ax = plt.subplots(1, 1)
        plt_center = cutout_image[frm_q:-frm_q,frm_q:-frm_q].shape
        plt.plot(plt_center[0]/2-0.5, plt_center[1]/2-0.5, color='r', marker='+', ms=ms, mew=mew)
        plt.imshow(cutout_image[frm_q:-frm_q,frm_q:-frm_q], origin='lower', norm=LogNorm())
        plt.show()
    if return_center==False:
        return cutout_image
    elif return_center==True:
        return cutout_image, center_pos

def exp_grid(img,nums,drc):
    """
    expand the image frame with zero, with nums, direct to expand
        img: 
            2d array_like
        num:
            number of pixels to input
        drc:
            direction. 1: -x 2: -y 3: x 4: -y (i.e.: from -x, anticlockwise)
    """
    if drc==1:
        exp_img=np.concatenate((np.zeros([len(img), nums]),img), axis=1)
    if drc==2:
        exp_img=np.concatenate((np.zeros([nums, len(img.T)]), img), axis=0)
    if drc==3:
        exp_img=np.concatenate((img,np.zeros([len(img), nums])), axis=1)
    if drc==4:
        exp_img=np.concatenate((img,np.zeros([nums, len(img.T)])), axis=0)
    return exp_img


def plot_overview(img, center_target = None,  target_label = None, c_psf_list=None, label=None, ifsave=False, filename='filename',
                  if_plot = True):
    """
    Plot the overview of the image, highlight the location of the QSO and PSFs.
    
    Parameter
    --------
        img: 
            A FOV image.
            
        center_target: 
            The central position of the pixels of the QSO.
            
        c_psf_list: 
            A list of PSF positions.
            
        label:
            Define label if want to lable this plot.
    """    
    import copy, matplotlib
    my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad('black')
    vmax = 2.2
    vmin = 1.e-2
    target_box_size = np.min(img.shape)/72
    PSF_box_size = np.min(img.shape)/109
    fig = plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    ax.imshow(img,origin='lower', cmap=my_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    if center_target is not None:
        target_reg = pix_region(center_target, radius= target_box_size)
        target_mask = target_reg.to_mask(mode='center')
        if target_label == None:
            target_label = 'target'
        ax.text(center_target[0]-2*target_box_size, center_target[1]+1.5*target_box_size, target_label,color='white', fontsize=20)
        ax.add_patch(target_mask.bbox.as_artist(facecolor='none', edgecolor='white', linewidth=2))
    name = 'PSF'
    count=0
    if c_psf_list is not None:
        for i in range(len(c_psf_list)):
            PSF_reg = pix_region(c_psf_list[i], radius= PSF_box_size)
            PSF_mask = PSF_reg.to_mask(mode='center')
            ax.add_patch(PSF_mask.bbox.as_artist(facecolor='none', edgecolor='blue', linewidth=2))
            ax.text(c_psf_list[i][0]-2*PSF_box_size, c_psf_list[i][1]+2*PSF_box_size, '{1}{0}'.format(count, name),color='white', fontsize=15)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            count += 1
#    plt.colorbar(cax)
    if not label == None:
        ax.text(len(img)*0.05, len(img)*0.8, label,color='white', fontsize=30)
    if ifsave == True:
        fig.savefig(filename+'.pdf')    
    if if_plot == True:
        plt.show()
    else:
        plt.close()

def psf_clean(psf, nsigma=3, npixels = None, contrast=0.001, nlevels=25, if_plot=False, 
              ratio_to_replace=0.03, print_string = 'clean segm', if_print_fluxratio=False,
              clean_soft=True):
    if npixels is None:
        npixels = int((len(psf)/13)**2)
    import copy
    _psf = copy.deepcopy(psf)
    from galight.tools.measure_tools import detect_obj
    apertures, seg, _, tbl = detect_obj(_psf, if_plot=if_plot, nsigma=nsigma, 
                        npixels = npixels, contrast=contrast, 
                        nlevels=nlevels)
    cl_list = clean_aperture_list(_psf, apertures, findsoft=(1-clean_soft) )
    print(cl_list)
    kron_fluxes = [float(tbl[tbl['label']==j]['kron_flux']) for j in range(len(tbl))]
    fluxes_ratios = np.array(kron_fluxes)/kron_fluxes[0]
    if if_print_fluxratio==True:
        print(fluxes_ratios[1:])
    for i in range(1,len(kron_fluxes)):
        if fluxes_ratios[i] > ratio_to_replace and i not in cl_list :
            print(print_string, i)
            _psf[seg == i+1 ] = np.flip(_psf)[seg == i+1]
    return _psf

def stack_PSF(data, psf_POS_list, psf_size = 71,  oversampling=1, maxiters=10, tool = 'photutils'):
    if tool == 'photutils':
        from astropy.table import Table
        from astropy.nddata import NDData
        from photutils.psf import extract_stars
        from photutils import EPSFBuilder 
        stars_tbl = Table()
        stars_tbl['x'] = np.array(psf_POS_list)[:,0]
        stars_tbl['y'] = np.array(psf_POS_list)[:,1]
        nddata = NDData(data=data) 
        #nddata = NDData(data=self.fov_image) 
        stars = extract_stars(nddata, stars_tbl, size=psf_size)  
        epsf_builder=EPSFBuilder(oversampling=oversampling, maxiters=maxiters,progress_bar=True,shape=psf_size)
        epsf,fitted_stars=epsf_builder(stars)        
        stack_psf = epsf.data
        return stack_psf
   
def clean_aperture_list(image, apertures, findsoft=True):
    cl_list = []
    for i in range(1,len(apertures)):
        #Recognize the aperture feature that is sysmetric.
        ap_rot = copy.deepcopy(apertures[i])
        ap_rot.positions = np.array([len(image)]*2) - apertures[i].positions
        _flux = apertures[i].do_photometry(image, method='exact')[0][0]
        _flux_rot = ap_rot.do_photometry(image, method='exact')[0][0]
        if abs(_flux/_flux_rot) < 2.5:
            cl_list.append(i)
        if findsoft == True:
            #Recognize the aperture feature are too flat.
            ap_exp = copy.deepcopy(apertures[i])
            ap_exp.a = ap_exp.a * 1.2
            ap_exp.b = ap_exp.b * 1.2
            _flux_exp = ap_exp.do_photometry(image, method='exact')[0][0]
            _flux_dens = _flux/(apertures[i].a * apertures[i].b)
            _flux_dens_ext = (_flux_exp-_flux)/(ap_exp.a*ap_exp.b - (apertures[i].a * apertures[i].b))
            if _flux_dens_ext/_flux_dens > 0.8:
                cl_list.append(i)
    return cl_list

   
def common_data_class_aperture(data_process_list, l_idx = 0,  return_idx = 0):
    """
    From a list of data_process to generate a commen best apertures.
    
    Parameter
    --------
        l_idx: The leading idx of the aperture to start with
        return_idx: The idx of the data_process in list, common aperture to be returned for the idx
        
    Return
    --------
        A list of common apertures
    """
    from galight.tools.measure_tools import mask_obj
    _data_process_list = copy.deepcopy(data_process_list)
    deltaPix_list = np.array([_data_process_list[i].deltaPix for i in range(len(_data_process_list))])
    ratio_list = deltaPix_list/deltaPix_list[0]
    apertures = copy.deepcopy(_data_process_list[l_idx].apertures)
    target_stamp = _data_process_list[l_idx].target_stamp
    for i in range(len(_data_process_list)):
        if i != l_idx:
            covers = mask_obj(target_stamp, apertures, if_plot=False, sum_mask = True)
            for j in range(len(_data_process_list[i].apertures)):
                aper = _data_process_list[i].apertures[j]
                center_shift = (aper.positions - len(_data_process_list[i].target_stamp)/2) * ratio_list[i]
                aper.positions = center_shift + len(target_stamp)/2
                aper.a =  aper.a * ratio_list[i]
                aper.b =  aper.b * ratio_list[i]
                new_cover = mask_obj(target_stamp, [aper], if_plot=False, sum_mask = True)
                if np.sum(covers - new_cover*covers) > np.sum(1-new_cover)/2 :               
                    apertures.append(aper)
    rm_list = []
    #check all the collected apertures
    for i in range(1,len(apertures)):
        other_apertures = [apertures[j] for j in range(len(apertures)) if i!=j and i not in rm_list]
        all_cover = mask_obj(target_stamp, other_apertures, if_plot=False, sum_mask = True)
        one_cover = mask_obj(target_stamp, [apertures[i]], if_plot=False, sum_mask = True)
        if np.sum(all_cover) - np.sum(all_cover*one_cover) < np.sum(1-one_cover)/1.6:
            rm_list.append(i)
    apertures = [apertures[i] for i in range(len(apertures)) if i not in rm_list]
    
    for i in range(len(apertures)):
        aper = apertures[i]
        center_shift = (aper.positions - len(target_stamp)/2) / ratio_list[return_idx]
        aper.positions = center_shift + len(_data_process_list[return_idx].target_stamp)/2
        aper.a =  aper.a / ratio_list[return_idx]
        aper.b =  aper.b / ratio_list[return_idx]
    return apertures