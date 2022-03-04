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
                      if_plot=False):
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
    from photutils import centroid_2dg
    temp_center = np.asarray(center)
#    print temp_center.astype(int)
    radius = radius
    img_init = cutout(image=image, center=temp_center.astype(int), radius=radius)
    frm_q = int(len(img_init)/2.5)  #Aa quarter scale of the frame
    ms, mew = 30, 2.
    if kernel == 'center_bright':
        test_center =  np.asarray(np.where(img_init == img_init[frm_q:-frm_q,frm_q:-frm_q].max()))[:,0]
        center_shift = np.array((test_center- radius))[::-1]
        center_pos = (temp_center.astype(int) + np.round(center_shift))
        cutout_image = cutout(image=image, center=center_pos, radius=radius)
        plt_center = img_init[frm_q:-frm_q,frm_q:-frm_q].shape
        if if_plot==True:
            plt.plot(plt_center[0]/2-0.5, plt_center[1]/2-0.5, color='c', marker='+', ms=ms, mew=mew)  #-0.5 to shift the "+" to the center
            plt.imshow(cutout_image[frm_q:-frm_q,frm_q:-frm_q], origin='lower', norm=LogNorm())
            plt.show()
    elif kernel == 'center_gaussian':
        # test_center = frm_q + centroid_2dg(img_init[frm_q:-frm_q,frm_q:-frm_q])
        gauss_center = centroid_2dg(img_init)
        center_shift = gauss_center - radius
        center = np.asarray(center)
        center_pos = center.astype(int) + center_shift
        img_init = cutout(image=image, center=center_pos, radius=radius)
        if if_plot==True :
            fig, ax = plt.subplots(1, 1)
            plt_center = img_init[frm_q:-frm_q,frm_q:-frm_q].shape
            plt.plot(plt_center[0]/2-0.5, plt_center[1]/2-0.5, color='r', marker='+', ms=ms, mew=mew)
            plt.imshow(img_init[frm_q:-frm_q,frm_q:-frm_q], origin='lower', norm=LogNorm())
            plt.show()
        cutout_image = img_init
    else:
        raise ValueError("kernel is not defined")
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

def psf_clean(psf, nsigma=3, npixels = None, contrast=0.001, nlevels=25, if_plot=False):
    if npixels is None:
        npixels = int((len(psf)/13)**2)
    import copy
    _psf = copy.deepcopy(psf)
    from galight.tools.measure_tools import detect_obj
    _, seg = detect_obj(_psf*500, if_plot=if_plot, nsigma=nsigma, 
                        npixels = npixels, contrast=contrast, 
                        nlevels=nlevels,segm_map=True)
    seg = seg.data
    seg_idx = seg[int(len(psf)/2), int(len(psf)/2)]
    _psf[ (seg!= seg_idx) * (seg!= 0)] = np.flip(_psf)[((seg!= seg_idx) * (seg!= 0))]
    return _psf