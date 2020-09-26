#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:38:27 2020

@author: Xuheng Ding

A group of function to measure the photometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
import copy
import matplotlib
from photutils import make_source_mask
from decomprofile.tools_data.astro_tools import plt_fits 
my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad('black')
import photutils

from packaging import version

def find_loc_max(image, neighborhood_size = 8, threshold = 5):
    """
    Find all the local maximazation in a 2D array, used to search the targets such as QSOs and PSFs.
    This function is created and inspired based on:
        https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
   
    Parameter
    --------
        image: input image (2D array)
        neighborhood_size: Define the region size to filter the local minima.
        threshold: define the significance (flux value) of the maximazation point. The lower, the more would be found.
    
    Return
    --------
        A list of x and y of the targets.
    """    
    data_max = filters.maximum_filter(image, neighborhood_size) 
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return x, y

def search_local_max(image, radius=120, view=False, **kwargs):
    """
    Search all the maxs. The edges position with a lot of zeros would be excluded.
    
    Parameter
    --------
        image: the input image (2D array)
        radius: a radius used to test if any empty pixels around.
        
    Return
    --------
        A list of positions of 'PSF'
    """
    from decomprofile.tools_data.cutout_tools import cutout
    PSFx, PSFy =find_loc_max(image, **kwargs)
    PSF_locs = []
    ct = 0
    for i in range(len(PSFx)):
        cut_img = cutout(image, [PSFx[i], PSFy[i]], radius=radius)
        cut_img[np.isnan(cut_img)] = 0
        if np.sum(cut_img==0)  > len(cut_img)**2/20:
            continue
        PSF_locs.append([PSFx[i], PSFy[i]])
        if view ==True:
            print("plot for position: [{0}, {1}]".format(PSFx[i], PSFy[i]), "idx:", ct)
            print("total flux:", cut_img.sum())
            print("measure FWHM:", np.round(measure_FWHM(cut_img),3))
            plt_fits(cut_img)
            print("================")
            ct += 1
    return  PSF_locs
    
def measure_FWHM(image, radius = 10):
    """
    Fit image as 2D gaussion to calculate the FWHM on four directions.
    
    Parameter
    --------
        image: the input image (2D array)
        radius: define the distance (2*radius) to sample the pixel value and fit Gaussian.
        
    Return
    --------
        FWHM on 4 different direcions.
    """    
    seed_num = 2*radius+1
    frm = len(image)
    q_frm = int(frm/4)
    x_center = np.where(image == image[q_frm:-q_frm,q_frm:-q_frm].max())[1][0]
    y_center = np.where(image == image[q_frm:-q_frm,q_frm:-q_frm].max())[0][0]
    
    x_n = np.asarray([image[y_center][x_center+i] for i in range(-radius, radius+1)]) # The x value, vertcial 
    y_n = np.asarray([image[y_center+i][x_center] for i in range(-radius, radius+1)]) # The y value, horizontal 
    xy_n = np.asarray([image[y_center+i][x_center+i] for i in range(-radius, radius+1)]) # The up right value, horizontal
    xy__n =  np.asarray([image[y_center-i][x_center+i] for i in range(-radius, radius+1)]) # The up right value, horizontal
    from astropy.modeling import models, fitting
    g_init = models.Gaussian1D(amplitude=y_n.max(), mean=radius, stddev=1.5)
    fit_g = fitting.LevMarLSQFitter()
    g_x = fit_g(g_init, range(seed_num), x_n)
    g_y = fit_g(g_init, range(seed_num), y_n)
    g_xy = fit_g(g_init, range(seed_num), xy_n)
    g_xy_ = fit_g(g_init, range(seed_num), xy__n)
    from astropy.stats import gaussian_fwhm_to_sigma
    FWHM_ver = g_x.stddev.value /gaussian_fwhm_to_sigma  # The FWHM = 2*np.sqrt(2*np.log(2)) * stdd = 2.355*stdd
    FWHM_hor = g_y.stddev.value /gaussian_fwhm_to_sigma
    FWHM_xy = g_xy.stddev.value /gaussian_fwhm_to_sigma * np.sqrt(2.)   #the sampling on the xy direction is expend by a factor of np.sqrt(2.).
    FWHM_xy_ = g_xy_.stddev.value /gaussian_fwhm_to_sigma * np.sqrt(2.)
    return FWHM_ver, FWHM_hor, FWHM_xy, FWHM_xy_

def flux_in_region(image,region,mode='exact'):
    '''
    Measure the total flux inside a region.
    
    Parameter
    --------
        image: 2-D array image;
        region: region generated by pix_region;
        mode: mask mode, 'exact', 'center', default is 'exact'.
        
    Returns
    --------
        Total flux
    '''
    mask = region.to_mask(mode=mode)
    data = mask.cutout(image)
    tot_flux= np.sum(mask.data * data)
    return tot_flux

from decomprofile.tools_data.cutout_tools import pix_region

def flux_profile(image, center, radius=35,start_p=1.5, grids=20, x_gridspace=None, if_plot=False,
                 fits_plot=False, mask_image=None):
    '''
    Obtain the flux profile of a 2D image, region at the center position.
    
    Parameters
    --------
        image: A 2-D array image;
        center: center point of the profile;
        radius: radius of the profile edge, default = 35;
        grids: number of points to sample the flux, default = 20;
        if_plot: if plot the profile
        fits_plot: if plot the fits file with the regions.
        mask_list: a list of reg filenames used to generate a mask.
        
    Returns
    --------
        1. A 1-D array of the tot_flux value of each 'grids' in the profile sampled radius. 
        2. The grids of each pixel radius.
        3. The region file for each radius.
    '''
    if x_gridspace == None:
        r_grids=(np.linspace(0,1,grids+1)*radius)[1:]
        diff = start_p - r_grids[0]
        r_grids += diff             #starts from pixel 0.5
    elif x_gridspace == 'log':
        r_grids=(np.logspace(-2,0,grids+1)*radius)[1:]
        diff =  start_p - r_grids[0]
        r_grids += diff             #starts from pixel 0.5
    r_flux = np.empty(grids)
    regions = []
    mask = np.ones(image.shape)
    if mask_image is not None:
        mask = mask_image * mask
    for i in range(len(r_grids)):
        region = pix_region(center, r_grids[i])
        r_flux[i] =flux_in_region(image*mask, region)
        regions.append(region)
    if fits_plot == True:
        ax=plt.subplot(1,1,1)
        cax=ax.imshow((image*mask),norm=LogNorm(),origin='lower')#,cmap='gist_heat')
        #ax.add_patch(mask.bbox.as_artist(facecolor='none', edgecolor='white'))
        for i in range(grids):
            ax.add_patch(regions[i].as_artist(facecolor='none', edgecolor='orange'))
        plt.colorbar(cax)
        plt.show()
    if if_plot == True:
        minorLocator = AutoMinorLocator()
        fig, ax = plt.subplots()
        plt.plot(r_grids, r_flux, 'x-')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='r')
        plt.grid()
        ax.set_ylabel("Total Flux")
        ax.set_xlabel("Pixels")
        if x_gridspace == 'log':
            ax.set_xscale('log')
            plt.xlim(start_p*0.7, ) 
        plt.grid(which="minor")
        plt.show()
    return r_flux, r_grids, regions

def SB_profile(image, center, radius=35, start_p=1.5, grids=20,
               x_gridspace = None, if_plot=False, fits_plot=False,
               if_annuli= False, mask_image=None):
    '''
    Derive the SB profile of one image start at the center.
    
    Parameters
    --------
        image: A 2-D array image;
        center: The center point of the profile;
        radius: The radius of the profile favourable with default equals to 35;
        grids: The number of points to sample the flux with default equals to 20;
        if_plot: if plot the profile
        fits_plot: if plot the fits file with the regions.
        if_annuli: False: The overall surface brightness with a circle. True, return annuli surface brightness between i and i-1 cirle.
        mask_image: if is not None, will use this image as mask.
    Returns
    --------
        A 1-D array of the SB value of each 'grids' in the profile with the sampled radius.
    '''
    mask = np.ones(image.shape)
    if mask_image is not None:
        mask = mask * mask_image
    r_flux, r_grids, regions=flux_profile(image*mask, center, radius=radius, start_p=start_p, grids=grids,
                                          x_gridspace=x_gridspace, if_plot=False, fits_plot=False)
    region_size = np.zeros([len(r_flux)])
    for i in range(len(r_flux)):
        circle=regions[i].to_mask(mode='exact')
        circle_mask =  circle.cutout(mask)              #Define the mask for the region frame.
        region_size[i]=(circle.data * circle_mask).sum()    #circle.data is the region size of each pixel. 
    if if_annuli ==False:
        r_SB= r_flux/region_size
    elif if_annuli == True:
        r_SB = np.zeros_like(r_flux)
        r_SB[0] = r_flux[0]/region_size[0]
        r_SB[1:] = (r_flux[1:]-r_flux[:-1]) / (region_size[1:]-region_size[:-1])
    if fits_plot == True:
        ax=plt.subplot(1,1,1)
        cax=ax.imshow(image*mask,norm=LogNorm(),origin='lower')
        for i in range(grids):
            ax.add_patch(regions[i].as_artist(facecolor='none', edgecolor='orange'))
        plt.colorbar(cax)
        plt.show()
    if if_plot == True:
        minorLocator = AutoMinorLocator()
        fig, ax = plt.subplots()
        plt.plot(r_grids, r_SB, 'x-')
        ax.xaxis.set_minor_locator(minorLocator)
        plt.tick_params(which='both', width=2)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='r')
        plt.grid()
        ax.set_ylabel("Surface Brightness")
        ax.set_xlabel("Pixels")
        if x_gridspace == 'log':
            ax.set_xscale('log')
            plt.xlim(start_p*0.7, ) 
        plt.grid(which="minor")
        plt.show()
    return r_SB, r_grids

def profiles_compare(prf_list, prf_name_list = None, x_gridspace = None ,
                     grids = 20,  norm_pix = 3, if_annuli=False,
                     y_log=False, scale_list=None):
    '''
    Compare the SB profile between different images. 
    Parameter
    --------
        prf_list: a list of image profiles.
        prf_name_list: a list of name for each profiles.
        norm_pix: The x-position (i.e. pixel) to norm the profiles.
        scale_list: a list for the scaled value for the resultion, default as scale as 1 (set by None), i.e. same resolution.
    Return
    --------
        A plot of SB comparison.
    '''
    if x_gridspace == None:
        radius = 6
    elif x_gridspace == 'log':
        radius = len(prf_list[1])/2
    if scale_list == None:
        scale_list = [1] * len(prf_list)
    minorLocator = AutoMinorLocator()
    fig, ax = plt.subplots(figsize=(10,7))
    prf_NO = len(prf_list)
    if prf_name_list == None:
        prf_name_list = ["Profile-{0}".format(i) for i in range(len(prf_list))]
    if len(prf_name_list)!=len(prf_list):
        raise ValueError("The profile name is not in right length")
    for i in range(prf_NO):
        b_c = int(len(prf_list[i])/2)
        b_r = int(len(prf_list[i])/6)
        center = np.reshape(np.asarray(np.where(prf_list[i]== prf_list[i][b_c-b_r:b_c+b_r,b_c-b_r:b_c+b_r].max())),(2))[::-1]
        scale = scale_list[i]
        r_SB, r_grids = SB_profile(prf_list[i], center, radius=radius*scale,
                                   grids=grids, x_gridspace=x_gridspace,if_annuli=if_annuli)
        
        if isinstance(norm_pix,int) or isinstance(norm_pix,float):
            count = r_grids <= norm_pix * scale
            idx = count.sum() -1
#            print("idx:",idx)
            r_SB /= r_SB[idx]      #normalize the curves
        r_grids /= scale
        
        if y_log == False:
            plt.plot(r_grids, r_SB, 'x-', label=prf_name_list[i])
        elif y_log == True:
            plt.plot(r_grids, np.log10(r_SB), 'x-', label=prf_name_list[i])
            # plt.ylim(0, 0.5) 
    ax.xaxis.set_minor_locator(minorLocator)
    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    plt.grid()
    ax.set_ylabel("Scaled Surface Brightness")
    ax.set_xlabel("Pixels")
    if x_gridspace == 'log':
        ax.set_xscale('log')
        # plt.xlim(1.3, ) 
    plt.grid(which="minor")
    plt.legend()
    plt.show() 
    return fig

def measure_bkg(img, if_plot=False, nsigma=2, npixels=25, dilate_size=11):
    from astropy.stats import SigmaClip
    from photutils import Background2D, SExtractorBackground  
    sigma_clip = SigmaClip(sigma=3., maxiters=10)
    bkg_estimator = SExtractorBackground()
    if version.parse(photutils.__version__) > version.parse("0.7"):
        mask_0 = make_source_mask(img, nsigma=nsigma, npixels=npixels, dilate_size=dilate_size)
    else:
        mask_0 = make_source_mask(img, snr=nsigma, npixels=npixels, dilate_size=dilate_size)
    mask_1 = (np.isnan(img))
    mask = mask_0 + mask_1
    bkg = Background2D(img, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
                       mask=mask)
    from matplotlib.colors import LogNorm
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    ax.imshow(img, norm=LogNorm(), origin='lower') 
    #bkg.plot_meshes(outlines=True, color='#1f77b4')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if if_plot:
        plt.show()  
    else:
        plt.close()
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    ax.imshow(mask, origin='lower') 
    #bkg.plot_meshes(outlines=True, color='#1f77b4')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if if_plot:
        plt.show()  
    else:
        plt.close()    
    bkg_light = bkg.background* ~mask_1
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    ax.imshow(bkg_light, origin='lower', cmap='Greys_r')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if if_plot:
        plt.show()  
    else:
        plt.close()
    return bkg_light   

def string_find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def cr_mask(image, filename='test_circle.reg'):
    '''
    The creat a mask using a DS9 .reg file. The pixels in the region are 0, ouside ones are 1.
    
    Parameter
    --------
        image: a 2D array image as a template frame.
        filename: full name of the region file.
        
    Return
    --------
        A image.shape array. Pixels in the region is 0, otherwise 1.
    '''
    with open(filename, 'r') as input_file:
        reg_string=input_file.read().replace('\n', '')
    if "physicalcircle" in reg_string:
        abc=string_find_between(reg_string, "(", ")")
        reg_info=np.fromstring(abc, dtype=float, sep=',')
        center, radius = reg_info[:2]-1 , reg_info[2]
        region = pix_region(center, radius)
        box = 1-region.to_mask(mode='center').data
    elif "physicalbox" in reg_string:
        abc=string_find_between(reg_string, "(", ")")
        reg_info=np.fromstring(abc, dtype=float, sep=',')
        center = reg_info[:2] - 1
        x_r, y_r = reg_info[2:4]  # x_r is the length of the x, y_r is the length of the y
        box = np.zeros([np.int(x_r)+1, np.int(y_r)+1]).T
    else:
        print(reg_string)
        raise ValueError("The input reg is un-defined yet")
    frame_size = image.shape
    box_size = box.shape
    x_edge = np.int(center[1]-box_size[0]/2) #The position of the center is x-y switched.
    y_edge = np.int(center[0]-box_size[1]/2)
    mask = np.ones(frame_size)
    mask_box_part = mask[x_edge:x_edge+box_size[0],y_edge: y_edge + box_size[1]]
    mask_box_part *= box
    return mask    


def detect_obj(image, nsigma=2.8, exp_sz= 1.2, npixels = 15, if_plot=False, auto_sort_center = True):  
    """
    Define the apeatures for all the objects in the image.
    
    Parameter
    --------
        img : 2-D array like
        The input image
    
        exp_sz : float
        The level to expand the mask region.
    
        nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.
        
        npixels: int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.
        b: The blash of blash.
        
        if_plot: bool
        If ture, plot the detection figure.    
        
    Return
    --------
        A list of photutils defined apeatures that cover the detected objects.
    """ 
    from photutils import detect_threshold
    from astropy.stats import gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    from photutils import detect_sources,deblend_sources   
    from photutils import source_properties
    if version.parse(photutils.__version__) > version.parse("0.7"):
        threshold = detect_threshold(image, nsigma=nsigma)
    else:
        threshold = detect_threshold(image, snr=nsigma)
    # center_image = len(image)/2
    sigma = 3.0 * gaussian_fwhm_to_sigma # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(image, threshold, npixels=npixels, filter_kernel=kernel)
    segm_deblend = deblend_sources(image, segm, npixels=npixels,
                                    filter_kernel=kernel, nlevels=25,
                                    contrast=0.001)
    #Number of objects segm_deblend.data.max()
    cat = source_properties(image, segm_deblend)
    columns = ['id', 'xcentroid', 'ycentroid', 'source_sum', 'orientation', 'area']
    tbl = cat.to_table(columns=columns)
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['id'] -= 1
    
    apertures = []
    segm_deblend_size = segm_deblend.areas
    from photutils import EllipticalAperture
    for obj in cat:
        size = segm_deblend_size[obj.id-1]
        position = (obj.xcentroid.value, obj.ycentroid.value)
        a_o = obj.semimajor_axis_sigma.value
        b_o = obj.semiminor_axis_sigma.value
        size_o = np.pi * a_o * b_o
        r = np.sqrt(size/size_o)*exp_sz
        a, b = a_o*r, b_o*r
        if version.parse(photutils.__version__) > version.parse("0.7"):
            theta = obj.orientation.value / 180 * np.pi
        else:
            theta = obj.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))     
    
    if auto_sort_center == True:
        center = np.array([len(image)/2, len(image)/2])
        dis_sq = [np.sum((apertures[i].positions - center)**2) for i in range(len(apertures))]
        dis_sq = np.array(dis_sq)
        c_idx = np.where(dis_sq == dis_sq.min())[0][0]
        apertures = [apertures[c_idx]] + [apertures[i] for i in range(len(apertures)) if i != c_idx] 
        cat = [cat[c_idx]] + [cat[i] for i in range(len(cat)) if i != c_idx] 
        tbl['id'][0] = c_idx
        tbl['id'][c_idx] = 0
        
    if if_plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 10))
        vmin = 1.e-3
        vmax = 2.1 
        ax1.imshow(image, origin='lower', cmap=my_cmap, norm=LogNorm(), vmin=vmin, vmax=vmax)
        ax1.set_title('Data')
        if version.parse(photutils.__version__) > version.parse("0.7"):
            ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.make_cmap(random_state=12345))
        else:
            ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap(random_state=12345))
        for i in range(len(cat)):
            ax2.text(cat[i].xcentroid.value, cat[i].ycentroid.value, '{0}'.format(i), fontsize=15,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 1})
        for i in range(len(apertures)):
            aperture = apertures[i]
            if version.parse(photutils.__version__) > version.parse("0.7"):
                aperture.plot(color='white', lw=1.5, axes=ax1)
                aperture.plot(color='white', lw=1.5, axes=ax2)           
            else:
                aperture.plot(color='white', lw=1.5, ax=ax1)
                aperture.plot(color='white', lw=1.5, ax=ax2)                       
        ax2.set_title('Segmentation Image')
        plt.show()    
        print(tbl)
    return apertures

def mask_obj(image, apertures, if_plot = True):
    from regions import PixCoord, EllipsePixelRegion
    from astropy.coordinates import Angle
    masks = []  # In the script, the objects are 1, emptys are 0.
    for i in range(len(apertures)):
        aperture = apertures[i]
        if isinstance(apertures[0].positions[0],np.ndarray):
            x, y = aperture.positions[0]
        elif isinstance(apertures[0].positions[0],float):
            x, y = aperture.positions
        center = PixCoord(x=x, y=y)
        theta = Angle(aperture.theta/np.pi*180.,'deg')
        reg = EllipsePixelRegion(center=center, width=aperture.a*2, height=aperture.b*2, angle=theta)
        patch = reg.as_artist(facecolor='none', edgecolor='red', lw=2)
        mask_set = reg.to_mask(mode='center')
        mask = mask_set.to_image((len(image),len(image)))
        mask = 1- mask
        if if_plot:
            print( "plot mask for object {0}:".format(i) )
            fig, axi = plt.subplots(1, 1, figsize=None)
            axi.add_patch(patch)
            axi.imshow(mask, origin='lower')
            plt.show()
        masks.append(mask)
    return masks

def esti_bgkstd(image, nsigma=2, exp_sz= 1.5, npixels = 15, if_plot=False):
    apertures = detect_obj(image, nsigma=nsigma , exp_sz=exp_sz, npixels = npixels)
    mask_list = mask_obj(image, apertures, if_plot=False)
    mask = np.ones_like(image)
    for i in range(len(mask_list)):
        mask *= mask_list[i]
    image_mask = image*mask
    stdd = np.std(image_mask[image_mask!=0])
    if if_plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))
        ax1.imshow(image_mask, origin='lower', cmap=my_cmap, norm=LogNorm())
        ax1.set_title('pixels used to estimate background stdd')
        values = ax2.hist(image_mask[image_mask!=0])
        ax2.plot(np.zeros(5)+np.median(image_mask[image_mask!=0]), np.linspace(0,values[0].max(),num=5), 'k--', linewidth = 4)
        ax2.set_title('pixel value histogram and median point (mid = {0:.4f}).'.format(np.median(image_mask[image_mask!=0])))
    return stdd

def model_flux_cal(params_list, model_list = None):
    """
    Calculate the flux, itergral to infinite, i.e., same defination as Galfit.
    
    Parameter
    --------
        params_list: a list of (Sersic) Soure params defined by Lenstronomy
        model_list: a list of names of light profile model.
    Return
    --------
        Total calcualted flux 
    """    
    from lenstronomy.LightModel.light_model import LightModel
    if model_list is None:
        model_list = ['SERSIC_ELLIPSE'] * len(params_list)
    light = LightModel(model_list)
    flux = light.total_flux(params_list)
    return flux
