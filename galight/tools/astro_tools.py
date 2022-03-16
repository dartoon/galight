#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:32:21 2020

@author: Xuheng Ding
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits

from astropy.wcs import WCS
from matplotlib.colors import LogNorm

def read_pixel_scale(header):
    """
    Readout the pixel scale from a pyfits file.
    
    Parameter
    --------
        header: 
            A fits file header from pyfits.open('filename').
        
    Return
    --------
        The pixel scale in arcsec scale.
    """
    wcs = WCS(header)
    # diff_RA_DEC = wcs.all_pix2world([0,0],[0,100],1)
    # diff_scale = np.sqrt((diff_RA_DEC[0][1]-diff_RA_DEC[0][0])**2 + (diff_RA_DEC[1][1]-diff_RA_DEC[1][0])**2)
    # pix_scale = diff_scale * 3600 / 100
    from astropy.wcs.utils import proj_plane_pixel_scales
    scales = proj_plane_pixel_scales(wcs) * 3600  #From degree to arcsec
    if scales[0] != scales[1]:
        print('Warning: Pixel scale is not the same along x and y!!!')
    pix_scale = scales[0] 
    return pix_scale

def read_fits_exp(header):
    """
    Readout the header information from a pyfits file.
    
    Parameter
    --------
        header:
            A fits file header from pyfits.open('filename').
        
    Return
    --------
        The pixel scale in arcsec scale.
    """    
    return header['EXPTIME']

def plt_fits(img, norm = None, figsize = None, colorbar = False, savename = None, vmin= None, vmax=None):
    """
    Directly plot a 2D image using imshow.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if norm is None or norm == 'log':
        norm = LogNorm(vmin=vmin, vmax=vmax)#np.max(img[~np.isnan(img)]))
    else:
        norm = None
    plt.imshow(img, norm=norm, origin='lower') 
    if colorbar == True:
        plt.colorbar()
    if savename is not None:
        plt.savefig(savename)
    plt.show()     
    # plt.imshow(img, norm=LogNorm(), cmap = 'gist_heat', origin='low')   
    # plt.colorbar()
    # plt.show()
    
def plt_fits_color(imgs, savename = None, **args):
    from astropy.visualization import make_lupton_rgb
    rgb_default = make_lupton_rgb(imgs[0], imgs[1], imgs[2], **args)
    plt.imshow(rgb_default, origin='lower')
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def plt_many_fits(imgs, texts = None, prop = None, savename = None, labels = None, hide_axes = False,
                  if_plot=True, cmap=None):
    _row = int(len(imgs) / 5) + 1
    if _row<=1:
        _row=2
    fig, (axs) = plt.subplots(_row, 5, figsize=(15, 3 + 3 * (_row-1)))
    import matplotlib as mat
    mat.rcParams['font.family'] = 'STIXGeneral'
    for i in range(len(imgs)):
        _i = int(i / 5)
        _j = int(i % 5)
        axs[_i][_j].imshow(imgs[i], origin='lower', norm=LogNorm(), cmap=cmap)
        frame_size = len(imgs[i])
        if labels is None:
            label = "ini_ID = {0}".format(i)
        else:
            label = labels[i]
        plttext = axs[_i][_j].text(frame_size*0.05, frame_size*0.87, label,
                 fontsize=17, weight='bold', color='black')
        plttext.set_bbox(dict(facecolor='white', alpha=0.5))
        if texts is not None:
            plttext = axs[_i][_j].text(frame_size*0.05, frame_size*0.05, "{1} = {0}".format(round(texts[i],3), prop ),
                     fontsize=17, weight='bold', color='black')
            plttext.set_bbox(dict(facecolor='white', alpha=0.5))
        if hide_axes == True:
            axs[_i][_j].axes.xaxis.set_visible(False)
            axs[_i][_j].axes.yaxis.set_visible(False)
    for i in range(len(imgs), 5*_row):
        _i = int(i / 5)
        _j = int(i % 5)
        axs[_i][_j].axes.xaxis.set_visible(False)
        axs[_i][_j].axes.yaxis.set_visible(False)
        axs[_i][_j].axis('off')
    # for i in range( 5 - len(imgs)%5 ):
    #     axs[-1][-(i+1)].axis('off')
    if savename is not None:
        plt.savefig(savename)
    if if_plot == True:
        plt.show()    
    else:
        plt.close()