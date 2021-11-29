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
    diff_RA_DEC = wcs.all_pix2world([0,0],[0,100],1)
    diff_scale = np.sqrt((diff_RA_DEC[0][1]-diff_RA_DEC[0][0])**2 + (diff_RA_DEC[1][1]-diff_RA_DEC[1][0])**2)
    pix_scale = diff_scale * 3600 / 100
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

def plt_fits(img, norm = LogNorm(), figsize = None, colorbar = False):
    """
    Directly plot a 2D image using imshow.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(img, norm=norm, origin='lower', vmax = np.max(img[~np.isnan(img)])) 
    if colorbar == True:
        plt.colorbar()
    plt.show()     
    # plt.imshow(img, norm=LogNorm(),origin='low')   
    # plt.colorbar()
    # plt.show()
    
def plt_fits_color(imgs, **args):
    from astropy.visualization import make_lupton_rgb
    rgb_default = make_lupton_rgb(imgs[0], imgs[1], imgs[2], **args)
    plt.imshow(rgb_default, origin='lower')
    plt.show()