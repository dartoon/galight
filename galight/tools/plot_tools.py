#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:28:10 2020

@author: Xuheng Ding
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from matplotlib.colors import LogNorm
from galight.tools.measure_tools import SB_profile
import copy, matplotlib
from matplotlib.ticker import ScalarFormatter
from galight.tools.astro_tools import plt_fits, plt_fits_color, plt_many_fits
from galight.tools.cutout_tools import plot_overview

my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad('black')
import matplotlib as matt
matt.rcParams['font.family'] = 'STIXGeneral'

def coordinate_arrows(ax, d, color='black', arrow_size=0.02):
    d0 = d / 12.
    p0 = d / 12.
    pt = d / 7.
    deltaPix = 1
    ra0, dec0 = (d - d0) / deltaPix, d0 / deltaPix
    xx_, yy_ = ra0, dec0
    xx_ra, yy_ra = (ra0 - p0, dec0)
    xx_dec, yy_dec = (ra0, dec0 + p0)
    xx_ra_t, yy_ra_t = (ra0 - pt, dec0)
    xx_dec_t, yy_dec_t = (ra0, dec0 + pt)

    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_ra - xx_) * deltaPix, (yy_ra - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc=color, ec=color, linewidth=1.2)
    ax.text(xx_ra_t * deltaPix, yy_ra_t * deltaPix, "E", color=color, fontsize=12, ha='center')
    ax.arrow(xx_ * deltaPix, yy_ * deltaPix, (xx_dec - xx_) * deltaPix, (yy_dec - yy_) * deltaPix,
             head_width=arrow_size * d, head_length=arrow_size * d, fc
             =color, ec=color, linewidth=1.2)
    ax.text(xx_dec_t * deltaPix, yy_dec_t * deltaPix, "N", color=color, fontsize=12, ha='center')
    
def scale_bar(ax, d, dist=1/0.13, text='1"', color='black', flipped=False, fontsize=15):
    if flipped:
        p0 = d - d / 15. - dist
        p1 = d / 15.
        ax.plot([p0, p0 + dist], [p1, p1], linewidth=2, color=color)
        ax.text(p0 + dist / 2., p1 + 0.02 * d, text, fontsize=fontsize, color=color, ha='center')
    else:
        p0 = d / 15.
        ax.plot([p0, p0 + dist], [p0, p0], linewidth=2, color=color)
        ax.text(p0 + dist / 2., p0 + 0.02 * d, text, fontsize=fontsize, color=color, ha='center')


def total_compare(flux_list_2d, label_list_2d, flux_list_1d, label_list_1d,
                  deltaPix = 1., zp=27.0, target_ID = 'target_ID',
                  mask_image=None, if_annuli=False,
                  arrows=False, show_plot = True):
    """
    Make quick plots to compare the flux profiles in a list and show the normalized residual.
    
    Parameter
    --------
        flux_list_2d: 
            A list of 2D flux array, that will use plt.imshow() to plot and show.
            e.g., [data, pointsource_list, galaxy_model_list, normalized residual]
            
        label_list_2d: 
            A list of lables for flux_list_2d.
            e.g., ['data', 'model', 'point source(s)', 'galaxy(s)']
            
        flux_list_1d:  
            A list of 2D flux array, that will be plot as 1D profile in the very right panel.
        
        label_list_1d: 
            The labels for flux_list_1d.
        
        mask_image: 
            A 2D mask for the flux_list_2d image.
        
        arrows: bool. 
            If show the arrows for pointing the North and East.
        
        if_annuli: bool.
            If True, the 1D profile will show the surface brightness in the annuli apertures. 
    """
    # norm = LogNorm() #ImageNormalize(stretch=SqrtStretch())
    cl_num = len(flux_list_2d) + 1 
    f = plt.figure(0, figsize=(6.5+ (cl_num-1)*3.5,4))    
    # f = plt.figure(0, figsize=(17.0,4))  #3
    # f = plt.figure(0, figsize=(20.5,4))  #4
    # f = plt.figure(0, figsize=(24.0,4))  #5
    ax_l = [plt.subplot2grid((6,cl_num), (0,i), rowspan=6) for i in range(len(flux_list_2d)-1)] #The image plot
    ax_r = plt.subplot2grid((6,cl_num), (0,cl_num-2), rowspan=6)   #The residual plot
    ax_rt = plt.subplot2grid((6,cl_num), (0,cl_num-1), rowspan=5)
    ax_rb = plt.subplot2grid((6,cl_num), (5,cl_num-1), rowspan=1)
    frame_size = len(flux_list_2d[0])
    mask = np.ones_like(flux_list_2d[0])
    if mask_image is not None:
        mask = mask * mask_image
    for i in range(len(flux_list_2d)-1):
        if i >1:
            flux_list_2d[i] = flux_list_2d[i] * mask
        if i == 0:
            im_i = ax_l[i].imshow(flux_list_2d[i],origin='lower',cmap=my_cmap, norm=LogNorm(vmax = flux_list_2d[0].max(), vmin = 1.e-4))
            clim=im_i.properties()['clim'] #To uniform the color bar scale.
            ax_l[i].set_ylabel(target_ID, fontsize=15, weight='bold')
        else:
            im_i = ax_l[i].imshow(flux_list_2d[i],origin='lower',cmap=my_cmap, norm=LogNorm(), clim=clim)
            ax_l[i].get_yaxis().set_visible(False)
        ax_l[i].get_xaxis().set_visible(False)
        scale_bar(ax_l[i], frame_size, dist=1/deltaPix, text='1"', color = 'white')
        if arrows == True:
            coordinate_arrows(ax_l[i], frame_size, arrow_size=0.03, color = 'white')
        ticks= np.array([1.e-4, 1.e-3, 1.e-2,1.e-1,0, 10])
        cb_i = f.colorbar(im_i, ax=ax_l[i], shrink=0.48, pad=0.01,  orientation="horizontal", 
                          aspect=15, ticks=ticks)
        # cb_i.ax.set_xticklabels([1.e-4, 1.e-3, 1.e-2,1.e-1,0, 10])   
        if len(label_list_2d[i])>10:
            fontsize = 17
        else:
            fontsize = 20
        ax_l[i].text(frame_size*0.05, frame_size*0.9, label_list_2d[i],fontsize=fontsize, weight='bold', color='white')
    #Plot normalized residual map:
    norm_residual = flux_list_2d[-1]
    im_r = ax_r.imshow(norm_residual * mask, origin='lower',cmap='bwr', vmin=-6, vmax=6)
    scale_bar(ax_r, frame_size, dist=1/deltaPix, text='1"')
    if arrows == True:
        coordinate_arrows(ax_r, frame_size, arrow_size=0.03)
    ax_r.get_xaxis().set_visible(False)
    ax_r.get_yaxis().set_visible(False)
    f.colorbar(im_r, ax=ax_r, shrink=0.48, pad=0.01,   orientation="horizontal", aspect=15) 
    ax_r.text(frame_size*0.05, frame_size*0.9, 'normalized residual',fontsize=17, weight='bold', color='black')
    plt.subplots_adjust(wspace=-0.5, hspace=0)
    
    #Plot the 1D profile:
    label_SB_list = label_list_1d #Not show the residual, in order of data, model, QSO, galaxy in principle.
    flux_SB_list = flux_list_1d
    radi = len(flux_list_1d[0])/2
    if if_annuli == False:
        for i in range(len(label_SB_list)):
            center = len(flux_SB_list[i])/2, len(flux_SB_list[i])/2
            if label_SB_list[i] == 'data':
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius= radi, grids = 50,
                                           mask_image=mask_image, fits_plot=False)
            else:
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log', radius= radi,
                                           grids = 30, mask_image = mask_image)
            r_mag = - 2.5 * np.log10(r_SB) + zp 
            if label_SB_list[i] == 'data':
                ind = len(r_mag)-(r_mag == r_mag[-1]).sum()
                ax_rt.plot(r_grids[:ind], r_mag[:ind], 'o', color = 'whitesmoke',markeredgecolor="black", label=label_SB_list[i])
            else:
                ax_rt.plot(r_grids, r_mag, '-', label=label_SB_list[i])
        ax_rt.set_ylabel('$\mu$(mag, pixel$^{-2}$)', fontsize=12)
        ax_rt.invert_yaxis()
        r_mag_0 = 2.5 * np.log10(SB_profile(flux_SB_list[0], center, x_gridspace = 'log', radius= radi,
                                            grids = 30, mask_image=mask_image)[0])
        r_mag_1 = 2.5 * np.log10(SB_profile(np.sum(flux_SB_list[1:], axis=0), center, x_gridspace = 'log', grids = 30,radius= radi)[0])
        ind = len(r_mag_0)-(r_mag_0 == r_mag_0[-1]).sum()
        ax_rb.plot(r_grids[:ind]*deltaPix, (r_mag_0-r_mag_1)[:ind], 'ro')   
        ax_rb.set_yticks([-0.5,-0.25, 0., 0.25])
        ax_rb.set_ylabel('$\Delta\mu$', fontsize=15)
        plt.ylim([-0.5,0.5])
    elif if_annuli == True:
        for i in range(len(label_SB_list)):
            center = len(flux_SB_list[i])/2, len(flux_SB_list[i])/2
            if label_SB_list[i] == 'data':
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius = radi, grids = 50, 
                                           mask_image = mask_image, fits_plot=False, if_annuli = if_annuli)
                ax_rt.plot(r_grids, r_SB, 'o', color = 'whitesmoke',markeredgecolor="black", label=label_SB_list[i])
            else:
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius=radi,grids = 30, mask_image=mask_image, if_annuli = if_annuli)
                ax_rt.plot(r_grids, r_SB, '-', label=label_SB_list[i])
        ax_rt.set_ylabel('$SB_{annuli}$(counts, pixel$^{-2}$)', fontsize=12)
        r_SB_0 = (SB_profile(flux_SB_list[0], center, x_gridspace = 'log', radius= radi, if_annuli = if_annuli, 
                                            grids = 30,
                                            mask_image = mask_image)[0])
        r_SB_1 = (SB_profile(flux_SB_list[1], center, x_gridspace = 'log', grids = 30, if_annuli = if_annuli,radius= radi)[0])
        ax_rb.plot(r_grids*deltaPix, (r_SB_0- r_SB_1), 'ro')   
        ax_rb.set_yticks([-5,-2.5, 0., 2.5])
        ax_rb.set_ylabel('$\Delta SB$', fontsize=15)
        plt.ylim([-5,5])
    ax_rt.set_xlabel('pixel', fontsize=15)
    ax_rt.xaxis.set_label_position('top')
    ax_rt.xaxis.tick_top() 
    ax_rt.set_xscale('log')
    ax_rt.set_xticks([2,4,6,10,15,20,30,50,100,150])
    ax_rt.xaxis.set_major_formatter(ScalarFormatter())
    ax_rt.set_xlim([(r_grids).min()*0.85,r_grids.max()+6])
    ax_rt.yaxis.set_label_position('right')
    ax_rt.yaxis.tick_right()
    ax_rt.yaxis.set_ticks_position('both') 
    ax_rt.legend()
    x = np.linspace((r_grids*deltaPix).min()*0.85, (r_grids.max()+6)*deltaPix)
    y = x * 0
    ax_rb.set_xlabel('arcsec', fontsize=15)
    ax_rb.set_xscale('log')
    ax_rb.set_xticks([0.1, 0.2, 0.5, 1, 2,5,10,20])
    ax_rb.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_rb.plot(x, y, 'k--')  
    ax_rb.yaxis.set_label_position('right')
    ax_rb.yaxis.tick_right()
    ax_rb.yaxis.set_ticks_position('both')
    ax_rb.set_xlim([(r_grids*deltaPix).min()*0.85, (r_grids.max()+6)*deltaPix])
    pos4_o = ax_rt.get_position() # get the original position
    pos5_o = ax_rb.get_position() # get the original position
    pos4 = [pos4_o.x0+0.112 - 0.009 * cl_num , pos4_o.y0 + 0.10, pos4_o.width*0.72, pos4_o.height*0.8]
    pos5 = [pos5_o.x0+0.112 - 0.009 * cl_num , pos5_o.y0+0.08, pos5_o.width*0.72, pos5_o.height*1.1]      
    ax_rt.set_position(pos4) # set a new position
    ax_rb.set_position(pos5) # set a new position
    if show_plot == True:
        plt.show()       
    else:
        plt.close()
    return f


def profile_plots(flux_list_2d, label_list_2d, flux_list_1d, label_list_1d,
                  deltaPix = 1., zp=27.0, target_ID = 'target_ID',
                  mask_image=None, if_annuli=False,
                  arrows=False, show_plot = True):
    """
    Similar to total_compare(), i.e., to compare a list of light profiles but without showing normlized residual.
    """
    # norm = LogNorm() #ImageNormalize(stretch=SqrtStretch())
    cl_num = len(flux_list_2d) + 1 
    f = plt.figure(0, figsize=(6.5+ (cl_num-1)*3.5,4))    
    # f = plt.figure(0, figsize=(17.0,4))  #3
    # f = plt.figure(0, figsize=(20.5,4))  #4
    # f = plt.figure(0, figsize=(24.0,4))  #5
    ax_l = [plt.subplot2grid((6,cl_num), (0,i), rowspan=6) for i in range(len(flux_list_2d))] #The image plot
    ax_rt = plt.subplot2grid((6,cl_num), (0,cl_num-1), rowspan=6)
    # ax_rb = plt.subplot2grid((6,cl_num), (5,cl_num-1), rowspan=1)
    frame_size = len(flux_list_2d[0])
    mask = np.ones_like(flux_list_2d[0])
    if mask_image is not None:
        mask = mask * mask_image
    for i in range(len(flux_list_2d)):
        if i >1:
            flux_list_2d[i] = flux_list_2d[i] * mask
        if i == 0:
            im_i = ax_l[i].imshow(flux_list_2d[i] * mask ,origin='lower',cmap=my_cmap, norm=LogNorm(vmax = flux_list_2d[0].max(), vmin = 1.e-4))
            clim=im_i.properties()['clim'] #To uniform the color bar scale.
            ax_l[i].set_ylabel(target_ID, fontsize=15, weight='bold')
        else:
            im_i = ax_l[i].imshow(flux_list_2d[i],origin='lower',cmap=my_cmap, norm=LogNorm(), clim=clim)
            ax_l[i].get_yaxis().set_visible(False)
        ax_l[i].get_xaxis().set_visible(False)
        scale_bar(ax_l[i], frame_size, dist=1/deltaPix, text='1"', color = 'white')
        if arrows == True:
            coordinate_arrows(ax_l[i], frame_size, arrow_size=0.03, color = 'white')
        ticks= np.array([1.e-4, 1.e-3, 1.e-2,1.e-1,0, 10])
        cb_i = f.colorbar(im_i, ax=ax_l[i], shrink=0.48, pad=0.01,  orientation="horizontal", aspect=15, 
                          ticks= ticks)
        # cb_i.set_ticks([1.e-5, 1.e-4, 1.e-3, 1.e-2,1.e-1,0,1,10,100])   
        if len(label_list_2d[i])>10:
            fontsize = 17
        else:
            fontsize = 20
        ax_l[i].text(frame_size*0.05, frame_size*0.9, label_list_2d[i],fontsize=fontsize, weight='bold', color='white')

    plt.subplots_adjust(wspace=-0.45, hspace=0)
        
    #Plot the 1D profile:
    label_SB_list = label_list_1d #Not show the residual, in order of data, model, QSO, galaxy in principle.
    flux_SB_list = flux_list_1d
    radi = len(flux_list_1d[0])/2
    if if_annuli == False:
        for i in range(len(label_SB_list)):
            center = len(flux_SB_list[i])/2, len(flux_SB_list[i])/2
            if label_SB_list[i] == 'data':
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius= radi, grids = 50,
                                           mask_image=mask_image, fits_plot=False)
            else:
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log', radius= radi,
                                           grids = 30, mask_image = mask_image)
            r_mag = - 2.5 * np.log10(r_SB) + zp 
            if label_SB_list[i] == 'data':
                ind = len(r_mag)-(r_mag == r_mag[-1]).sum()
                ax_rt.plot(r_grids[:ind], r_mag[:ind], 'o', color = 'whitesmoke',markeredgecolor="black", label=label_SB_list[i])
            else:
                ax_rt.plot(r_grids, r_mag, '-', label=label_SB_list[i])
        ax_rt.set_ylabel('$\mu$(mag, pixel$^{-2}$)', fontsize=12)
        ax_rt.invert_yaxis()
    elif if_annuli == True:
        max_y = 0
        for i in range(len(label_SB_list)):
            center = len(flux_SB_list[i])/2, len(flux_SB_list[i])/2
            if label_SB_list[i] == 'data':
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius = radi, grids = 50, 
                                           mask_image = mask_image, fits_plot=False, if_annuli = if_annuli)
                ax_rt.plot(r_grids, r_SB, 'o', color = 'whitesmoke',markeredgecolor="black", label=label_SB_list[i])
            else:
                r_SB, r_grids = SB_profile(flux_SB_list[i], center, x_gridspace = 'log',
                                           radius=radi,grids = 30, mask_image=mask_image, if_annuli = if_annuli)
                ax_rt.plot(r_grids, r_SB, '-', label=label_SB_list[i])
            if max_y < np.max(r_SB):
                max_y = np.max(r_SB)
        ax_rt.set_ylabel('$SB_{annuli}$(counts, pixel$^{-2}$)', fontsize=12)
        ax_rt.set_yscale('log')
        ax_rt.set_ylim([10**(-3), max_y])
    ax_rt.set_xlabel('pixel', fontsize=15)
    ax_rt.set_xscale('log')
    ax_rt.set_xticks([2,4,6,10,15,20,30,50,100,150])
    ax_rt.xaxis.set_major_formatter(ScalarFormatter())
    ax_rt.set_xlim([(r_grids).min()*0.85,r_grids.max()+6])
    
    #TODO!!!: Update the axis in arcsec on upside: 
    # ax_rt2 = ax_rt.twiny()
    # new_tick_locations = np.array([ax_rt.get_xlim()[0], 0.2, 0.5, 1, 2,5 , ax_rt.get_xlim()[1]])*deltaPix
    # ax_rt2.set_xticks(new_tick_locations)
    # ax_rt2.set_xticklabels([0.1, 0.2, 0.5, 1, 2,5])
    # ax_rt2.set_xlabel("arcsec")    
    
    ax_rt.yaxis.set_label_position('right')
    ax_rt.yaxis.tick_right()
    ax_rt.yaxis.set_ticks_position('both') 
    ax_rt.legend()
    pos4_o = ax_rt.get_position() # get the original position
    pos4 = [pos4_o.x0+0.112 - 0.009 * cl_num , pos4_o.y0 + 0.10, pos4_o.width*0.72, pos4_o.height*0.8]
    ax_rt.set_position(pos4) # set a new position
    if show_plot == True:
        plt.show()       
    else:
        plt.close()
    return f
