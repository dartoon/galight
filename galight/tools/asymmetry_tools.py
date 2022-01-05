#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:38:43 2022

@author: Dartoon
"""

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy import ndimage
# from cutout_tools import exp_grid
import scipy.optimize as op
class Measure_asy:
    """
    Measure the asymmetry of a image.
    
    Funtions including using interpolation to move a image. Rotation a image based on  
    --------
        a: The blash of blash
        b: The blash of blash
        
    Return
    --------
        A sth sth
    """
    def __init__(self,img=None, order=2):  #order relies on the background rms
        self.img = img
        self.dy,self.dx = img.shape
        self.x0 = int(self.dx/2)
        self.y0 = int(self.dy/2)
        self.order = order
        self.nx, self.ny = None, None   #For coordinate the pixel position
        if  self.order==1:
            self.model = self.img
        else:
            self.model = ndimage.spline_filter(self.img,output=np.float64,order=order)

    def getPix(self,sx,sy):
        x = (sx-self.nx)+self.x0
        y = (sy-self.ny)+self.y0
        return x,y

    def evaluateSource(self,sx,sy):
        ix,iy = self.getPix(sx,sy)
        return ndimage.map_coordinates(self.model,[[iy],[ix]],order=self.order)[0,:,:]
  
    def shift_img(self, shift_pix):
        """
        Shift the image 
        """
        y,x = np.indices((int(len(self.img)),int(len(self.img)))).astype(np.float64)
        self.nx=len(self.img)/2+shift_pix[0]  # 0.x if the residul is *.x
        self.ny=len(self.img)/2+shift_pix[1]  # 0.y if the residul is *.y
        return self.evaluateSource(x,y)
  
    def rotate_image(self, rotate_pix):
        """
        Rotate a image using the rotate_pox
        
        Parameter
        --------
            rotate_pix: the rotate pixel that relative to the center.
            
        """
        shift_pix = [-rotate_pix[0]*2, -rotate_pix[1]*2]
        shift_ = self.shift_img(shift_pix)
        rotate = np.flip(shift_)
        return rotate
    
    def residual(self, rotate_pix):
        rotate_ = self.rotate_image(rotate_pix)
        residual = self.img - rotate_  #Consider resdiual as data-model, where model is the rotation.
        return residual
    
    def abs_res(self, theta):
        res_ = self.residual(theta)
        return np.sum(abs(res_))
        
    def find_pos(self, theta = [0.0,0.0]):
        result = op.minimize(self.abs_res, theta, method='nelder-mead',
                options={'xatol': 1e-8, 'disp': True})
        return result
     
        #%%
##==============================================================================
## example of shift a position at (x,y)=(120.56, 120.64)
##==============================================================================
from galight.tools.astro_tools import plt_fits
image = pyfits.open('./test_img.fits')[0].data.copy()
plt_fits(image)
pos = [0.2,0.2]

img_class=Measure_asy(img=image, order=2)

result = img_class.find_pos(theta = [0,0])
residual = img_class.residual(result['x'])
plt_fits(residual)

print(result["x"])
print(residual.sum())

