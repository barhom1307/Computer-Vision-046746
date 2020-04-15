# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:22:20 2020

@author: Avinoam
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import maximum_filter

def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor( 3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im,(size,size),sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)

def displayPyramid(pyramid):
    plt.figure(figsize=(16,5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')

def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # Gaussian Pyramid - A matrix of grayscale images of size
    # (len(levels), shape(im))
    # levels - the levels of the pyramid where the blur at each level is
    # outputs
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    # created by differencing the Gaussian Pyramid input
    DoGPyramid = []
    for level in range(1, len(levels)):
        DoGPyramid.append(GaussianPyramid[level] - GaussianPyramid[level-1])
    DoGPyramid = np.stack(DoGPyramid)
    return DoGPyramid

def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    # Takes in DoGPyramid generated in createDoGPyramid and returns
    # PrincipalCurvature,a matrix of the same size where each point contains the
    # curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    # INPUTS
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    # OUTPUTS
    # PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    # point contains the curvature ratio R for the
    # corresponding point in the DoG pyramid
    PrincipalCurvature = []
    eps = 1e-5
    for dog_level in DoGPyramid:
        # dog_level = (dog_level-np.min(dog_level))/(np.max(dog_level)-np.min(dog_level))
        gxx = cv2.Sobel(dog_level, cv2.CV_64F, 2, 0, ksize=3).astype(float)
        gyy = cv2.Sobel(dog_level, cv2.CV_64F, 0, 2, ksize=3).astype(float)
        gxy = cv2.Sobel(dog_level, cv2.CV_64F, 1, 1, ksize=3).astype(float)
        trace = gxx + gyy
        det = (gxx*gyy) - (gxy**2)
        R = (trace**2) / (det + eps)
        PrincipalCurvature.append(R)
    return np.stack(PrincipalCurvature)

def getLocalExtrema(DoGPyramid, PrincipalCurvature, th_contrast, th_r):
    mask_r = PrincipalCurvature.astype(float) < th_r
    mask_c = np.abs(DoGPyramid.astype(float)) > th_contrast
    both_th_mask = np.bitwise_and(mask_r, mask_c)
    locsDoG = []
    for i in range(len(DoGPyramid)):
        max_vals = maximum_filter(DoGPyramid[i], size=(3,3))
        if i==0:
            max_mask = (max_vals-DoGPyramid[i+1] > 0)
        elif i==len(DoGPyramid)-1:
            max_mask = (max_vals-DoGPyramid[i-1] > 0)
        else:
            max_mask = np.bitwise_and(max_vals-DoGPyramid[i+1] > 0, max_vals-DoGPyramid[i-1] > 0)
        max_same_level = (max_vals == DoGPyramid[i]) 
        total_mask = np.bitwise_and(np.bitwise_and(both_th_mask[i], max_mask), max_same_level)
        valid_idx = np.argwhere(total_mask)
        level_col = np.expand_dims(np.array([i]*len(valid_idx)), 1)
        locsDoG.append(np.hstack((valid_idx, level_col)))
    return np.concatenate(locsDoG, axis=0).astype('int')

def DoGdetector(im, sigma0, k, levels, th_contrast, th_r):
    # Putting it all together
    # Inputs Description
    # --------------------------------------------------------------------------
    # im Grayscale image with range [0,1].
    # sigma0 Scale of the 0th image pyramid.
    # k Pyramid Factor. Suggest sqrt(2).
    # levels Levels of pyramid to construct. Suggest -1:4.
    # th_contrast DoG contrast threshold. Suggest 0.03.
    # th_r Principal Ratio threshold. Suggest 12.
    # Outputs Description
    # --------------------------------------------------------------------------
    # locsDoG N x 3 matrix where the DoG pyramid achieves a local extrema
    # in both scale and space, and satisfies the two thresholds.
    # gauss_pyramid A matrix of grayscale images of size (len(levels),imH,imW)
    GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoGPyramid = createDoGPyramid(GaussianPyramid, levels)
    # displayPyramid(DoGPyramid)
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(DoGPyramid, PrincipalCurvature, th_contrast, th_r)
    return locsDoG, GaussianPyramid
