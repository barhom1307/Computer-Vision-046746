# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:06:25 2020

@author: Avinoam
"""

import numpy as np
import os
from my_keypoint_det import DoGdetector, displayPyramid
import cv2
import matplotlib.pyplot as plt


def makeTestPattern(patchWidth, nbits):
    compareX = np.random.randint(0, patchWidth**2, nbits)
    compareY = np.random.randint(0, patchWidth**2, nbits)
    i = 0
    for idx_x,idx_y in zip(compareX, compareY):
        if idx_x == idx_y:    
            compareX[i] = compareX[i]+1
        i += 1
    return np.reshape(compareX, (nbits,1)), np.reshape(compareY,(nbits, 1))

def computeBrief(im, locsDoG, compareX, compareY):
    valid_points = []
    delta = int(patchWidth//2)
    for loc_point in locsDoG:
        if (loc_point[0] >= delta and loc_point[0] <= im.shape[0]-delta) \
            and (loc_point[1] >= delta and loc_point[1] <= im.shape[1]-delta):
                valid_points.append(loc_point)
    locs = np.reshape(np.array(valid_points),(len(valid_points),3)) 
    unique_locs = {(x,y):level for x,y,level in locs}
    locs = np.array([np.array([i[0],i[1],j]) for i,j in unique_locs.items()])
    desc = np.zeros((locs.shape[0],nbits))
    for idx,test_point in enumerate(locs):
        x, y, level = test_point
        patch = im[x-delta:x+delta+1, y-delta:y+delta+1]
        tmp_patch = patch.ravel()
        desc[idx] = np.array([int(tmp_patch[i]<tmp_patch[j]) for i,j in zip(compareX,compareY)])
    return locs, desc

def briefLite(im):
    sigma0 = 1
    k = np.sqrt(2)
    levels = [-1, 0, 1, 2, 3, 4]
    sigmaC = 0.03
    sigmaR = 12
    locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, sigmaC, sigmaR)
    if os.path.isfile('test_pattern.npy'):
        compareX, compareY = np.load('test_pattern.npy')
    else:
        compareX, compareY = makeTestPattern(patchWidth, nbits)
        np.save('test_pattern.npy', [compareX, compareY])
    locs, desc = computeBrief(im, locsDoG, compareX, compareY)
    return locs, desc

 
if __name__ == '__main__':
    patchWidth = 9
    nbits = 256
    
    im = cv2.imread(r'C:\Users\Avinoam\Desktop\data\pf_floor.jpg')
    # im = cv2.imread(r'C:\Users\Avinoam\Desktop\data\test_b.jpg')
    blur_im = cv2.GaussianBlur(im,(7,7),2)
    im_gray = cv2.cvtColor(blur_im, cv2.COLOR_BGR2GRAY)
    normalized_im = im_gray/255
    locs, desc = briefLite(normalized_im)

    # plot BRIEF Descriptor on original image
    cirList = []
    for x,y in zip(locs[:,0], locs[:,1]):
        cirList.append(plt.Circle((y, x), 2, color='r', fill=False))
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    _ = plt.axis('off')
    for cir in cirList:
        fig.add_subplot(111).add_artist(cir)    
    plt.show()

# =============================================================================
#     square_im = np.zeros((64,64))
#     square_im[10:40,10:40] = 1
#     square_locs, square_desc = briefLite(square_im)
#     cirList = []
#     for x,y in zip(square_locs[:,0], square_locs[:,1]):
#         cirList.append(plt.Circle((y, x), 1, color='r', fill=False))
#     fig = plt.figure()
#     plt.imshow(square_im, cmap='gray')
#     _ = plt.axis('off')
#     for cir in cirList:
#         fig.add_subplot(111).add_artist(cir)    
#     plt.show()
# 
# =============================================================================
