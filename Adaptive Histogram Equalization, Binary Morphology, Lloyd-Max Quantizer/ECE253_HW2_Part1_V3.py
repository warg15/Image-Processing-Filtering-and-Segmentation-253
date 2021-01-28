#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:20:05 2020

@author: femw90
"""
##William Argus A12802324
## ECE 253 HW1
'''
Academic Integrity Policy: Integrity of scholarship is essential for an academic commu-
nity. The University expects that both faculty and students will honor this principle and
in so doing protect the validity of University intellectual work. For students, this means
that all academic work will be done by the individual to whom it is assigned, without
unauthorized aid of any kind.
By including this in my report, I agree to abide by the Academic Integrity Policy men-
tioned above.
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot 
import math
import cv2

def AHE(im, win_size):
    half = int(win_size/2)
    imPadded = np.pad(im, half, mode = 'symmetric')
    sz = np.shape(im)
    output = np.zeros(np.shape(im))
    for x in range(half, sz[0]+half):
        #print(x)
        for y in range(half, sz[1]+half):
            rank = 0
            for i in range(x-half, x+half):
                for j in range(y-half, y+half):
                    if(imPadded[x,y] > imPadded[i,j]): rank += 1
            output[x-half,y-half] = rank*(255/(win_size**2))
    return output


#set-up call to function
#image beach.png for win size = 33; 65 and 129
win_size = [33, 65, 129]
im = plt.imread('Beach.png')*255


#code to easily downsize without having to change any parameters other than "downsize"
from skimage.transform import rescale
downSize = 1
im = rescale(im, 1/downSize, anti_aliasing=False)
win_size = [33/downSize, 65/downSize, 129/downSize]
win_size = [33, 65, 129]

plt.figure(1)
plt.title('Original Image')
plt.imshow(im, cmap = 'gray')

rows = 2
cols = 2
axes=[]
fig=plt.figure(2)


for a in range(len(win_size)):
    print(a)
    b = AHE(im, win_size[a])
    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=("AHE, win_size: "+str(win_size[a]))
    axes[-1].set_title(subplot_title)  
    plt.imshow(b, cmap = 'gray')
fig.tight_layout()    
plt.show()

axesB=[]
figB=plt.figure(3)
img = cv2.imread('Beach.png',0)
#img = im

b = cv2.equalizeHist(img)
axesB.append( figB.add_subplot(1, 1, 1) )
subplot_title=("HE")
axesB[-1].set_title(subplot_title)  
plt.imshow(b, cmap = 'gray')
figB.tight_layout()    
plt.show()

'''
for a in range(len(win_size)):
    #print(a)
    b = cv2.equalizeHist(img)
    axesB.append( figB.add_subplot(rows, cols, a+1) )
    subplot_title=("HE, win_size: "+str(win_size[a]))
    axesB[-1].set_title(subplot_title)  
    plt.imshow(b, cmap = 'gray')
figB.tight_layout()    
plt.show()
'''