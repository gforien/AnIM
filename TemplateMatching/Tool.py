#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust tool for 2D image processing, template matching especially, involving template,
image, and confidence plot
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.io import imread as read 
from skimage.io import imsave as save


def sad(x,y):
    sad=0
    for u in range(t_array.shape[0]):
        for v in range(t_array.shape[1]):
            sad+=abs(i_array[x+u,x+v]-t_array[u,v])
    return sad


t_depart=time.clock()
#path to images and templates
i_path = os.getcwd()+"/circles/"
t_path = os.getcwd()+"/templates/"
#file names (TODO could be args in cmd)
i_file_name="circles0.png"
t_file_name="aTemplate.png"
#Creation of path for result images
r_path=i_path+"/"+i_file_name+"Result/"
try:  
    os.mkdir(r_path)
except OSError:  
    print ("Creation of reception directory failed")
#get numpy arrays of image and template
i_array=read(i_path+i_file_name)
t_array=read(i_path+t_file_name)#be careful about path
result_array=np.zeros((i_array.shape[0],i_array.shape[1]), dtype=np.double)
#Show images to user
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
ax1.imshow(i_array, cmap=plt.cm.gray, interpolation='nearest')
ax1.set_title('Image')
ax2.imshow(t_array, cmap=plt.cm.gray, interpolation='nearest')
ax2.set_title('Template')
plt.show()
'''
for x in range(int(t_array.shape[0]/2),int(i_array.shape[0]-t_array.shape[0]/2)-1):
    for y in range(int(t_array.shape[1]/2),int(i_array.shape[1]-t_array.shape[0]/2)-1):
        if i_array[x,y,:].any():
            result_array[x,y,:]=sad(x,y)
'''  
'''

‘reflect’ (d c b a | a b c d | d c b a)

    The input is extended by reflecting about the edge of the last pixel.
‘constant’ (k k k k | a b c d | k k k k)

    The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.
‘nearest’ (a a a a | a b c d | d d d d)

    The input is extended by replicating the last pixel.
‘mirror’ (d c b | a b c d | c b a)

    The input is extended by reflecting about the center of the last pixel.
‘wrap’ (a b c d | a b c d | a b c d)

    The input is extended by wrapping around to the opposite edge.

'''
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(10, 6))
constant_array=ndimage.convolve(i_array, t_array, mode='constant', cval=0.0)
print("constant completed at ",time.clock()-t_depart,"s")
reflect_array=ndimage.convolve(i_array, t_array, mode='reflect')
print("reflect completed at ",time.clock()-t_depart,"s")
mirror_array=ndimage.convolve(i_array, t_array, mode='mirror')
print("mirror completed at ",time.clock()-t_depart,"s")
wrap_array=ndimage.convolve(i_array, t_array, mode='wrap')
print("wrap completed at ",time.clock()-t_depart,"s")
#display
ax1.imshow(constant_array, cmap=plt.cm.gray, interpolation='nearest')
ax1.set_title('constant')    
ax2.imshow(reflect_array, cmap=plt.cm.gray, interpolation='nearest')
ax2.set_title('reflect')    
ax3.imshow(mirror_array, cmap=plt.cm.gray, interpolation='nearest')
ax3.set_title('mirror')    
ax4.imshow(mirror_array, cmap=plt.cm.gray, interpolation='nearest')
ax4.set_title('wrap')   
plt.show()
save(r_path+'constant.png',constant_array)
save(r_path+'reflect.png',reflect_array)
save(r_path+'mirror.png',mirror_array)
save(r_path+'wrap.png',wrap_array)
print("completed in ",time.clock()-t_depart,"s")