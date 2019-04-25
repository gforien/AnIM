#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust tool for 2D image processing, template matching especially, involving template,
image, and confidence plot
"""
import os
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from skimage.io import imread as read 
from skimage.io import imsave as save
from skimage.transform import pyramid_gaussian 
from scipy.signal import correlate as correlate
from scipy.misc import imresize as imresize

def timer(fonction,args):
    t_depart=time.clock()
    fonction(*args)
    print("fonction:"+fonction+" terminée en ",time.clock()-t_depart,"s")

#path to images and templates
i_path = os.getcwd()+"/circles/"
t_path = os.getcwd()+"/templates/"
#file names (TODO could be args in cmd)
i_file_name="circles1.png"
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
#Relative dimensions that template and images have
dimensions=[t_array.shape[0]/i_array.shape[0],t_array.shape[1]/i_array.shape[1]]

'''
FAST NCC as according to J.P. Lewis and Karl Briechle papers on the matter
it is argued that to compute a NCC
'''

def sum_tables(im):
    t_depart=time.clock()
    s=np.zeros_like(im)
    sq=np.zeros_like(im)
    for u in range(1,im.shape[0]):
        for v in range(1,im.shape[1]):
            s[u,v]=im[u,v]+s[u-1,v]+s[u,v-1]-s[u-1,v-1]
            sq[u,v]=im[u,v]*im[u,v]+s[u-1,v]*s[u-1,v]+s[u,v-1]*s[u,v-1]-s[u-1,v-1]*s[u-1,v-1]
    print("sum computations ",time.clock()-t_depart,"s")
    return (s,sq)
    
def fast_ncc(im,tm):
    print("////start of fast ncc")
    
    

def fft_corr():
    corr_array=correlate(i_array, t_array,'same','fft')
    ind = np.unravel_index(np.argmax(corr_array, axis=None), corr_array.shape)
    print(ind[0]/i_array.shape[0],ind[1]/i_array.shape[1])
    fig, (ax1) = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
    ax1.set_title("fft correlation result")
    ax1.imshow(corr_array, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    
def pyramid_without_opti(image):
    pyramidImg= tuple(pyramid_gaussian(image, downscale=2, multichannel=True))
    max_layer=len(pyramidImg)
    layer=0
    downsampling_factor=1
    while layer<max_layer:
        print("layer",layer)
        down_img=skimage.img_as_uint(pyramidImg[layer])#convert image from float to int
        template_shape=(int(t_array.shape[0]*pyramidImg[layer].shape[0]/i_array.shape[0]),
                        int(t_array.shape[1]*pyramidImg[layer].shape[1]/i_array.shape[1]))
        if template_shape[0]>0 and template_shape[0]>0:
            resized_template=imresize(t_array,template_shape)
            save(r_path+'resized'+str(layer)+'.png',resized_template)
            corr_array=correlate(down_img, resized_template,'same','fft')
            ind = np.unravel_index(np.argmax(corr_array, axis=None), corr_array.shape)
            print("max at ind ///abs",ind[0],ind[1],"///rel",ind[0]/down_img.shape[0],ind[1]/down_img.shape[1],"///val",corr_array[ind])
            fig, (ax1,ax2,ax3)= plt.subplots(ncols=3, nrows=1, figsize=(10, 6))
            ax1.set_title("Layer "+str(layer)+' :image, template and correlation result')
            ax1.imshow(pyramidImg[layer], cmap=plt.cm.gray, interpolation='nearest')
            ax2.imshow(resized_template, cmap=plt.cm.gray, interpolation='nearest')
            ax3.imshow(corr_array, cmap=plt.cm.gray, interpolation='nearest')
            plt.show()
            downsampling_factor*=2
            layer+=1
            save(r_path+'downsampled'+str(layer)+'.png',pyramidImg[layer])
            save(r_path+'result'+str(layer)+'.png',corr_array)
        else:
            break

def getRoi(layer,ind):#Computes region of interest given layer, and index of max correlation found in previous layer
    down_img=skimage.img_as_uint(pyramidImg[layer])#convert image from float to int
    coef=getTemplateShape(layer)[0]#Marche si image carrée
    xinf=int(ind[2]*(down_img.shape[0]-coef))
    xsup=int(ind[2]*(down_img.shape[0]+coef))
    if xinf<0 : xinf=0
    if xinf>down_img.shape[0] : xinf=down_img.shape-1
    print("bornes x",xinf,xsup)
    roi=down_img[range(xinf,xsup),:]
    return roi
'''
def getMasked(layer,ind):
    down_img=skimage.img_as_uint(pyramidImg[layer])#convert image from float to int
    maskroi=np.ones(down_img.shape)
    coef=2*getTemplateShape(layer)[0]
    xinf=int(ind[2]*(down_img.shape[0]-coef))
    xsup=int(ind[2]*(down_img.shape[0]+coef))
    yinf=int(ind[2]*(down_img.shape[1]-coef))
    ysup=int(ind[2]*(down_img.shape[1]+coef))
    maskroi[range(xinf,xsup),range(yinf,ysup)]=0
    print(maskroi)
    roi = ma.masked_array(down_img, mask=maskroi)
    return roi
'''
def getTemplateShape(layer):#Computes the shape the template should have for resizing
    template_shape=(int(t_array.shape[0]*pyramidImg[layer].shape[0]/i_array.shape[0]),
                    int(t_array.shape[1]*pyramidImg[layer].shape[1]/i_array.shape[1]))
    print("template shape : ",template_shape)
    return template_shape

def find_max(arr):
    ind = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    indrel=(ind[0],ind[1],ind[0]/arr.shape[0],ind[1]/arr.shape[1])
    print("max at index abs",indrel[0],indrel[1],'rel',indrel[2],indrel[3],'val',arr[indrel[0],indrel[1]])
    return indrel

def pyramid_recursive(layer,ind):
    print("////////////layer",layer)
    template_shape=getTemplateShape(layer)
    if template_shape[0]==0 or template_shape[0]==0:
        return ind
    resized_template=imresize(t_array,template_shape)
    print("ind",ind)
    roi=getRoi(layer,ind)
    corr_array=correlate(roi, resized_template,'same')
    coef=getTemplateShape(layer)[0]#Marche si image carrée
    xinf=int(ind[2]*(down_img.shape[0]-coef))
    xsup=int(ind[2]*(down_img.shape[0]+coef))
    if xinf<0 : xinf=0
    if xinf>down_img.shape[0] : xinf=down_img.shape-1
    ind=find_max(corr_array)
    if layer-1==0:
        return ind
    return pyramid_recursive(layer-1, ind)
          
#path to images and templates
i_path = os.getcwd()+"/circles/"
t_path = os.getcwd()+"/templates/"
#file names (TODO could be args in cmd)
i_file_name="circles1.png"
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
#Relative dimensions that template and images have
dimensions=[t_array.shape[0]/i_array.shape[0],t_array.shape[1]/i_array.shape[1]]
"""
#Pyramids
pyramidImg = tuple(pyramid_gaussian(i_array, downscale=2, multichannel=False))
pyramidTemplate=tuple(pyramid_gaussian(t_array, downscale=2, multichannel=False))
max_layer=len(pyramidImg)-1
#Show images to user
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
ax1.imshow(i_array, cmap=plt.cm.gray, interpolation='nearest')
ax1.set_title('Image')
ax2.imshow(t_array, cmap=plt.cm.gray, interpolation='nearest')
ax2.set_title('Template')
plt.show()
  

print("starting tests")
timer(pyramid_recursive, (max_layer-3,(250,250)))
#print(pyramid_recursive(max_layer-3,(250,250,0.5,0.5)))
#pyramid_recursive(max_layer-3,[250,250,0.5,0.5])
def main():
    
    image = "python_and_check_logo.gif"
    msg = "Do you like this picture?"
    choices = ["Yes","No","No opinion"]
    reply = buttonbox(msg, image=image, choices=choices)
"""