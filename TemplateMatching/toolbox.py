#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import csv
import numpy.ma as ma
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from skimage.io import imread as read 
from skimage.io import imsave as save
from skimage.transform import pyramid_gaussian 
from scipy.signal import correlate as correlate
from scipy.signal import ifft2 as ifft2
from scipy.signal import fft2 as fft2
from scipy.misc import imresize as imresize

csv_path="/home/tom/Documents/AnIM/TemplateMatching/circle.csv"
tp="/home/tom/Documents/docPIR/circles/aTemplate.png"
i=9#6
ip="/home/tom/Documents/AnIM/TemplateMatching/circles/circles"+str(i)+".jpg"
f=read(ip)
t=read(tp)
pyramid=tuple(pyramid_gaussian(f, downscale=2, multichannel=False))
max_layer=len(pyramid)-1

def correlsan(f,t):
    corr=np.zeros_like(f.shape[0],f.shape[1],f.shape[2])
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            for z in range(f.shape[2]):
                fpatch=np.zeros_like((t.shape[0],t.shape[1]))
                #On doit tout mettre bien sur fpatch, à faire plus tard
                corr[x,y,z]=ifft2(fft2(fpatch)*np.conj(fft2(t)))
            
def rec_pyr(layer,ind):
    #Retrouver l'image associée au layer
    couche=pyramid[layer]
    #En fonction de ça on resize le template 
    template_shape=(int(t.shape[0]*pyramid[layer].shape[0]/f.shape[0]),
                    int(t.shape[1]*pyramid[layer].shape[1]/f.shape[1]))
    rt=imresize(t,template_shape)
    #On définit les bornes d'intérêt(max si layer max)
    sigma=2.3
    coefx=sigma*rt.shape[0]
    coefy=sigma*rt.shape[1]
    xinf=int(ind[0]*couche.shape[0]-coefx)
    xsup=int(ind[0]*couche.shape[0]+coefx)
    yinf=int(ind[1]*couche.shape[1]-coefy)
    ysup=int(ind[1]*couche.shape[1]+coefy)

    if xinf<0 or layer==max_layer-3: xinf=0
    if xsup>couche.shape[0] or layer==max_layer-3: xsup=couche.shape[0]-1
    if yinf<0 or layer==max_layer-3: yinf=0
    if ysup>couche.shape[1] or layer==max_layer-3: ysup=couche.shape[1]-1
    #On correle en passant par fft
    corr=correlate(couche[xinf:xsup,yinf:ysup], rt,'same','fft')
    #On cherche le max
    maxind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    ind=(maxind[0]/couche.shape[0],maxind[1]/couche.shape[1])
    #print("max found at",ind)
    if layer==0: return ind
    return rec_pyr(layer-1, ind)

def read_nth_line(n):
    csvFile=open('circle.csv', 'r')
    reader = csv.reader(csvFile)
    i=0
    while(i<n+1):#because there is a title line and circles start at 0
        data=next(reader)
        i+=1
    csvFile.close()
    return data
    
def get_offset(reltuple,abstuple,imshape):#first tuple(the guess one) has relative coords
    print("answer",abstuple)
    print("guess",int(reltuple[0]*imshape[0]),int(reltuple[1]*imshape[1]))
    #offset=(abs(int(reltuple[0]*imshape[0])-abstuple[0]),abs(int(reltuple[1]*imshape[1])-abstuple[1]))
    return 0
t_dep=time.clock()

guess=(rec_pyr(max_layer-3,(0.5,0.5)))
print("terminé en",time.clock()-t_dep)
answer=read_nth_line(i+1)
print(get_offset(guess,answer,f.shape))
