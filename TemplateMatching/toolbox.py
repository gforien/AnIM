#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nibabel as nib
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
from scipy.misc import imresize as imresize
from numpy import resize as resize_nd
"""
csv_path="/home/tom/Documents/AnIM/TemplateMatching/circle.csv"
tp="/home/tom/Documents/docPIR/circles/aTemplate.png"
i=9#6
ip="/home/tom/Documents/AnIM/TemplateMatching/circles/circles"+str(i)+".jpg"
f=read(ip)
t=read(tp)
pyramid=tuple(pyramid_gaussian(f, downscale=2, multichannel=False))
max_layer=len(pyramid)-1
"""
'''
def correlsan(f,t):
    corr=np.zeros_like(f.shape[0],f.shape[1],f.shape[2])
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            for z in range(f.shape[2]):
                fpatch=np.zeros_like((t.shape[0],t.shape[1]))
                #On doit tout mettre bien sur fpatch, à faire plus tard
                corr[x,y,z]=ifft2(fft2(fpatch)*np.conj(fft2(t)))
'''

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

def rec_pyr_3d(pyramid,layer,max_layer,image,template,ind):
    #Retrouver l'image associée au layer
    print("going at layer ",layer)
    couche=pyramid[layer]
    print("couche shape",couche.shape)
    #En fonction de ça on resize le template
    template_shape=(int(template.shape[0]*pyramid[layer].shape[0]/image.shape[0]),
                    int(template.shape[1]*pyramid[layer].shape[1]/image.shape[1]),
                    int(template.shape[2] * pyramid[layer].shape[2] / image.shape[2]))
    r_t=resize_nd(template,template_shape)

    coefx=sigma*r_t.shape[0]
    coefy=sigma*r_t.shape[1]
    coefz=sigma*r_t.shape[2]
    xinf=int(ind[0]*couche.shape[0]-coefx)
    xsup=int(ind[0]*couche.shape[0]+coefx)
    yinf=int(ind[1]*couche.shape[1]-coefy)
    ysup=int(ind[1]*couche.shape[1]+coefy)
    zinf = int(ind[2] * couche.shape[2] - coefz)
    zsup = int(ind[2] * couche.shape[2] + coefz)
    if xinf<0 or layer==max_layer-3: xinf=0
    if xsup>couche.shape[0] or layer==max_layer-3: xsup=couche.shape[0]-1
    if yinf<0 or layer==max_layer-3: yinf=0
    if ysup>couche.shape[1] or layer==max_layer-3: ysup=couche.shape[1]-1
    if zinf<0 or layer==max_layer-3: zinf=0
    if zsup>couche.shape[2] or layer==max_layer-3: zsup=couche.shape[2]-1
    #On correle en passant par fft
    corr=correlate(couche[xinf:xsup,yinf:ysup,zinf:zsup]-np.average(couche[xinf:xsup,yinf:ysup,zinf:zsup]), r_t-np.average(r_t),'same','fft')
    print("corr shape", corr.shape)
    print("image shape",image.shape)
    print("template shape", template.shape)
    #On cherche le max
    maxind = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    print("max found at abs", maxind)
    ind=(maxind[0]/couche.shape[0], maxind[1]/couche.shape[1], maxind[2]/couche.shape[2])
    print("max found at rel",ind)
    if layer == 0: return ind
    return rec_pyr_3d(pyramid,layer-1,max_layer,image,template,ind)

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

'''guess=(rec_pyr(max_layer-3,(0.5,0.5)))
print("terminé en",time.clock()-t_dep)
answer=read_nth_line(i+1)
print(get_offset(guess,answer,f.shape))
'''
def routine_3d():
    print("&&&&&&&&&&&&&&&&&&&&the big thing")
    templatepath="/home/tom/Documents/docPIR/average/1326.nii.gz"
    fullpath="/home/tom/Documents/docPIR/visceral/volumes/CTce_ThAb/10000100_1_CTce_ThAb.nii.gz"
    template=nib.load(templatepath)
    image=nib.load(fullpath)
    t_dep = time.clock()
    pyramid=tuple(pyramid_gaussian(image.get_fdata(), downscale=8, multichannel=False))
    print("pyramide construite en ",time.clock()-t_dep)
    max_layer=len(pyramid)
    #(pyramid, layer, max_layer, image, template, ind)
    maxind=rec_pyr_3d(pyramid,max_layer-3,max_layer ,image.get_fdata(),template.get_fdata(),(0.5,0.5,0.5))
    print("maxind",maxind)

if __name__=="__main__":
    routine_3d()
