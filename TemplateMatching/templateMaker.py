#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import os

segpath="/home/tom/Documents/docPIR/visceral/segmentations/CTce_ThAb/"
fullpath="/home/tom/Documents/docPIR/visceral/volumes/CTce_ThAb"
segfiles = os.listdir( segpath )#list of all files in working dir
typicalfilename="10000100_1_CTce_ThAb_2_0.nii.gz"
classtolabel={} #sidequest
patients=["100","104","105","106","108","109","110","111","112","113","127","128","129","130","131","132","133","134","135","136"]
#patients=["100","104","105","106","108"]
classes=["1247","1302","1326","170","187" ,"237","2473" ,"29193" ,"29662" ,"29663" ,"30324" ,"30325" ,"32248" ,"32249" ,"40357" ,"40358" ,"480" ,"58" ,"7578" ,"86" ,"0" ,"1","2" ]
#classes=["1247","1302","1326","170"]
def get_class(filename):
    arr=filename.split("_")
    return arr[4]
def get_patient(filename):
    return filename[5:8]

def merge(nib_mask,nib_img):#merges two nib images and returns a NP array !!!
    mask_arr=nib_mask.get_fdata()
    img_arr=nib_img.get_fdata()
    print("merging")
    merge=np.zeros(mask_arr.shape)
    for x in range(mask_arr.shape[0]):
        for y in range(mask_arr.shape[1]):
            for z in range(mask_arr.shape[2]):
                if mask_arr[x,y,z]!=0:
                    merge[x,y,z]=img_arr[x,y,z]
    return merge

def empty_array(arr2d):
    for i in np.nditer(arr2d):
        if i!=0:
            return False 
    return True 

def englob(img):
    print("englob")
    xinf=0
    while(img[xinf,:,:].any()==False):
        xinf+=1
    xsup=img.shape[0]-1
    while(img[xsup,:,:].any()==False):
        xsup-=1
    print("englob x done")
    yinf=0
    while(img[:,yinf,:].any()==False):
        yinf+=1
    ysup=img.shape[1]-1
    while(img[:,ysup,:].any()==False):
        ysup-=1
    print("englob y done")
    zinf=0
    while(img[:,:,zinf].any()==False):
        zinf+=1
    zsup=img.shape[2]-1
    while(img[:,:,zsup].any()==False):
        zsup-=1
    print("englob z done")
    return img[xinf:xsup,yinf:ysup,zinf:zsup]

def add_two(arr1,arr2):
    print("adding")
    xmax=min(arr1.shape[0],arr2.shape[0])
    ymax=min(arr1.shape[1],arr2.shape[1])
    zmax=min(arr1.shape[2],arr2.shape[2])
    res=np.zeros((xmax,ymax,zmax))
    for x in range(xmax):
        for y in range(ymax):
            for z in range(zmax):
                res[x,y,z]=arr1[x,y,z]+arr2[x,y,z]
    return res
                
def make_templates(excluded_patient):
    #TODO add patient path
    valid_dirs=[]
    for directory in segfiles:
        if get_patient(directory)!=excluded_patient and get_patient(directory) in patients:
            valid_dirs.append(directory)
            print("found new file",directory)
    for classe in classes:
        print("starting class",classe)
        sumofall=np.zeros((512,512,512))
        nb=0
        for directory in valid_dirs:
            if(get_class(directory)==classe):
                nb+=1
                print("\t patient"+get_patient(directory))
                vol_filename="10000"+get_patient(directory)+"_1_CTce_ThAb.nii.gz"
                #Join path and filenames(too lazy to do it right)
                vol_dir=os.path.join(fullpath,vol_filename)
                seg_dir=os.path.join(segpath,directory)
                #load images with nibabels
                mask=nib.load(seg_dir)
                full=nib.load(vol_dir)
                #merge, englob, and append to list
                mer=merge(mask,full)
                print("size of mer",mer.shape)
                mer=englob(mer)
                print("size of englob",mer.shape)
                sumofall=add_two(sumofall,mer)
                sumofall=englob(sumofall)
        sumofall=np.dot(1/nb,sumofall)
        print("saving")
        path_to_save=os.path.join('/home/tom/Documents/docPIR/average',classe+'.nii.gz')
        img = nib.Nifti1Image(sumofall, np.eye(4))
        img.to_filename(path_to_save)
        #nib.save(sumofall,path_to_save )
        print("done with ",classe)

make_templates("100")
            
