#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:27:39 2019

@author: tom
"""

import easygui
import os
from skimage.io import imread as read 

def main_routine():
    easygui.msgbox("Please choose directory of files to analyse")
    path=easygui.fileopenbox()
    print(path)
    try:
        imageArray=read(path)
    except:
        easygui.exceptionbox()
    reply=easygui.boolbox(msg='Is this the right image?', title=' ', choices=('[Y]es', '[N]o'), image=path, default_choice='Yes', cancel_choice='No')
    while reply=='[N]o':
        path=easygui.fileopenbox()
    choices=["Correlate","FFT Correlate","Freq"]
    choice=easygui.choicebox("Please choose method of extraction :",choices=choices)
    if(choice==choices[0]):
        pass
    if(choice==choices[1]):
        pass
    else:
        pass

main_routine()