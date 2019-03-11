#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas
import matplotlib.pyplot as plt
from skimage.io import imsave as save
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

pathCircle = os.getcwd()+"/circles/"
pathCloud= os.getcwd()+"/clouds/"
pathTemplate= os.getcwd()+"/templates/"
try:  
    os.mkdir(pathCircle)
    os.mkdir(pathCloud)
    os.mkdir(pathTemplate)
except OSError:  
    print ("Creation of a directory failed")


corruption=0.2#for corrupting randomly, must be between 0.0 and 1.0
i=0

listPosx = []
listPosy = []

for i in range(100):    
    img = np.zeros((500, 500), dtype=np.int16)
    # fill circle
    posx = random.randint(0,500)
    posy = random.randint(0,500)
    listPosx.append(posx)
    listPosy.append(posy)
    rr, cc = circle(posy, posx , 50+random.randint(-3,3), img.shape)
    img[rr, cc] =255
    save(pathCircle+"/circles"+str(i)+".jpg", img)

listIndex = [i for i in range(100)]
df = pandas.DataFrame({"posx": listPosx, "posy": listPosy}, listIndex)
df.to_csv("./circle.csv", sep=',',index=False)




for j in range(100):
    img = np.zeros((500, 500), dtype=np.int16)
    rr, cc = circle(100+random.randint(-10,10), 100+random.randint(-10,10), 50+random.randint(-3,3), img.shape)
    img[rr, cc] =255
    
    rr, cc = circle(160+random.randint(-10,10), 100+random.randint(-10,10), 20+random.randint(-3,3), img.shape)
    img[rr, cc] =255
    
    rr, cc = circle(190+random.randint(-10,10), 100+random.randint(-10,10), 30, img.shape)
    img[rr, cc] =255
    save(pathCloud+"clouds"+str(j)+".jpg", img)


#declaring the template to slide (here it's a circle)
tmplt = np.zeros((100, 100), dtype=np.int16)
rr, cc = circle(50,50, 50, tmplt.shape)
tmplt[rr, cc] =255
save(pathTemplate+"aTemplate.png", tmplt)


    