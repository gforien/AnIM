#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave as save
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

path = os.getcwd()+"/circles/"
try:  
    os.mkdir(path)
except OSError:  
    print ("Creation of the directory %s failed" % path)

fig, (ax2) = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))


corruption=0.2#for corrupting randomly, must be between 0.0 and 1.0
i=0
for i in range(100):
    
    img = np.zeros((500, 500), dtype=np.int16)
    '''
    rr, cc, val = circle_perimeter_aa(20+random.randint(-5,5), 30+random.randint(-1,10), 30+random.randint(-1,15))
    img[rr, cc] = val
    '''
    # fill circle
    rr, cc = circle(random.randint(0,500), random.randint(0,500), 50+random.randint(-3,3), img.shape)
    img[rr, cc] =255
    '''    
    rr, cc, val = circle_perimeter_aa(50+random.randint(-15,15), 70+random.randint(-10,10),5+random.randint(-4,3))
    img[rr, cc] = val
     '''   
    ax2.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title('Created image')
    ax2.axis('off')
        
    plt.show()
    save(path+"/circles"+str(i)+".png", img)

#declaring the template to slide (here it's a circle)
tmplt = np.zeros((100, 100), dtype=np.int16)
rr, cc = circle(50,50, 50, tmplt.shape)
tmplt[rr, cc] =255
save(path+"aTemplate.png", tmplt)
