
import skimage.io as io
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import nibabel as nib
import matplotlib.patches as patches
from scipy.signal import correlate as correlate
from scipy.signal import fftconvolve as fftconvolve
import time
from skimage.transform import pyramid_gaussian,downscale_local_mean


import gc
from templateMaker import bbox2_3D,broadcast_2d,broadcast_3d,get_patient


def shotgun(image,template,factor):
    print("shotgun loading...")
    dim_factor=(factor ,factor,factor)
    print("image shape",image.shape)
    print("template shape",template.shape)
    resized_image=downscale_local_mean(image,dim_factor)
    resized_template=downscale_local_mean(template,dim_factor)
    print("resized image shape",resized_image.shape)
    print("resized template shape",resized_template.shape)
    nxcorr=fft_nxcorr(resized_image,resized_template)
    print("first shot taken !")
    maxind=np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
    print("corr max is ",np.amax(nxcorr))
    print("at",maxind)

    relative_index=(maxind[0]/resized_image.shape[0],
                    maxind[1]/resized_image.shape[1],
                    maxind[2]/resized_image.shape[2])
    abs_ind=(int(relative_index[0]*image.shape[0]),
             int(relative_index[1]*image.shape[1]),
             int(relative_index[2]*image.shape[2]))
    sigma = 1
    del (resized_image)
    del (resized_template)
    del (nxcorr)
    gc.collect()
    xinf, xsup = abs_ind[0] - int(sigma * image.shape[0] / 2), abs_ind[0] + int(sigma * image.shape[0] / 2)
    yinf, ysup = abs_ind[1] - int(sigma * image.shape[1] / 2), abs_ind[1] + int(sigma * image.shape[1] / 2)
    zinf, zsup = abs_ind[2] - int(sigma * image.shape[2] / 2), abs_ind[2] + int(sigma * image.shape[2] / 2)
    # We readjust the border just in case
    if xinf < 0: xinf = 0
    if yinf < 0: yinf = 0
    if zinf < 0: zinf = 0
    if xsup > image.shape[0]: xsup = image.shape[0] - 1
    if ysup > image.shape[1]: xsup = image.shape[1] - 1
    if zsup > image.shape[2]: xsup = image.shape[2] - 1
    print("RoI shape",image[xinf:xsup,yinf:ysup,zinf:zsup].shape)
    nxcorr = fft_nxcorr(image[xinf:xsup,yinf:ysup,zinf:zsup], template)
    maxind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
    print("second bam !")
    print("corr max is ", np.amax(nxcorr))
    print("at", maxind)
    display_square(image,template.shape,maxind)
    return maxind

def pyr(pyramid,template,layer,ind=(0.0,0.0,0.0)):
    print("couche ",layer)
    couche=pyramid[layer]
    print("\tcouche shape",couche.shape)
    print('relative ind passed to this layer is', ind )
    max_layer=len(pyramid)-1
    #Ratio in every dimension between image and template dimensions
    #NOTE : this doesn't change during the whole procedure
    ratios=(template.shape[0]/pyramid[0].shape[0],
            template.shape[1]/pyramid[0].shape[1],
            template.shape[2]/pyramid[0].shape[2])
    #We multiply by the image of layer's dims
    resized_shape=(int(ratios[0]*couche.shape[0]),
                   int(ratios[1]*couche.shape[1]),
                   int(ratios[2]*couche.shape[2]))
    downscaling_factor=(pyramid[0].shape[0] // pyramid[layer].shape[0],
                        pyramid[0].shape[1] // pyramid[layer].shape[1],
                        pyramid[0].shape[2] // pyramid[layer].shape[2])
    #Here's where we downscale
    r_t=downscale_local_mean(template,downscaling_factor)
    #display(couche, r_t)
    print("\tresied template shape",r_t.shape)
    print("\texected shape is",resized_shape)
    #We now have to calculate the borders region of interest
    #It is based on previous iterations (or not if it is the first one)
    if(layer==max_layer-3):
        #the layer is the maximum one we need togo through the whole image
        nxcorr = fft_nxcorr(f=couche, t=r_t)
        corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
        print("\t corr index",corrind)
        realind = (corrind[0], corrind[1], corrind[2])
    else:
        #We have to calculate what is the absolute index in our case
        abs_ind=(int(couche.shape[0]*ind[0]),
                 int(couche.shape[1]*ind[1]),
                 int(couche.shape[2]*ind[2]))
        print("\tabs ind",abs_ind)
        #We use a parameter sigma that will determine how big the region is
        #Let's establish that if sigma = 1 the RoI is of one template
        sigma=1.1
        xinf,xsup=abs_ind[0]-int(sigma*resized_shape[0]/2),abs_ind[0]+int(sigma*resized_shape[0]/2)
        yinf, ysup = abs_ind[1] - int(sigma * resized_shape[1] / 2), abs_ind[1] + int(sigma * resized_shape[1] / 2)
        zinf, zsup = abs_ind[2] - int(sigma * resized_shape[2] / 2), abs_ind[2] + int(sigma * resized_shape[2] / 2)
        #We readjust the border just in case
        if xinf<0:xinf=0
        if yinf<0:yinf=0
        if zinf<0:zinf=0
        if xsup> couche.shape[0]:xsup=couche.shape[0]-1
        if ysup > couche.shape[1]: xsup = couche.shape[1] - 1
        if zsup > couche.shape[2]: xsup = couche.shape[2] - 1
        print("\t RoI shape ",couche[xinf:xsup,yinf:ysup,zinf:zsup].shape)
        #We correlate with a view of the image rather than another object
        #to have the max in the coordinates of the image
        nxcorr=fft_nxcorr(f=couche[xinf:xsup,yinf:ysup,zinf:zsup],t=r_t)
        corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
        realind = (xinf + corrind[0], yinf + corrind[1], zinf + corrind[2])
        print("corr index",corrind)
        print("\tnxcorr max",np.amax(nxcorr))
        #display(nxcorr,r_t)
    # Find max in corr
    corrind = np.unravel_index(np.argmax(nxcorr, axis=None), nxcorr.shape)
    print("max in corr at",corrind)

    print("\trealind",realind)
    relative_ind=(realind[0] / couche.shape[0], realind[1] / couche.shape[1], realind[2] / couche.shape[2])
    if layer<=1:
        return ind
    return pyr(pyramid, template, layer - 1, relative_ind)

def fft_nxcorr(f,t):
    """
        This method is heavily influenced (and in some occasions copied)  by the code of

        Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>

        that can be found here :
        https://pastebin.com/x1NJqWWm

        referencing:
            Hermosillo et al 2002: Variational Methods for Multimodal Image
            Matching, International Journal of Computer Vision 50(3),
            329-343, 2002
            <http://www.springerlink.com/content/u4007p8871w10645/>

            Lewis 1995: Fast Template Matching, Vision Interface,
            p.120-123, 1995
            <http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html>

            <http://en.wikipedia.org/wiki/Cross-correlation#Normalized_cross-correlation>

    """
    t = np.float32(t)
    f = np.float32(f)

    std_t, mean_t,tsize= np.std(t), np.mean(t),t.size
    xcorr = correlate(f, t,'same','fft')

    """
    Rather than calculate integral tables to get the local sums, we will convolve by an array of ones, that has the shape
    of the template. using fft again
    """
    convolver=np.ones_like(t)
    ls_a=fftconvolve(f,convolver,'same')
    ls2_a = fftconvolve(f ** 2, convolver, 'same')

    # local standard deviation of the input array
    ls_diff = ls2_a - (ls_a ** 2) / tsize
    ls_diff = np.where(ls_diff < 0, 0, ls_diff)
    sigma_a = np.sqrt(ls_diff)

    # standard deviation of the template
    sigma_t = np.sqrt(t.size - 1.) * std_t

    # denominator: product of standard deviations
    denom = sigma_t * sigma_a

    # numerator: local mean corrected cross-correlation
    numer = (xcorr - ls_a * mean_t)

    # sigma_t cannot be zero, so wherever the denominator is zero, this must
    # be because sigma_a is zero (and therefore the normalized cross-
    # correlation is undefined), so set nxcorr to zero in these regions
    tol = np.sqrt(np.finfo(denom.dtype).eps)
    nxcorr = np.where(denom < tol, 0, numer / denom)

    # if any of the coefficients are outside the range [-1 1], they will be
    # unstable to small variance in a or t, so set them to zero to reflect
    # the undefined 0/0 condition
    nxcorr = np.where(np.abs(nxcorr - 1.) > np.sqrt(np.finfo(nxcorr.dtype).eps), nxcorr, 0)
    return nxcorr



def display(image,template):
    fig, [[ax1, ax2, ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, num='Result of Template Search')
    ax1.imshow(image[:, :, int(image.shape[2]/2)], interpolation='nearest')
    ax1.set_title('image XY')
    ax2.imshow(image[int(image.shape[0]/2), :, :], interpolation='nearest')
    ax2.set_title('image YZ')
    ax3.imshow(image[:,int(image.shape[1]/2), :], interpolation='nearest')
    ax3.set_title('image XZ')
    ax4.imshow(template[:, :, int(template.shape[2] / 2)], interpolation='nearest')
    ax4.set_title('template XY')
    ax5.imshow(template[int(template.shape[0] / 2), :, :], interpolation='nearest')
    ax5.set_title('template YZ')
    ax6.imshow(template[:, int(template.shape[1] / 2), :], interpolation='nearest')
    ax6.set_title('template XZ')
    plt.show()

def display_square(image,tshape, maxind,solution=None):
    x_t,y_t,z_t=tshape
    fig, [ax1 , ax2 , ax3] = plt.subplots(1, 3, num='Result of Template Search')
    rectxy = patches.Rectangle((maxind[1] - int(y_t / 2), maxind[0] - int(x_t / 2)), y_t, x_t, linewidth=1,
                             edgecolor='r', facecolor='none')
    rectyz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[1] - int(y_t / 2)), z_t, y_t, linewidth=1,
                               edgecolor='r', facecolor='none')
    rectxz = patches.Rectangle((maxind[2] - int(z_t / 2), maxind[0] - int(x_t / 2)), z_t, x_t, linewidth=1,
                               edgecolor='r', facecolor='none')

    ax1.imshow(image[:,:,maxind[2]], interpolation='nearest')
    ax1.add_patch(rectxy)
    ax2.imshow(image[maxind[0], :,:], interpolation='nearest')
    ax2.add_patch(rectyz)
    ax3.imshow(image[:, maxind[1], :], interpolation='nearest')
    ax3.add_patch(rectxz)
    if solution!=None:
        x_s,y_s,z_s=solution
        solxy = patches.Rectangle((maxind[1] - int(y_t / 2), x_s - int(x_t / 2)), y_t, x_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        solyz = patches.Rectangle((maxind[2] - int(z_t / 2), y_s - int(y_t / 2)), z_t, y_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        solxz = patches.Rectangle((maxind[2] - int(z_t / 2), z_s - int(x_t / 2)), z_t, x_t, linewidth=1,
                                   edgecolor='g', facecolor='none')
        ax1.add_patch(solxy)
        ax2.add_patch(solyz)
        ax3.add_patch(solxz)
    plt.show()

def load_images(fullpath='/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb/10000100_1_CTce_ThAb.nii.gz'
                ,templatepath='/media/tom/TOSHIBA EXT/PIR/100/58.nii.gz'):
    print("loading image", fullpath)
    image = nib.load(fullpath)
    print("loading template", templatepath)
    template = nib.load(templatepath)

    return image,template

def routine_3d():
    print("3d routine starting")
    image,template=load_images()
    #display(image.get_fdata(),template.get_fdata())

    image_shape=image.get_fdata().shape
    pyramid = tuple(pyramid_gaussian(image.get_fdata(), downscale=2, multichannel=False))
    max_layer = len(pyramid) - 1
    #template_rotate= np.rot90(template.get_fdata(), k=1)
    ind=pyr(pyramid=pyramid,template=template.get_fdata(),layer=max_layer-3)

    #ind=shotgun(image.get_fdata(),template.get_fdata(),3)
    print("index is",ind)
    inflated_index=(int(image.shape[0]*ind[0]),int(image.shape[1]*ind[1]),int(image.shape[2]*ind[2]))
    #display_square(image.get_fdata(),template.shape,inflated_index)

def test_classe(classe,templates_path='/media/tom/TOSHIBA EXT/PIR',results_path='media/tom/TOSHIBA EXT/PIR_RESULTATS',fullpath='/media/tom/TOSHIBA EXT/visceral/volumes/CTce_ThAb'):
    print("testing classe",classe)
    classe_template_name=str(classe)+".nii.gz"
    all_patients = os.listdir(templates_path)
    with open(str(classe)+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["patient", "i", "j","k"])
        for imagename in os.listdir(fullpath):

            image,template=load_images(fullpath=os.path.join(fullpath,imagename),templatepath=os.path.join(templates_path,get_patient(imagename),classe_template_name))
            pyramid = tuple(pyramid_gaussian(image.get_fdata(), downscale=2, multichannel=False))
            max_layer = len(pyramid) - 1
            ind = pyr(pyramid=pyramid, template=template.get_fdata(), layer=max_layer - 3)
            inflated_index = (int(image.shape[0] * ind[0]), int(image.shape[1] * ind[1]), int(image.shape[2] * ind[2]))
            writer.writerow([get_patient(imagename),inflated_index[0],inflated_index[1],inflated_index[2]])

def routine_2d():
    image = io.imread('/home/tom/Images/autres/berners.jpg', as_gray=True)
    template=image[270:300,850:900]
    i_row, i_col = image.shape
    print(image.shape)
    t_row, t_col = template.shape
    t_dep = time.clock()
    correlation = fft_nxcorr(image, template)
    maxind = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)
    print("correlation construite en ", time.clock() - t_dep)
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, num='ND Template Search')
    ax1.imshow(image, interpolation='nearest')
    ax1.set_title('Search image')
    ax2.imshow(template, interpolation='nearest')
    ax2.set_title('Template')
    print(maxind)
    ax3.imshow(correlation, interpolation='nearest')
    ax3.set_title('Normalized cross-correlation')
    rect = patches.Rectangle((maxind[1] - int(t_col / 2), maxind[0] - int(t_row / 2)), t_col, t_row, linewidth=1,
                             edgecolor='r', facecolor='none')
    ax3.add_patch(rect)
    plt.show()

if __name__=='__main__':
    #routine_3d()
    #routine_2d()
    #perfo_test()
    test_classe(58)

