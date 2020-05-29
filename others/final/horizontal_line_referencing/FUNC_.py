from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats

################################################################################

def average_profile(theta_start, theta_end, delta_theta, xc, yc, s, img):

    k1 = len(np.arange(theta_start, theta_end, delta_theta))
    k2 = len(range(0,s))

    haha = np.zeros(s)
    haha1 = np.zeros(s)
    count = -1

    for k in np.arange(theta_start, theta_end, delta_theta):
        count = count + 1
        theta = k
        b = image_profile(img, xc, yc, theta, s)
        haha = haha + b

    haha = haha/count

    haha1 = haha

    return haha1

################################################################################

################################################################################

def image_profile(img_in, xc, yc, theta, s):

    profile_out = np.zeros(s)

    for j in range(0,s):
        xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
        yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
        profile_out[j] = img_in[yy,xx]

    return profile_out

################################################################################

################################################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

################################################################################

################################################################################

def srgbtoxyz(sRGB):
    M = np.matrix([[0.4887180, 0.3106803, 0.2006017],[0.1762044, 0.8129847, 0.0108109],[0.0000000, 0.0102048, 0.9897952]])
    XYZ = M * np.vstack(sRGB)

    return XYZ

################################################################################

################################################################################

def RGBnorm2RGBprime(a):
    if a <= 0.04045:
        b = a/12.92
    else:
        b = np.power((a+0.055)/1.055, 2.4)
    return b

################################################################################

################################################################################

def XYZnormtolab(X_norm, Y_norm, Z_norm):

    def func(p):
        if p > np.power(6/29, 3):
            q = np.power(p, 1/3)
        else:
            q = ((841/108)*p) + (4/29)
        return q

    L = (116*func(Y_norm)) - 16
    a = 500*(func(X_norm) - func(Y_norm))
    b = 200*(func(Y_norm) - func(Z_norm))

    lab = np.zeros((1,1,3))

    lab[0,0,0] = L
    lab[0,0,1] = a
    lab[0,0,2] = b

    return lab

################################################################################

################################################################################

def sRGBtoLab(sRGB, bit_depth, XYZ_illumination, Lab_illumination):

    def func(p):
        if p > np.power(6/29, 3):
            q = np.power(p, 1/3)
        else:
            q = ((841/108)*p) + (4/29)
        return q

    size = np.shape(sRGB)

    RGB_norm  = np.zeros(size)
    RGB_prime = np.zeros(size)
    XYZ       = np.zeros(size)
    XYZ_norm  = np.zeros(size)
    Lab       = np.zeros(size)

    RGB_norm = sRGB/((2**bit_depth)-1)

    mask_RGB = RGB_norm > 0.04045
    RGB_prime[mask_RGB]  = np.power((RGB_norm[mask_RGB]+0.055)/1.055, 2.4)
    RGB_prime[~mask_RGB] = RGB_norm[~mask_RGB]/12.92

    if   XYZ_illumination == 'D50':
        RGBtoXYZ = np.array([[0.4360747, 0.3850649, 0.1430804], \
                             [0.2225045, 0.7168786, 0.0606169], \
                             [0.0139322, 0.0971045, 0.7141733]])
    elif XYZ_illumination == 'D65':
        RGBtoXYZ = np.array([[0.4124564, 0.3575761, 0.1804375], \
                             [0.2126729, 0.7151522, 0.0721750], \
                             [0.0193339, 0.1191920, 0.9503041]])
    for i in range(size[0]):
        for j in range(size[1]):
            XYZ[i,j] = np.dot(RGBtoXYZ, RGB_prime[i,j].T)

    if   Lab_illumination == 'D50':
        XYZ_reference = np.array([[96.422],[100.000],[082.521]])/100.000
    elif Lab_illumination == 'D65':
        XYZ_reference = np.array([[95.047],[100.000],[108.883]])/100.000

    XYZ_norm = XYZ/XYZ_reference.T

    for i in range(size[0]):
        for j in range(size[1]):
            Lab[i,j,0] = (116*func(XYZ_norm[i,j,1])) - 16
            Lab[i,j,1] = 500*(func(XYZ_norm[i,j,0]) - func(XYZ_norm[i,j,1]))
            Lab[i,j,2] = 200*(func(XYZ_norm[i,j,1]) - func(XYZ_norm[i,j,2]))

    return Lab

################################################################################

################################################################################

def find_center(img, threshold):
    gray = rgb2gray(img)
    binary = gray > threshold

    plt.subplot(1,2,1)
    plt.imshow(gray, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.imshow(binary, cmap=plt.get_cmap('gray'))

    plt.show()

    return None

################################################################################

################################################################################

def generate_axisymmetric(ref_R, ref_G, ref_B, radius):

    sq_side = int(round(radius/np.sqrt(2)))-1
    img_axisymmetric = np.zeros((2*sq_side+1, 2*sq_side+1, 3))
    count_x = -1

    for i in range(-sq_side, sq_side+1):
        count_x = count_x + 1
        count_y = -1
        for j in range(-sq_side, sq_side+1):
            count_y = count_y + 1
            dist = int(round(np.sqrt((np.power(i,2)) + (np.power(j,2)))))
            img_axisymmetric[count_y,count_x,0] = ref_R[dist]
            img_axisymmetric[count_y,count_x,1] = ref_G[dist]
            img_axisymmetric[count_y,count_x,2] = ref_B[dist]

    return img_axisymmetric

################################################################################
