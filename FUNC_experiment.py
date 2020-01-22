from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats
from skimage import color
from skimage import io
import re
import math
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.transform import hough_circle
from skimage.draw import circle_perimeter
from skimage.graph import route_through_array

########################################

def color_8bit(rgb):
    return (rgb/((2**8)-1)).astype('uint8')

########################################

def experiment_readimage(image_name):
    image_address = os.getcwd() + '/' + image_name
    image = io.imread(image_address)
    return image

########################################

def experiment_threshold(image, threshold=0):

    gray = color.rgb2gray(image)*float((2**16)-1)
    gray_filter = filters.gaussian(gray)

    edge_sobel = sobel(gray_filter).astype('int')
    min, max = edge_sobel.min(), edge_sobel.max()

    print("########### THRESHOLD ###########")
    print("Edge Sobel Image")
    print(f"[min,max] = [{min},{max}]")

    if threshold == 0:

        threshold = int(threshold_otsu(edge_sobel))
        binary = edge_sobel < threshold

        plt.close()
        f = plt.figure(1, figsize=(8,10))
        ax1 = plt.subplot(2,2,1)
        ax1.imshow(gray, cmap='gray')
        ax1.set_title("Grayscale Image")

        ax2 = plt.subplot(2,2,2)
        ax2.imshow(gray_filter, cmap='gray')
        ax2.set_title("Filtered Grayscale Image")

        ax3 = plt.subplot(2,2,3)
        ax3.imshow(edge_sobel, cmap='gray')
        ax3.set_title("Edge Sobel Image")

        ax4 = plt.subplot(2,2,4)
        ax4.imshow(binary, cmap='gray')
        ax4.set_title("Binary Image")

        plt.show(block=False)

        print(f"Threshold start = {threshold}")
        char = input("Correct (y/n)?: ")

        while char != 'y':
            threshold = input("Threshold = ")
            binary = edge_sobel < threshold

            ax4.cla()
            ax4.imshow(binary, cmap='gray')
            plt.show(block=False)

            char = input("Correct (y/n)?: ")
    else:
        binary = edge_sobel < threshold

    print("Threshold done!!")
    print("#################################\n")
    plt.close()

    return gray, binary

########################################

def experiment_crop(image, crop=[[0,0],[0,0]]):

    print("############# CROP #############")
    print("Binary Image")

    if np.array_equal( crop, np.zeros((2,2)) ):

        plt.close()
        f = plt.figure(1, figsize=(7,7))
        ax = plt.subplot(1,1,1)
        ax.imshow(image, cmap='gray')
        ax.grid()
        ax.set_title("Binary Image")
        plt.show(block=False)

        char = input("Crop image (y/n)?: ")

        if char == 'y':
            ltc = np.array(input("left-top corner = ").split(',')).astype('int')
            rbc = np.array(input("right-bottom corner = ").split(',')).astype('int')
        else:
            ltc = [0,0]
            rbc = list(np.shape(image))

        crop = [ [ltc[0],ltc[1]], [rbc[0],rbc[1]] ]

    crop = np.array(crop)
    cropped_image = image[crop[0,1]:crop[1,1],crop[0,0]:crop[1,0]]

    print("Crop done!!")
    print("#################################\n")
    plt.close()

    return cropped_image, crop

########################################

def experiment_circlefit(gray, binary, crop, diameter_extents=[[0,0],[0,0],[0,0],[0,0]]):

    print("########## CIRCLE FIT ##########")
    print("Cropped Binary Image")

    if np.array_equal( diameter_extents, np.zeros((4,2)) ):

        plt.close()
        f = plt.figure(1, figsize=(7,7))
        ax = plt.subplot(1,1,1)
        ax.imshow(binary, cmap='gray')
        ax.grid()
        ax.set_title("Cropped Binary Image")
        plt.show(block=False)

        print('Approximate diameter extents (in pixels) -')
        le = np.array(input('left extents = ').split(',')).astype('int')
        re = np.array(input('right extents = ').split(',')).astype('int')
        te = np.array(input('top extents = ').split(',')).astype('int')
        be = np.array(input('bottom extents = ').split(',')).astype('int')

    else:
        diameter_extents = np.array(diameter_extents)
        le = diameter_extents[0,:]
        re = diameter_extents[1,:]
        te = diameter_extents[2,:]
        be = diameter_extents[3,:]

    diameter_minimum_value = np.mean([min(re)-max(le), min(be)-max(te)])
    diameter_maximum_value = np.mean([max(re)-min(le), max(be)-min(te)])

    hough_radii = np.arange(int(diameter_minimum_value/2), int(diameter_maximum_value/2), 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])

    xc, yc = c + crop[0,0] , r + crop[0,1]
    cc, rr = cc + crop[0,0], rr + crop[0,1]

    x_res, y_res = list(np.shape(gray))

    center = np.array([xc, yc])
    radius = int(round(min(xc, yc, x_res-1-xc, y_res-1-yc)))

    print('Circle fit done!!')
    print('#################################\n')
    plt.close()

    f = plt.figure(1, figsize=(7,7))
    ax = plt.subplot(1,1,1)
    ax.imshow(gray, cmap='gray')
    ax.scatter(xc, yc, marker='x', color='black')
    ax.scatter(cc, rr, marker='.', color='black')
    plt.show()

    return center, radius

########################################

def experiment_analysis(mod_image_e, theta_start, theta_end, center, radius_px, px_microns):

    xc = center[0]
    yc = center[1]

    x_px = np.shape(mod_image_e)[0]
    y_px = np.shape(mod_image_e)[1]

    r_microns = np.arange(0,radius_px+1,1)*px_microns
    r_mm = r_microns/1000.0

    # delta_theta = round(math.degrees(math.atan(2.0/radius_px)),1)
    delta_theta = 0.1
    n = int((theta_end-theta_start)/delta_theta)
    s = radius_px + 1

    output = np.zeros((n,s,3), dtype = int)

    for i in range(n):
        theta = theta_start + ((i-1)*delta_theta)
        for j in range(s):
            xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
            yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
            output[i,j,0] = mod_image_e[yy,xx,0]
            output[i,j,1] = mod_image_e[yy,xx,1]
            output[i,j,2] = mod_image_e[yy,xx,2]

    R_sum = np.zeros(s)
    G_sum = np.zeros(s)
    B_sum = np.zeros(s)

    for j in range(s):
        count = 0
        for i in range(n):
            count = count + 1
            R_sum[j] = R_sum[j] + output[i,j,0]
            G_sum[j] = G_sum[j] + output[i,j,1]
            B_sum[j] = B_sum[j] + output[i,j,2]

    R_avg = R_sum/count
    G_avg = G_sum/count
    B_avg = B_sum/count

    rgb_colors = np.dstack((R_avg, G_avg, B_avg))

    ref_colors = image_sRGB_to_Lab(rgb_colors)
    image_axi = image_axisymmetric(rgb_colors)

    return r_mm, rgb_colors, ref_colors, image_axi

########################################

def experiment_dropextents(image_axi, radius_px, rgb_colors, ref_colors):

    l = int(np.mean([np.shape(image_axi)[0],np.shape(image_axi)[1]]))
    s = int((l-1)/2)

    rr, cc = circle_perimeter(0,0,radius_px)

    f = plt.figure(1, figsize=(8,6))
    ax1 = plt.subplot(2,2,1)
    ax1.imshow(color_8bit(image_axi), extent=[-s,+s,-s,+s])
    ax1.set_xlim([-s,+s])
    ax1.set_ylim([-s,+s])

    ax2 = plt.subplot(2,2,2)
    ax2.imshow(color_8bit(image_axi), extent=[-s,+s,-s,+s])
    ax2.scatter(0,0, marker='x', color='black')
    ax2.scatter(cc,rr, marker='.', color='black')
    ax2.set_xlim([-s,+s])
    ax2.set_ylim([-s,+s])

    ax3 = plt.subplot(2,1,2)
    ax3.plot(range(len(ref_colors[0,:,0])), ref_colors[0,:,0], linestyle='-', color='black')
    ax3.scatter(range(len(ref_colors[0,:,0])), ref_colors[0,:,0], marker='x', color='red')
    ax3.grid()

    plt.show(block=False)

    radius_px_mod = radius_px

    print('########### RADIUS ###########')
    print(f"Start radius (in pixels) = {radius_px_mod}")
    char = input('Correct (y/n)?: ')

    while char != 'y':

        radius_px_mod = int(input("Radius (in pixels) = "))
        rr, cc = circle_perimeter(0,0,radius_px_mod)

        ax2.cla()
        ax2.imshow(color_8bit(image_axi), extent=[-s,+s,-s,+s])
        ax2.scatter(0,0, marker='x', color='black')
        ax2.scatter(cc,rr, marker='.', color='black')
        ax2.set_xlim([-s,+s])
        ax2.set_ylim([-s,+s])

        ax3.cla()
        ax3.plot(range(len(ref_colors[0,:,0])), ref_colors[0,:,0], linestyle='-', color='black')
        ax3.scatter(range(len(ref_colors[0,:,0])), ref_colors[0,:,0], marker='x', color='red')
        ax3.axvline(x=radius_px_mod, linestyle='--', color='black')
        ax3.grid()

        plt.show(block=False)

        char = input('Correct (y/n)?: ')

    print('Radius done!!')
    print('#################################\n')
    plt.close()

    return radius_px_mod

########################################

def image_sRGB_to_Lab(rgb_colors):

    l = np.shape(rgb_colors)[1]
    ref_colors = np.zeros(np.shape(rgb_colors), dtype = 'float')

    for i in range(l):
        lab = rgb2lab(rgb_colors[0,i,:])
        ref_colors[0,i,0] = lab[0]
        ref_colors[0,i,1] = lab[1]
        ref_colors[0,i,2] = lab[2]

    return ref_colors

########################################

def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / ((2.0**16)-1.0)

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2 deg, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab

########################################

def image_axisymmetric(rgb_colors):

    l = np.shape(rgb_colors)[1]
    a = int(np.floor((l-1)/np.sqrt(2)))
    image_axi = np.zeros(((2*a)+1, (2*a)+1, 3), dtype=int)

    for i in np.arange(0,(2*a)+1,1):
        for j in np.arange(0,(2*a)+1,1):
            dist = int(round(np.sqrt(((i-a)**2)+((j-a)**2))))
            image_axi[i,j,0] = int(rgb_colors[0,dist,0])
            image_axi[i,j,1] = int(rgb_colors[0,dist,1])
            image_axi[i,j,2] = int(rgb_colors[0,dist,2])

    return image_axi

########################################

def experiment_savefile(experiment_image_file, radius_px, r_mm, ref_colors, rgb_colors, px_microns, image_axi):

    f = os.getcwd()
    info_file = f + '/info'
    if os.path.exists(info_file):
        print('Info folder already exists!')
    else:
        os.mkdir(info_file)

    os.chdir(info_file)

    head, _, _ = experiment_image_file.partition('.')
    experimental_folder = info_file + '/' + head

    if os.path.exists(experimental_folder):
        print('Experimental folder already exists!')
    else:
        os.mkdir(experimental_folder)

    os.chdir(experimental_folder)

    ref_colors1 = ref_colors[0,:,0]
    ref_colors2 = ref_colors[0,:,1]
    ref_colors3 = ref_colors[0,:,2]

    ref_R = rgb_colors[0,:,0]
    ref_G = rgb_colors[0,:,1]
    ref_B = rgb_colors[0,:,2]

    np.savetxt('radius.txt', [int(radius_px)], fmt='%d')
    np.savetxt('r_mm.txt', r_mm, fmt='%0.6f')
    np.savetxt('px_microns.txt', [px_microns], fmt='%0.6f')

    np.savetxt('ref_colors1.txt', ref_colors1, fmt='%0.6f')
    np.savetxt('ref_colors2.txt', ref_colors2, fmt='%0.6f')
    np.savetxt('ref_colors3.txt', ref_colors3, fmt='%0.6f')

    np.savetxt('ref_R.txt', ref_R, fmt='%d')
    np.savetxt('ref_G.txt', ref_G, fmt='%d')
    np.savetxt('ref_B.txt', ref_B, fmt='%d')

    np.savetxt('bigger_picture_R.txt', image_axi[:,:,0], fmt='%d')
    np.savetxt('bigger_picture_G.txt', image_axi[:,:,1], fmt='%d')
    np.savetxt('bigger_picture_B.txt', image_axi[:,:,2], fmt='%d')

    return None

########################################
