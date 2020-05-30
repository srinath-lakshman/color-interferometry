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

def circle_perimeter1(x,y,r):
    yy, xx = circle_perimeter(x,y,r)
    return xx, yy

########################################

def experiment_circlefit(image_filename='', center=[0,0], crop=0, threshold=0, radii=[0,0]):

    image = io.imread(image_filename)

    print("########## CENTERING ##########")
    print("Gray Image")

    gray = color.rgb2gray(image)*float((2**16)-1)
    x_res, y_res = list(np.shape(gray))
    xc, yc = (x_res-1)/2, (y_res-1)/2

    plt.close()
    f = plt.figure(1, figsize=(6,4))
    ax1 = plt.subplot(2,3,1)
    ax1.imshow(gray, cmap='gray')
    ax1.axvline(x = xc, linestyle='-', color='black')
    ax1.axhline(y = yc, linestyle='-', color='black')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image')

    ax2 = plt.subplot(2,3,2)
    ax2.set_title('Centered Image')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = plt.subplot(2,3,3)
    ax3.set_title('Cropped Image')
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = plt.subplot(2,3,4)
    ax4.set_title('Edge Image')
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = plt.subplot(2,3,5)
    ax5.set_title('Binary Image')
    ax5.set_aspect('equal')
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6 = plt.subplot(2,3,6)
    ax6.set_title('Circlefit Image')
    ax6.set_aspect('equal')
    ax6.set_xticks([])
    ax6.set_yticks([])

    plt.show(block=False)

    if np.array_equal( center, np.zeros((1,2)) ):
        center = np.array(input('Approximate center = ').split(',')).astype('int')

    radius = int(min(center[0], center[1], x_res-1-center[0], y_res-1-center[1]))
    gray_centered = gray[center[1]-radius:center[1]+radius+1, center[0]-radius:center[0]+radius+1]

    ax2.imshow(gray_centered, cmap='gray', extent=[-radius,+radius,-radius,+radius])
    ax2.set_xlim(-radius,+radius)
    ax2.set_ylim(-radius,+radius)
    ax2.axvline(x = 0, linestyle='-', color='black')
    ax2.axhline(y = 0, linestyle='-', color='black')
    plt.show(block=False)

    print('Centering done!!')
    print('#################################\n')

    print("############# CROP #############")
    print("Centered Image")

    if crop == 0:
        crop = int(input("Crop distance from center = "))

    gray_crop = gray_centered[radius-crop:radius+crop+1,radius-crop:radius+crop+1]

    ax3.imshow(gray_crop, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    plt.show(block=False)

    print('Crop done!!')
    print('#################################\n')

    gray_filter = filters.gaussian(gray_crop)
    edge_sobel = sobel(gray_filter).astype('int')
    min_val, max_val = edge_sobel.min(), edge_sobel.max()

    print("########### THRESHOLD ###########")
    print("Edge Sobel Image")
    print(f"[min,max] = [{min_val},{max_val}]")

    ax4.imshow(edge_sobel, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])

    binary = edge_sobel < int(threshold_otsu(edge_sobel))

    ax5.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    plt.show(block=False)

    if threshold == 0:
        threshold = int(input("Threshold = "))

    if threshold > 0:
        binary = edge_sobel > abs(threshold)
    else:
        binary = edge_sobel < abs(threshold)

    ax5.cla()
    ax5.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    ax5.set_title('Binary Image')
    ax5.set_aspect('equal')
    ax5.set_xlim(-crop, +crop)
    ax5.set_ylim(-crop, +crop)
    ax5.set_xticks([])
    ax5.set_yticks([])
    plt.show(block=False)

    print('Threshold done!!')
    print('#################################\n')

    print("############ RADIUS #############")
    print("Binary Image")

    if np.array_equal( radius, np.zeros((1,2)) ):
        radii = np.array(input('Radius extents = ').split(',')).astype('int')

    hough_radii = np.arange(radii[0], radii[1], 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    x_circle_center = c
    y_circle_center = r
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])
    x_circle_perimeter = cc
    y_circle_perimeter = rr

    print('Circle fit done!!')
    print('#################################\n')

    ax5.scatter(x_circle_center-crop, -(y_circle_center-crop), marker='x', color='red')
    ax5.scatter(x_circle_perimeter-crop, -(y_circle_perimeter-crop), marker='.', color='red')

    # plt.figure(2)
    # plt.imshow(binary, cmap='gray')
    # plt.scatter(x_circle_center, y_circle_center, marker='x', color='red')
    # plt.scatter(x_circle_perimeter, y_circle_perimeter, marker='.', color='red')
    # plt.show()

    delta_x = center[0] - crop
    delta_y = center[1] - crop

    ax6.imshow(gray, cmap='gray')
    ax6.scatter(x_circle_center+delta_x, y_circle_center+delta_y, marker='x', color='black')
    ax6.scatter(x_circle_perimeter+delta_x, y_circle_perimeter+delta_y, marker='.', color='black')

    # plt.figure(2)
    # plt.subplot(1,2,1)
    # plt.imshow(binary, cmap='gray')
    # plt.scatter(x_circle_center, y_circle_center, marker='x', color='red')
    # plt.scatter(y_circle_perimeter, x_circle_perimeter, marker='.', color='red')
    #
    # plt.subplot(1,2,2)
    # plt.imshow(binary, cmap='gray', extent=[-(crop+0.5),+(crop+0.5),-(crop+0.5),+(crop+0.5)])
    # plt.scatter(x_circle_center-crop, -(y_circle_center-crop), marker='x', color='red')
    # plt.scatter(y_circle_perimeter-crop, -(x_circle_perimeter-crop), marker='.', color='red')

    center_px = [x_circle_center+delta_x, y_circle_center+delta_y]
    radius_px = hough_radii[ridx]
    radius_max_px = int(round(min(center_px[0], center_px[1], x_res-1-center_px[0], y_res-1-center_px[1])))

    print(f"center = {center_px}")
    print(f"radius = {radius_px}")
    print(f"radius max = {radius_max_px}")

    plt.show()

    return center_px, radius_px, radius_max_px

########################################

def experiment_analysis(theta_start, theta_end, center, px_microns):

    xc = center[0]
    yc = center[1]

    channel_R = np.loadtxt("channel_R.txt")
    channel_G = np.loadtxt("channel_G.txt")
    channel_B = np.loadtxt("channel_B.txt")

    mod_image_e = np.dstack((channel_R, channel_G, channel_B))

    x_px = np.shape(mod_image_e)[0]
    y_px = np.shape(mod_image_e)[1]

    # delta_theta = round(math.degrees(math.atan(2.0/radius_px)),1)
    delta_theta = 0.1
    n = int((theta_end-theta_start)/delta_theta)
    s = int(round(min(center[0], center[1], x_px-1-center[0], y_px-1-center[1])))

    r_microns = np.arange(0,s,1)*px_microns
    r_mm = r_microns/1000.0

    output = np.zeros((n,s,3), dtype = int)

    for i in range(n):
        theta = theta_start + ((i-1)*delta_theta)
        for j in range(s):
            xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
            yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
            # if xx == x_px:
            #     xx = x_px - 1
            # if yy == y_px:
            #     yy = y_px - 1
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

def experiment_savefile(image_filename, r_mm, ref_colors, rgb_colors, px_microns, image_axi):

    os.chdir('..')
    os.chdir('..')

    for f_subfolder in ['colors', image_filename.split('.')[0]]:
        if os.path.exists(f_subfolder):
            print(f"{f_subfolder} folder already exist!")
        else:
            print(f"{f_subfolder} folder does not exist!")
            os.mkdir(f_subfolder)
            print(f"{f_subfolder} folder created!")
        os.chdir(f_subfolder)

    ref_colors1 = ref_colors[0,:,0]
    ref_colors2 = ref_colors[0,:,1]
    ref_colors3 = ref_colors[0,:,2]

    ref_R = rgb_colors[0,:,0]
    ref_G = rgb_colors[0,:,1]
    ref_B = rgb_colors[0,:,2]

    radius_px = len(r_mm)

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

    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    return None

########################################
