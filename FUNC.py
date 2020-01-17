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

########################################

def color_8bit(rgb):
    return (rgb/((2**8)-1)).astype('uint8')

########################################

def image_readfile(image_name):
    image_address = os.getcwd() + '/' + image_name
    image = io.imread(image_address)
    return image

########################################

def length_analysis(lengthscale_file):

    f = os.getcwd()
    main_lengthscale_file = f + '/' + lengthscale_file

    rgb = io.imread(main_lengthscale_file)
    gray = color.rgb2gray(rgb)*float((2**16)-1)

    gray_filter = filters.gaussian(gray)
    edge_sobel = sobel(gray_filter)
    threshold = threshold_otsu(edge_sobel)
    binary = edge_sobel > threshold

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(gray, cmap='gray')
    plt.grid()
    plt.title('Grayscale Image')

    plt.subplot(2,2,2)
    plt.imshow(gray_filter, cmap='gray')
    plt.grid()
    plt.title('Filtered Grayscale Image')

    plt.subplot(2,2,3)
    plt.imshow(edge_sobel, cmap='gray')
    plt.grid()
    plt.title('Edge Sobel Image')

    plt.subplot(2,2,4)
    plt.imshow(binary, cmap='gray')
    plt.grid()
    plt.title('Binary Image')
    plt.show(block=False)

    print('Approximate extents of diameter (in pixels) -')
    left_extents = np.array(input('left extents = ').split(',')).astype('int')
    right_extents = np.array(input('right extents = ').split(',')).astype('int')
    top_extents = np.array(input('top extents = ').split(',')).astype('int')
    bottom_extents = np.array(input('bottom extents = ').split(',')).astype('int')

    diameter_minimum_value = np.mean([min(right_extents)-max(left_extents), min(bottom_extents)-max(top_extents)])
    diameter_maximum_value = np.mean([max(right_extents)-min(left_extents), max(bottom_extents)-min(top_extents)])

    hough_radii = np.arange(int(diameter_minimum_value/2), int(diameter_maximum_value/2), 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])

    diameter_pixels = int(2*hough_radii[ridx])
    diameter_mm = input('Enter diameter value (in mm) = ')

    length_scale_mm = int(diameter_mm)
    length_pixels = int(diameter_pixels)
    px_microns = round((1000.0*length_scale_mm)/length_pixels,3)

    plt.close()
    plt.figure(1)
    plt.imshow(gray, cmap='gray')
    plt.scatter(cc,rr)
    plt.scatter(c,r)
    plt.title('Grayscale Image')
    plt.show(block=False)

    print('Calculated diameter value (in pixels) =', length_pixels)
    print('1 pixel =', px_microns, 'microns')

    input('\nPress [ENTER] to continue...')
    plt.close()

    return px_microns

########################################

def find_center(image_v):

    gray = color.rgb2gray(image_v)*float((2**16)-1)
    gray_filter = filters.gaussian(gray)
    edge_sobel = sobel(gray_filter)

    x_res = np.shape(image_v)[0]
    y_res = np.shape(image_v)[1]

    min_intensity = gray.min().astype(int)
    max_intensity = gray.max().astype(int)

    print('[min, max] = [' + str(min_intensity) +', ' + str(max_intensity) + ']')

    avg_intensity = np.mean([min_intensity, max_intensity]).astype(int)
    print('Threshold = ' + str(avg_intensity))

    char = 'n'

    i = 0

    while char != 'y':
        if i==0:
            threshold = avg_intensity
        else:
            threshold = int(input('Threshold = '))

        binary = edge_sobel < threshold

        plt.close()
        plt.figure(1)
        plt.subplot(2,2,1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale Image')

        plt.subplot(2,2,2)
        plt.imshow(gray_filter, cmap='gray')
        plt.title('Filtered Grayscale Image')

        plt.subplot(2,2,3)
        plt.imshow(edge_sobel, cmap='gray')
        plt.title('Edge Sobel Image')

        plt.subplot(2,2,4)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Image')

        plt.show(block=False)

        char = input('Exit (y/n): ')
        i = i + 1

    plt.close()

    binary = binary[100:350,250:550]

    plt.figure(2)
    plt.imshow(binary, cmap='gray')
    plt.grid()
    plt.title('Binary Image')
    plt.show(block=False)

    print('Approximate extents of diameter (in pixels) -')
    left_extents = np.array(input('left extents = ').split(',')).astype('int')
    right_extents = np.array(input('right extents = ').split(',')).astype('int')
    top_extents = np.array(input('top extents = ').split(',')).astype('int')
    bottom_extents = np.array(input('bottom extents = ').split(',')).astype('int')

    diameter_minimum_value = np.mean([min(right_extents)-max(left_extents), min(bottom_extents)-max(top_extents)])
    diameter_maximum_value = np.mean([max(right_extents)-min(left_extents), max(bottom_extents)-min(top_extents)])

    hough_radii = np.arange(int(diameter_minimum_value/2), int(diameter_maximum_value/2), 1)
    hough_res = hough_circle(binary, hough_radii)
    ridx, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    rr, cc = circle_perimeter(r,c,hough_radii[ridx])

    xc, yc = c, r

    center = [xc, yc]
    radius = int(round(min(xc, yc, x_res-1-xc, y_res-1-yc)))

    plt.close()
    plt.figure(2)
    plt.imshow(binary, cmap='gray')
    plt.scatter(cc,rr)
    plt.scatter(c,r)
    plt.title('Binary Image')
    plt.show(block=False)

    print('Calculated center of newtons rings (in pixels) =', center)
    print('Calculated radius of newtons rings (in pixels) =', radius)

    input('\nPress [ENTER] to continue...')
    plt.close()

    return center, radius

########################################

def analysis_reference(image_filename, mod_image_v, center, radius_px, px_microns):

    xc = center[0]
    yc = center[1]

    x_px = np.shape(mod_image_v)[0]
    y_px = np.shape(mod_image_v)[1]

    if '400' in image_filename or 'la1172' in image_filename:
        print('\nf400 lens used (LA1172)')
        f_lens = 400.0  #LA1172
        R_lens = 206.0
        lens_config = 'f0400'
    elif '300' in image_filename or 'la1484' in image_filename:
        print('\nf300 lens used (LA1484)')
        f_lens = 300.0  #LA1484
        R_lens = 154.5
        lens_config = 'f0300'
    elif '1000' in image_filename or 'la1464' in image_filename:
        print('\nf1000 lens used (LA1464)')
        f_lens = 1000.0  #LA1464
        R_lens = 515.1
        lens_config = 'f1000'
    else:
        f_lens = 0
        R_lens = 0
        lens_config = 'none'
        print('No lens configuration detected. Exit the code.')

    option = input('Correct (y/n)?: ')

    if option == 'y' or 'Y':
        r_microns = np.arange(0,radius_px,1)*px_microns
        r_mm = r_microns/1000.0
        h_microns = (R_lens - np.sqrt((R_lens*R_lens)-(r_mm*r_mm)))*1000.0

        theta_start = 0
        theta_end = 360
        delta_theta = round(math.degrees(math.atan(2.0/radius_px)),1)
        n = int((theta_end-theta_start)/delta_theta)
        s = radius_px

        output = np.zeros((n,s,3), dtype = int)

        for i in range(n):
            theta = theta_start + ((i-1)*delta_theta)
            for j in range(s):
                xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
                yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
                output[i,j,0] = mod_image_v[yy,xx,0]
                output[i,j,1] = mod_image_v[yy,xx,1]
                output[i,j,2] = mod_image_v[yy,xx,2]

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

    else:

        print('Please re-run the code using proper lens configuration')
        r_mm = 0
        h_microns = 0
        rgb_colors = 0
        ref_colors = 0
        image_axi = 0

    print('Analysis of reference image completed!!')

    return r_mm, h_microns, rgb_colors, ref_colors, image_axi, lens_config

########################################

def savefile_reference(lens_config, radius_px, r_mm, h_microns, ref_colors, rgb_colors, px_microns):

    f = os.getcwd()
    info_file = f + '/info_' + lens_config
    if os.path.exists(info_file):
        print('\n**Info folder already exists**')
    else:
        os.mkdir(info_file)

    os.chdir(info_file)

    ref_colors1 = ref_colors[0,:,0]
    ref_colors2 = ref_colors[0,:,1]
    ref_colors3 = ref_colors[0,:,2]

    ref_R = rgb_colors[0,:,0]
    ref_G = rgb_colors[0,:,1]
    ref_B = rgb_colors[0,:,2]

    np.savetxt('radius.txt', [int(radius_px)], fmt='%d')
    np.savetxt('r_mm.txt', r_mm, fmt='%0.6f')
    np.savetxt('h_microns.txt', h_microns, fmt='%0.6f')
    np.savetxt('px_microns.txt', [px_microns], fmt='%0.6f')

    np.savetxt('ref_colors1.txt', ref_colors1, fmt='%0.6f')
    np.savetxt('ref_colors2.txt', ref_colors2, fmt='%0.6f')
    np.savetxt('ref_colors3.txt', ref_colors3, fmt='%0.6f')

    np.savetxt('ref_R.txt', ref_R, fmt='%d')
    np.savetxt('ref_G.txt', ref_G, fmt='%d')
    np.savetxt('ref_B.txt', ref_B, fmt='%d')

    print('Saving of reference image completed!!')

    return None

########################################

def image_color_normalization(image, background):

    R_avg = np.mean(background[:,:,0])
    G_avg = np.mean(background[:,:,1])
    B_avg = np.mean(background[:,:,2])

    R_ratio = image[:,:,0]/R_avg
    G_ratio = image[:,:,1]/G_avg
    B_ratio = image[:,:,2]/B_avg
    image_ratio = np.dstack((R_ratio, G_ratio, B_ratio))

    max_val = max(R_ratio.max(), G_ratio.max(), B_ratio.max())
    factor = 2.0

    image_color_corrected = (image_ratio*((np.power(2.0,16.0)-1.0)/(factor*max_val))).astype(int)

    return image_color_corrected

########################################

def analysis_experiment(mod_image_e, theta_start, theta_end, center, radius_px, px_microns):

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

def savefile_experimental(experiment_image_file, radius_px, r_mm, ref_colors, rgb_colors, px_microns):

    f = os.getcwd()
    info_file = f + '/info'
    if os.path.exists(info_file):
        print('Info folder already exists!')
    else:
        os.mkdir(info_file)

    os.chdir(info_file)

    experimental_folder = info_file + '/' + experiment_image_file[0:22]
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

    return None

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

def analysis_readme():

    n = np.loadtxt('radius.txt', dtype='int')
    px_microns = np.loadtxt('px_microns.txt')

    R_ch = np.loadtxt('ref_R.txt')
    G_ch = np.loadtxt('ref_G.txt')
    B_ch = np.loadtxt('ref_B.txt')

    L_ch = np.loadtxt('ref_colors1.txt')
    a_ch = np.loadtxt('ref_colors2.txt')
    b_ch = np.loadtxt('ref_colors3.txt')

    sRGB = np.dstack((R_ch, G_ch, B_ch))
    Lab = np.dstack((L_ch, a_ch, b_ch))

    return n, sRGB, Lab, px_microns

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

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    # #perform some checks
    # if lookahead < 1:
    #     raise ValueError, "Lookahead must be '1' or above in value"
    # if not (np.isscalar(delta) and delta >= 0):
    #     raise ValueError, "delta must be a positive number"

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]

########################################

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise (ValueError,
                'Input vectors y_axis and x_axis must have same length')

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

########################################
