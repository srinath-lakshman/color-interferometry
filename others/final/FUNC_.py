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

################################################################################
############################## general_functions ###############################
################################################################################

def image_readfile():
    return glob.glob('*.tif')[0]

########################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

########################################

def color_8bit(rgb):
    return (rgb/((2**8)-1)).astype('uint8')

########################################

def getNumbers(str):
    return re.findall(r'[0-9]+', str)

########################################

def sort(nums, var):
    n = len(nums)
    for i in range(n):
        for j in range(0,n-i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                var[j], var[j+1] = var[j+1], var[j]
    return nums, var

################################################################################
############################# background_reference #############################
################################################################################

def background_reference_analysis(image_file):

    rgb = io.imread(image_file)
    gray = rgb2gray(rgb)

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
            threshold = input('Threshold = ')

        mask = gray > threshold

        plt.close()

        plt.subplot(1,2,1)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))

        plt.subplot(1,2,2)
        plt.imshow(mask, cmap=plt.get_cmap('gray'))

        plt.show(block = False)

        char = raw_input('Exit (y/n): ')
        i = i + 1

    plt.close()

    R_channel = rgb[:,:,0]
    G_channel = rgb[:,:,1]
    B_channel = rgb[:,:,2]

    R_channel_mean = (np.mean(R_channel)).astype('int')
    G_channel_mean = (np.mean(G_channel)).astype('int')
    B_channel_mean = (np.mean(B_channel)).astype('int')

    mean_intensities = [R_channel_mean, G_channel_mean, B_channel_mean]
    threshold = [threshold]

    return threshold, mask, mean_intensities, R_channel, G_channel, B_channel

########################################

def background_reference_savefile(threshold, mask, mean_intensities, R_ch, G_ch, B_ch, info_file):

    os.makedirs(info_file)
    os.chdir(info_file)

    np.savetxt('threshold.txt', threshold, fmt='%d')
    np.savetxt('mask.txt', mask, fmt='%d')
    np.savetxt('mean_intensities.txt', mean_intensities, fmt='%d')
    np.savetxt('ch1_R.txt', R_ch, fmt='%d')
    np.savetxt('ch2_G.txt', G_ch, fmt='%d')
    np.savetxt('ch3_B.txt', B_ch, fmt='%d')

    return None

################################################################################
############################## lateral_reference ###############################
################################################################################

def lateral_reference_readfile():

    txtfile = glob.glob('*.txt')[0]

    file_contents = open(txtfile, 'r').read()
    first_line = file_contents[0:19]

    string1 = first_line.replace(' ', '')
    string2 = string1[12:]
    string3 = string2[::-1]

    units = string3[0:2]

    if units == 'mm':
        len_meters = float(string3[2:])/1000
    else:
        len_meters = 0

    return len_meters

########################################

def lateral_reference_analysis(image_file):

    rgb = io.imread(image_file)
    gray = rgb2gray(rgb)

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
            threshold = input('Threshold = ')

        binary = gray > threshold

        plt.close()

        plt.subplot(1,2,1)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))

        plt.subplot(1,2,2)
        plt.imshow(binary, cmap=plt.get_cmap('gray'))

        plt.show(block = False)

        char = raw_input('Exit (y/n): ')
        i = i + 1

    plt.close()

    plt.imshow(binary, cmap=plt.get_cmap('gray'))

    plt.show(block = False)

    x_limits = input('X-limits = ')
    y_limits = input('Y-limits = ')

    delta_x = abs(x_limits[0]-x_limits[1])
    delta_y = abs(y_limits[0]-y_limits[1])

    extents = np.zeros((2,2))
    extents[0,0] = x_limits[0]
    extents[1,0] = x_limits[1]
    extents[0,1] = y_limits[0]
    extents[1,1] = y_limits[1]

    avg_px = int(round((delta_x + delta_y)/2))

    return avg_px, threshold, extents

########################################

def lateral_reference_savefile(len, avg_px, threshold, extents, info_file):

    px_meters = len/avg_px

    os.makedirs(info_file)
    os.chdir(info_file)

    with open(r'readme.txt', 'w') as f:
        f.write('dot diameter = ' + str.format('{0:.0f}', len*(10**3)) + ' mm')
        f.write('\n\n')
        f.write('threshold = ' + str.format('{0:.0f}', threshold))
        f.write('\n')
        f.write('x-extents = [' + str.format('{0:.1f}', extents[0][0]) + ', ' + str.format('{0:.1f}', extents[1][0]) + ']')
        f.write('\n')
        f.write('y-extents = [' + str.format('{0:.1f}', extents[0][1]) + ', ' + str.format('{0:.1f}', extents[1][1]) + ']')
        f.write('\n')
        f.write('dot diameter = ' + str(avg_px) + ' pixels')
        f.write('\n')
        f.write('1 pixel = ' + str.format('{0:.3f}', px_meters*(10**6)) + ' microns')

    f.close()

    return None

################################################################################
############################## vertical_reference ##############################
################################################################################

def vertical_reference_lateral():

    with open(r'readme.txt') as file:
        lines = file.readlines()

    array = getNumbers(lines[6])
    px = float(array[1] + '.' + array[2])

    return px

########################################

def vertical_reference_background():

    mask = np.loadtxt('mask.txt')
    mean_intensities = np.loadtxt('mean_intensities.txt')
    R_ch = np.loadtxt('ch1_R.txt')
    G_ch = np.loadtxt('ch2_G.txt')
    B_ch = np.loadtxt('ch3_B.txt')

    return mask, mean_intensities, R_ch, G_ch, B_ch

########################################

def image_info_readfile(info_file):

    os.chdir(info_file)

    threshold = np.loadtxt('threshold.txt')
    xc = int(np.loadtxt('center.txt')[0])
    yc = int(np.loadtxt('center.txt')[1])
    radius_px = int(np.loadtxt('radius.txt'))

    center = [xc, yc]

    Rv_ch = np.loadtxt('ch1_R.txt')
    Gv_ch = np.loadtxt('ch2_G.txt')
    Bv_ch = np.loadtxt('ch3_B.txt')

    os.chdir('..')

    return threshold, center, radius_px, Rv_ch, Gv_ch, Bv_ch

########################################

def image_analysis1(image_file):

    rgb = io.imread(image_file)
    gray = rgb2gray(rgb)

    x_centered_image  = np.shape(rgb)[0]
    y_centered_image  = np.shape(rgb)[1]

    ########################### uncomment this #########################
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
            threshold = input('Threshold = ')

        binary = gray > threshold

        plt.close()

        plt.subplot(1,2,1)
        plt.imshow(gray, cmap=plt.get_cmap('gray'))

        plt.subplot(1,2,2)
        plt.imshow(binary, cmap=plt.get_cmap('gray'))

        plt.show(block = False)

        char = raw_input('Exit (y/n): ')
        i = i + 1

    plt.close()

    plt.imshow(binary, cmap=plt.get_cmap('gray'))

    plt.show(block = False)

    x_limits = input('X-limits = ')
    y_limits = input('Y-limits = ')
    ########################### uncomment this #########################

    ########################### comment this ###########################
    # threshold = 25000
    #
    # x_limits = np.zeros(2)
    # y_limits = np.zeros(2)
    #
    # x_limits[0] = 198.5
    # x_limits[1] = 309.5
    #
    # y_limits[0] = 202.5
    # y_limits[1] = 313.5
    ########################### comment this ###########################

    xc = int((x_limits[0] + x_limits[1])/2)
    yc = int((y_limits[0] + y_limits[1])/2)

    center = [xc, yc]
    radius = int(round(min(xc, yc, x_centered_image-1-xc, y_centered_image-1-yc)))

    R_channel = rgb[:,:,0]
    G_channel = rgb[:,:,1]
    B_channel = rgb[:,:,2]

    plt.close()

    return threshold, center, radius, R_channel, G_channel, B_channel

########################################

def planoconvex_readfile(radius_px, px_microns):

    # txtfile = glob.glob('*.txt')[0]
    txtfile = r'readme.txt'

    with open(txtfile) as f:
        lines = f.readlines()

    if getNumbers(lines[0])[0] == '1484':
        ff = 300.0
        RR = 154.5
    elif getNumbers(lines[0])[0] == '1509':
        ff = 100.0
        RR = 51.5
    elif getNumbers(lines[0])[0] == '1172':
        ff = 400.0
        RR = 206.0
    else:
        ff = 0.0
        RR = 0.0

    R_lens = RR

    r_microns = np.arange(0,radius_px+1,1)*px_microns
    r_mm = r_microns/1000.0
    h_microns = (R_lens - np.sqrt((R_lens*R_lens)-(r_mm*r_mm)))*1000.0

    return r_microns, r_mm, h_microns, R_lens

########################################

def image_color_corrected1(image_b, mask, mean_val, image_v):

    mod_image_norm = np.zeros(np.shape(image_b))

    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i,j] == 0:
                mod_image_norm[i,j,0] = 0
                mod_image_norm[i,j,1] = 0
                mod_image_norm[i,j,2] = 0
            else:
                mod_image_norm[i,j,0] = image_v[i,j,0]/image_b[i,j,0]
                mod_image_norm[i,j,1] = image_v[i,j,1]/image_b[i,j,1]
                mod_image_norm[i,j,2] = image_v[i,j,2]/image_b[i,j,2]

    max_val = max(mod_image_norm[:,:,0].max(), mod_image_norm[:,:,1].max(), mod_image_norm[:,:,2].max())

    factor = 2.0
    mod_image = (mod_image_norm*((np.power(2.0,16.0)-1.0)/(factor*max_val))).astype(int)

    return mod_image

########################################

def image_centered_and_cropped(image, center, radius_px):

    mod_image = image[center[1]-radius_px:center[1]+radius_px+1, center[0]-radius_px:center[0]+radius_px+1,:]

    return mod_image

########################################

def extract_color_channels1(mod_image_cropped, theta_start, theta_end):

    x_px = np.shape(mod_image_cropped)[0]
    y_px = np.shape(mod_image_cropped)[1]

    avg = int((x_px + y_px)/2)
    delta_theta = round(math.degrees(math.atan(2.0/avg)),1)
    radius_px = int((avg-1)/2)

    n = int((theta_end-theta_start)/delta_theta)

    xc = radius_px
    yc = radius_px
    s = radius_px+1

    output = np.zeros((n,s,3), dtype = int)

    for i in range(n):
        theta = theta_start + ((i-1)*delta_theta)
        for j in range(s):
            xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
            yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
            output[i,j,0] = mod_image_cropped[yy,xx,0]
            output[i,j,1] = mod_image_cropped[yy,xx,1]
            output[i,j,2] = mod_image_cropped[yy,xx,2]

    R_sum = np.zeros(s)
    G_sum = np.zeros(s)
    B_sum = np.zeros(s)

    for j in range(s):
        count = 0
        for i in range(n):
            if np.array_equal(output[i,j], [0, 0, 0]):
                count = count - 1
            else:
                count = count + 1
            R_sum[j] = R_sum[j] + output[i,j,0]
            G_sum[j] = G_sum[j] + output[i,j,1]
            B_sum[j] = B_sum[j] + output[i,j,2]

    R_avg = R_sum/count
    G_avg = G_sum/count
    B_avg = B_sum/count

    l = np.zeros(np.shape(R_avg), dtype=float)
    a = np.zeros(np.shape(G_avg), dtype=float)
    b = np.zeros(np.shape(B_avg), dtype=float)

    for j in range(s):
        lab = rgb2lab([R_avg[j],G_avg[j],B_avg[j]])
        l[j] = lab[0]
        a[j] = lab[1]
        b[j] = lab[2]

    side_len = int(np.floor(radius_px/np.sqrt(2)))

    image_axisymmetric = np.zeros(((2*side_len)+1, (2*side_len)+1, 3), dtype=int)

    for i in np.arange(0,(2*side_len)+1,1):
        for j in np.arange(0,(2*side_len)+1,1):
            dist = int(round(np.sqrt(((i-side_len)**2)+((j-side_len)**2))))
            image_axisymmetric[i,j,0] = int(R_avg[dist])
            image_axisymmetric[i,j,1] = int(G_avg[dist])
            image_axisymmetric[i,j,2] = int(B_avg[dist])

    return R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric

########################################

def extract_color_channels2(mod_image_cropped, theta):

    x_px = np.shape(mod_image_cropped)[0]
    y_px = np.shape(mod_image_cropped)[1]

    avg = int((x_px + y_px)/2)
    delta_theta = round(math.degrees(math.atan(2.0/avg)),1)
    radius_px = int((avg-1)/2)

    xc = radius_px
    yc = radius_px
    s = radius_px+1

    output = np.zeros((s,3), dtype = int)

    for j in range(s):
        xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
        yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
        output[j,0] = mod_image_cropped[yy,xx,0]
        output[j,1] = mod_image_cropped[yy,xx,1]
        output[j,2] = mod_image_cropped[yy,xx,2]

    R_avg = output[:,0]
    G_avg = output[:,1]
    B_avg = output[:,2]

    l = np.zeros(np.shape(R_avg), dtype=float)
    a = np.zeros(np.shape(G_avg), dtype=float)
    b = np.zeros(np.shape(B_avg), dtype=float)

    for j in range(s):
        lab = rgb2lab ([R_avg[j],G_avg[j],B_avg[j]])
        l[j] = lab[0]
        a[j] = lab[1]
        b[j] = lab[2]

    side_len = int(np.floor(radius_px/np.sqrt(2)))

    image_axisymmetric = np.zeros(((2*side_len)+1, (2*side_len)+1, 3), dtype=int)

    for i in np.arange(0,(2*side_len)+1,1):
        for j in np.arange(0,(2*side_len)+1,1):
            dist = int(round(np.sqrt(((i-side_len)**2)+((j-side_len)**2))))
            image_axisymmetric[i,j,0] = int(R_avg[dist])
            image_axisymmetric[i,j,1] = int(G_avg[dist])
            image_axisymmetric[i,j,2] = int(B_avg[dist])

    return R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric

########################################

def extract_color_channels3(mod_image_cropped, px_microns, R_lens):

    x_px = np.shape(mod_image_cropped)[0]
    y_px = np.shape(mod_image_cropped)[1]

    size = x_px*y_px

    avg = int((x_px + y_px)/2)
    radius_px = int((avg-1)/2)
    s = radius_px+1

    rr_mm = np.zeros(size)
    hh_microns = np.zeros(size)
    R_var = np.zeros(size)
    G_var = np.zeros(size)
    B_var = np.zeros(size)

    count = 0

    for i in range(x_px):
        for j in range(y_px):
            dist = (np.sqrt(((i-radius_px)**2) + ((j-radius_px)**2)))
            rr_mm[count] = dist*px_microns*(1/1000.0)
            hh_microns[count] = (R_lens - np.sqrt((R_lens*R_lens)-(rr_mm[count]*rr_mm[count])))*1000.0
            R_var[count] = mod_image_cropped[i,j,0]
            G_var[count] = mod_image_cropped[i,j,1]
            B_var[count] = mod_image_cropped[i,j,2]
            count = count + 1

    hh_microns = np.round(hh_microns, 3)
    sample = []
    count = 0

    for item in hh_microns:
        if item not in sample:
            sample.append(item)
            count = count + 1

    mod_rr_mm = np.zeros(count)
    mod_hh_microns = np.zeros(count)
    mod_R_var = np.zeros(count)
    mod_G_var = np.zeros(count)
    mod_B_var = np.zeros(count)

    count = 0
    count1 = -1
    for item in hh_microns:
        count1 = count1 + 1
        if item not in mod_hh_microns:
            mod_hh_microns[count] = hh_microns[count1]
            mod_rr_mm[count] = rr_mm[count1]
            mod_R_var[count] = R_var[count1]
            mod_G_var[count] = G_var[count1]
            mod_B_var[count] = B_var[count1]
            count = count + 1

    [mod_rr_mm, mod_hh_microns] = sort(mod_rr_mm, mod_hh_microns)
    [mod_rr_mm, mod_R_var] = sort(mod_rr_mm, mod_R_var)
    [mod_rr_mm, mod_G_var] = sort(mod_rr_mm, mod_G_var)
    [mod_rr_mm, mod_B_var] = sort(mod_rr_mm, mod_B_var)

    radius_px = len(mod_rr_mm)

    l = np.zeros(radius_px, dtype=float)
    a = np.zeros(radius_px, dtype=float)
    b = np.zeros(radius_px, dtype=float)

    for j in range(radius_px):
        lab = rgb2lab ([mod_R_var[j],mod_G_var[j],mod_B_var[j]])
        l[j] = lab[0]
        a[j] = lab[1]
        b[j] = lab[2]

    return mod_R_var, mod_G_var, mod_B_var, l, a, b, radius_px, mod_rr_mm, mod_hh_microns

########################################

def vertical_reference_savefile(threshold, center, radius, R_ch, G_ch, B_ch, r_mm, h_microns, l, a, b):

    np.savetxt('threshold.txt', [threshold], fmt='%d')
    np.savetxt('center.txt', center, fmt='%d')

    np.savetxt('ch1_R.txt', R_ch, fmt='%d')
    np.savetxt('ch2_G.txt', G_ch, fmt='%d')
    np.savetxt('ch3_B.txt', B_ch, fmt='%d')

    np.savetxt('r_mm.txt', r_mm, fmt='%0.6f')
    np.savetxt('h_microns.txt', h_microns, fmt='%0.6f')
    np.savetxt('var1.txt', l, fmt='%0.6f')
    np.savetxt('var2.txt', a, fmt='%0.6f')
    np.savetxt('var3.txt', b, fmt='%0.6f')
    np.savetxt('radius.txt', [radius], fmt='%d')

    return None

################################################################################
################################# experimental #################################
################################################################################

def experimental_savefile(threshold, center, radius_px, Re_ch, Ge_ch, Be_ch, R_avg, G_avg, B_avg):

    np.savetxt('threshold.txt', [threshold], fmt='%d')
    np.savetxt('center.txt', center, fmt='%d')
    np.savetxt('radius.txt', [radius_px], fmt='%d')

    np.savetxt('ch1_R.txt', Re_ch, fmt='%d')
    np.savetxt('ch2_G.txt', Ge_ch, fmt='%d')
    np.savetxt('ch3_B.txt', Be_ch, fmt='%d')

    np.savetxt('var1.txt',R_avg, fmt='%0.6f')
    np.savetxt('var2.txt',G_avg, fmt='%0.6f')
    np.savetxt('var3.txt',B_avg, fmt='%0.6f')

    return None

################################################################################
################################ final_analysis ################################
################################################################################

def analysis_vertical_readfile():

    r_ref = np.loadtxt('r_mm.txt')
    h_ref = np.loadtxt('h_microns.txt')

    px_ref = int(np.loadtxt('radius.txt'))

    R_ref = np.loadtxt('var1.txt')
    G_ref = np.loadtxt('var2.txt')
    B_ref = np.loadtxt('var3.txt')

    return r_ref, h_ref, px_ref, R_ref, G_ref, B_ref

########################################

def analysis_experimental_readfile():

    px_exp = int(np.loadtxt('radius.txt'))

    R_exp = np.loadtxt('var1.txt')
    G_exp = np.loadtxt('var2.txt')
    B_exp = np.loadtxt('var3.txt')

    return px_exp, R_exp, G_exp, B_exp

################################################################################

def readfile_vertical(file):

    with open(r'info.txt') as file:
        lines = file.readlines()

    xc = int(getNumbers(lines[2])[0])
    yc = int(getNumbers(lines[3])[0])
    radius = int(getNumbers(lines[4])[0])

    R_ch = np.loadtxt('ch1_R.txt')
    G_ch = np.loadtxt('ch2_G.txt')
    B_ch = np.loadtxt('ch3_B.txt')

    return xc, yc, radius, R_ch, G_ch, B_ch

################################################################################

################################################################################

def color_corrected_image2(image_b, mask, mean_val, image_v):

    mod_image_norm = np.zeros(np.shape(image_b))

    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i,j] == 0:
                mod_image_norm[i,j,0] = 0
                mod_image_norm[i,j,1] = 0
                mod_image_norm[i,j,2] = 0
            else:
                mod_image_norm[i,j,0] = image_v[i,j,0]/mean_val[0]
                mod_image_norm[i,j,1] = image_v[i,j,1]/mean_val[1]
                mod_image_norm[i,j,2] = image_v[i,j,2]/mean_val[2]

    max_val = max(mod_image_norm[:,:,0].max(), mod_image_norm[:,:,1].max(), mod_image_norm[:,:,2].max())

    factor = 2.0
    mod_image = (mod_image_norm*((np.power(2.0,16.0)-1.0)/(factor*max_val))).astype(int)

    return mod_image

################################################################################

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

################################################################################

def path_minimum_algorithm(h_ref, px_ref, R_ref, G_ref, B_ref, px_exp, R_exp, G_exp, B_exp, neigh):

    de = np.zeros((px_ref, px_exp))

    for i in range(px_ref):
        for j in range(px_exp):
            de[i,j] = np.sqrt((R_ref[i]-R_exp[j])**2 + (G_ref[i]-G_exp[j])**2)
            # de[i,j] = np.sqrt((G_ref[i]-G_exp[j])**2 + (B_ref[i]-B_exp[j])**2)
            # de[i,j] = np.sqrt((B_ref[i]-B_exp[j])**2 + (R_ref[i]-R_exp[j])**2)


###########################

    sum_de = 0
    count = 0
    avg_de = np.zeros(px_ref)
    track = np.zeros((px_ref, px_exp))

    for i in range(px_ref):
        for j in range(px_exp):
            if j == 0:
                index_val = i
            else:
                val_start = index_val - neigh
                val_end   = index_val + neigh
                if val_start < 0:
                    val_start = 0
                if val_end > px_ref:
                    val_end = px_ref
                index_val = np.argmin(de[val_start:val_end,j]) + val_start
            sum_de = de[index_val,j] + sum_de
            count = count + 1
            track[i,j] = index_val
        avg_de[i] = sum_de/count

    start_pixel_value = np.argmin(avg_de)

###########################

    # min_val = np.zeros(px_ref)
    # min_h_exp = np.zeros(px_ref)
    #
    # max_val = np.zeros(px_exp)
    # max_h_exp = np.zeros(px_exp)
    #
    # for i in range(0, px_ref):
    #     min_val[i] = np.argmin(de[i,:])
    #     min_h_exp[i] = h_ref[np.argmin(de[i,:])]
    #
    # for j in range(0, px_exp):
    #     max_val[j] = np.argmin(de[:,j])
    #     max_h_exp[j] = h_ref[np.argmin(de[:,j])]
    #
    # h_exp = max_h_exp
    # index = max_val

    # h_exp = h_ref[int(track[start_pixel_value,:])]

    return de, start_pixel_value, track

################################################################################

################################################################################

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

################################################################################
