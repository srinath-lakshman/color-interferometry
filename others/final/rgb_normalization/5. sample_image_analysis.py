from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz
from mpl_toolkits.mplot3d import axes3d
import matplotlib.image as mpimg
from FUNC_ import sRGBtoLab
from FUNC_ import find_center

################################################################################

f_lateral = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

################################################################################

# f_background = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/background_reference'
# os.chdir(f_background)

# background_intensities = np.loadtxt('RGB_intensity_ratio.txt')
#
# b_R = background_intensities[0]
# b_G = background_intensities[1]
# b_B = background_intensities[2]

################################################################################

f_vertical_reference = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/vertical_reference1'
os.chdir(f_vertical_reference)

ref_R = np.loadtxt('ref_R.txt')
ref_G = np.loadtxt('ref_G.txt')
ref_B = np.loadtxt('ref_B.txt')
rr = np.loadtxt('rr.txt')
hh = np.loadtxt('hh.txt')

plt.plot(ref_R, ref_G)
plt.show()

################################################################################

f_image = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/sample_impact_over_dry_glass'
os.chdir(f_image)

rgb_image = io.imread('sample_impact_over_dry_glass__C001H001S0001000035.tif')
rgb_image_array = (rgb_image/((2**8)-1)).astype('uint8')

x_px  = np.shape(rgb_image)[0]
y_px  = np.shape(rgb_image)[1]
ch_px = np.shape(rgb_image)[2]

# threshold = 1.325*10000
# find_center(rgb_image,threshold)

xc = int(round((228+257)/2))
yc = int(round((307+333)/2))

# radius = round(min(xc, yc, x_px-1-xc, y_px-1-yc))
radius = 110

rgb_centered_mod = rgb_image[yc, xc-radius:xc+radius+1,:]
rgb_centered_mod_array = (rgb_centered_mod/((2**8)-1)).astype('uint8')

N1 = np.shape(rgb_centered_mod)[0]

rgb_exp = np.zeros(np.shape(rgb_centered_mod))

for i in range(N1):
    rgb_exp[i,0] = rgb_centered_mod[i,0]/(rgb_centered_mod[i,0]+rgb_centered_mod[i,1]+rgb_centered_mod[i,2])
    rgb_exp[i,1] = rgb_centered_mod[i,1]/(rgb_centered_mod[i,0]+rgb_centered_mod[i,1]+rgb_centered_mod[i,2])
    rgb_exp[i,2] = rgb_centered_mod[i,2]/(rgb_centered_mod[i,0]+rgb_centered_mod[i,1]+rgb_centered_mod[i,2])

N_exp = N1
N_ref = len(rr)

de = np.zeros((N_exp, N_ref))
h_air = np.zeros(N_exp)
index_final = np.zeros(N_exp)

# print(ref_image_a[0], ref_image_a[sq_side-1])
# print(de[N_exp-1, N_ref-1])
# input()

for i in range(N_exp):
    for j in range(N_ref):
        de[i,j] = np.sqrt((rgb_exp[i,0]-ref_R[j])**2 + (rgb_exp[i,1]-ref_G[j])**2)

# plt.imshow(de, cmap='gray')
# plt.show()

k = 1

avg_de = np.zeros(N_ref)
avg = 0
count = 0

for j in range(N_ref):
    start_pixel = j
    # print(j)
    for i in range(N_exp):
        if i == 0:
            index = start_pixel
            avg = de[i,index] + avg
            count = count + 1
        else:
            val_start = start_pixel - k
            val_end   = start_pixel + k
            if val_start < 0:
                val_start = 0
            if val_end > len(rr):
                val_end = len(rr)
            index = np.argmin(de[i,val_start:val_end+1]) + val_start
            avg = de[i,index] + avg
            count = count + 1
    avg_de[j] = avg/count
    # avg_de[j] = np.mean(de[i,val_start:val_end+1]) + de[0,val_start:val_end+1]

# plt.plot(avg_de)
# plt.scatter(range(len(avg_de)),avg_de)
# plt.show()

start_pixel_value = np.argmin(avg_de)

print(start_pixel_value)
# input()

for i in range(N_exp):
    if i == 0:
        index = start_pixel_value
    else:
        val_start = start_pixel_value - k
        val_end   = start_pixel_value + k
        if val_start < 0:
            val_start = 0
        if val_end > len(rr):
            val_end = len(rr)
        index = np.argmin(de[i,val_start:val_end+1]) + val_start
    index_final[i] = index
    h_air[i] = hh[index]

# plt.figure(0)
# plt.plot(avg_de)

plt.figure(0)
plt.imshow(de.T, cmap='gray')
plt.plot(range(N_exp), index_final)

plt.figure(1)
plt.plot(range(N_exp), h_air)

plt.show()
