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
from FUNC_ import generate_axisymmetric
from PIL import Image
import pandas as pd

def avg_group(vA0, vB0):
    vA, ind, counts = np.unique(vA0, return_index=True, return_counts=True) # get unique values in vA0
    vB = vB0[ind]
    for dup in vB[counts>1]: # store the average (one may change as wished) of original elements in vA0 reference by the unique elements in vB
        vB[np.where(vA==dup)] = np.average(vB0[np.where(vA0==dup)])
    return vA, vB

################################################################################

f_lateral = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

################################################################################

f_background = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/background_reference'
os.chdir(f_background)

background_intensities = np.loadtxt('RGB_intensity_ratio.txt')
mask = np.loadtxt('mask.txt')

rgb_background_16 = io.imread('background_reference__C001H001S0001000001.tif')
rgb_background_16_array = (rgb_background_16/((2**8)-1)).astype('uint8')

b_R = background_intensities[0]
b_G = background_intensities[1]
b_B = background_intensities[2]

R_intensity = np.loadtxt('R_intensity.txt')
G_intensity = np.loadtxt('G_intensity.txt')
B_intensity = np.loadtxt('B_intensity.txt')

################################################################################

f_vertical_reference = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/vertical_reference1'
os.chdir(f_vertical_reference)

rgb_vertical_reference = io.imread('vertical_reference1__C001H001S0001000001.tif')
rgb_vertical_reference_array = (rgb_vertical_reference/((2**8)-1)).astype('uint8')

x_vertical_reference  = np.shape(rgb_vertical_reference)[0]
y_vertical_reference  = np.shape(rgb_vertical_reference)[1]
ch_vertical_reference = np.shape(rgb_vertical_reference)[2]

center_coordinates = np.loadtxt('center_coordinates.txt')

xc = int(center_coordinates[0])
yc = int(center_coordinates[1])
radius = int(center_coordinates[2])

rgb_vertical_reference_ratio = np.zeros(np.shape(rgb_vertical_reference))

for i in range(np.shape(mask)[0]):
    for j in range(np.shape(mask)[1]):
        if mask[i,j] == 0:
            rgb_vertical_reference_ratio[i,j,0] = 0
            rgb_vertical_reference_ratio[i,j,1] = 0
            rgb_vertical_reference_ratio[i,j,2] = 0
        else:
            rgb_vertical_reference_ratio[i,j,0] = rgb_vertical_reference[i,j,0]/R_intensity[i,j]
            rgb_vertical_reference_ratio[i,j,1] = rgb_vertical_reference[i,j,1]/G_intensity[i,j]
            rgb_vertical_reference_ratio[i,j,2] = rgb_vertical_reference[i,j,2]/B_intensity[i,j]

################################################################################
# For display purposes

max_val = max(rgb_vertical_reference_ratio[:,:,0].max(), rgb_vertical_reference_ratio[:,:,1].max(), rgb_vertical_reference_ratio[:,:,2].max())
rgb_vertical_reference_ratio_array = np.zeros(np.shape(rgb_vertical_reference))

for i in range(np.shape(mask)[0]):
    for j in range(np.shape(mask)[1]):
        rgb_vertical_reference_ratio_array[i,j,0] = int(round(((rgb_vertical_reference_ratio[i,j,0])/(max_val))*(2**7)))
        rgb_vertical_reference_ratio_array[i,j,1] = int(round(((rgb_vertical_reference_ratio[i,j,1])/(max_val))*(2**7)))
        rgb_vertical_reference_ratio_array[i,j,2] = int(round(((rgb_vertical_reference_ratio[i,j,2])/(max_val))*(2**7)))

rgb_vertical_reference_ratio_array = rgb_vertical_reference_ratio_array.astype(int)

# plt.subplot(1,3,1)
# plt.imshow(rgb_background_16_array)
#
# plt.subplot(1,3,2)
# plt.imshow(rgb_vertical_reference_array)
#
# plt.subplot(1,3,3)
# plt.imshow(rgb_vertical_reference_ratio_array)
#
# plt.show()

################################################################################

ff = 300                                                                        #focal length in millimeters
RR = 154.5                                                                      #lens curvature in millimeters

################################################################################

# count = 0
#
# r_total = np.zeros(np.power((2*radius) + 1,2))
# h_total = np.zeros(np.power((2*radius) + 1,2))
# r_var_total = np.zeros(np.power((2*radius) + 1,2))
# g_var_total = np.zeros(np.power((2*radius) + 1,2))
#
# for i in np.arange(xc-radius,xc+radius+1,1):
#     for j in np.arange(yc-radius,yc+radius+1,1):
#         dist = np.sqrt(((i-xc)**2)+((j-yc)**2))
#         sum = rgb_vertical_reference_ratio[i,j,0]+rgb_vertical_reference_ratio[i,j,1]+rgb_vertical_reference_ratio[i,j,2]
#         if sum != 0:
#             # print(i, j)
#             # print(count)
#             r_total[count] = dist*px
#             h_total[count] = (RR - np.sqrt((RR*RR)-(r_total[count]*r_total[count])))*1000
#             r_var_total[count] = rgb_vertical_reference_ratio[i,j,0]/sum
#             g_var_total[count] = rgb_vertical_reference_ratio[i,j,1]/sum
#             count = count + 1
#
# index_non_zeros = np.argwhere(r_total != 0)
# r_total1 = r_total[index_non_zeros]
# h_total1 = h_total[index_non_zeros]
# r_var_total1 = r_var_total[index_non_zeros]
# g_var_total1 = g_var_total[index_non_zeros]
#
# final_total = np.zeros((len(r_total1), 4))
#
# for i in range(len(r_total1)):
#     final_total[i,0] = r_total1[i]
#     final_total[i,1] = h_total1[i]
#     final_total[i,2] = r_var_total1[i]
#     final_total[i,3] = g_var_total1[i]
#
# final_total1 = final_total[final_total[:,0].argsort()]
#
# # np.savetxt('final_total1.txt', final_total1)
# # print('Completed!!')
# # input()
#
# [r_final_total, h_final_total] = avg_group(final_total1[:,0], final_total1[:,1])
# [_, r_var_final_total] = avg_group(final_total1[:,0], final_total1[:,1])
# [_, g_var_final_total] = avg_group(final_total1[:,0], final_total1[:,2])
#
# ax1 = plt.subplot(1,1,1)
# ax1.plot(final_total1[:,0], final_total1[:,2], color='red')
# ax1.plot(final_total1[:,0], final_total1[:,3], color='green')
# ax1.set_xlabel('r [mm]')
#
# ax2 = ax1.twinx()
# ax2.plot(final_total1[:,0], final_total1[:,1], color='black')
# ax2.set_ylabel(r'h [$\mu m$]')
#
# f_reference_values = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/reference_values'
# os.chdir(f_reference_values)
#
# np.savetxt('r_final_total.txt', r_final_total)
# np.savetxt('h_final_total.txt', h_final_total)
# np.savetxt('r_var_final_total.txt', r_var_final_total)
# np.savetxt('g_var_final_total.txt', g_var_final_total)
#
# plt.show()

################################################################################

rgb_vertical_reference_ratio_mod = rgb_vertical_reference_ratio[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]

ref_R_ratio = average_profile(0, 360, 0.5, radius, radius, radius, rgb_vertical_reference_ratio_mod[:,:,0])
ref_G_ratio = average_profile(0, 360, 0.5, radius, radius, radius, rgb_vertical_reference_ratio_mod[:,:,1])
ref_B_ratio = average_profile(0, 360, 0.5, radius, radius, radius, rgb_vertical_reference_ratio_mod[:,:,2])

rr_max = radius*px
rr = np.arange(0,rr_max,px)
hh = (RR - np.sqrt((RR*RR)-(rr*rr)))*1000

ax1 = plt.subplot(1,1,1)
ax1.plot(rr, ref_R_ratio, color='red')
ax1.plot(rr, ref_G_ratio, color='green')
ax1.plot(rr, ref_B_ratio, color='blue')
ax1.set_xlabel('r [mm]')

ax2 = ax1.twinx()
ax2.plot(rr, hh, color='black')
ax2.set_ylabel(r'h [$\mu m$]')

plt.show()
