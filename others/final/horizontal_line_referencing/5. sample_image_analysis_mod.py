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

f_image = r'//home/devici/github/color_interferometry/final/horizontal_line_referencing/sample_impact_over_dry_glass'
os.chdir(f_image)

rgb_image = io.imread('sample_impact_over_dry_glass__C001H001S0001000051.tif')
# rgb_image_array = (rgb_image/((2**8)-1)).astype('uint8')

threshold = 1.88*(10**4)
find_center(rgb_image,threshold)

x_px  = np.shape(rgb_image)[0]
y_px  = np.shape(rgb_image)[1]
ch_px = np.shape(rgb_image)[2]

xc = int(round((240+326)/2))
yc = int(round((207+296)/2))

radius = int(round(min(xc, yc, x_px-1-xc, y_px-1-yc)))

rgb_sample_ratio = np.zeros(np.shape(rgb_image))

for i in range(np.shape(mask)[0]):
    for j in range(np.shape(mask)[1]):
        if mask[i,j] == 0:
            rgb_sample_ratio[i,j,0] = 0
            rgb_sample_ratio[i,j,1] = 0
            rgb_sample_ratio[i,j,2] = 0
        else:
            rgb_sample_ratio[i,j,0] = rgb_image[i,j,0]/R_intensity[i,j]
            rgb_sample_ratio[i,j,1] = rgb_image[i,j,1]/G_intensity[i,j]
            rgb_sample_ratio[i,j,2] = rgb_image[i,j,2]/B_intensity[i,j]

max_rr = radius*px
rr = np.arange(0,max_rr,px)

rgb_sample_line = rgb_sample_ratio[yc, xc:xc+radius,:]
n_exp = np.shape(rgb_sample_line)[0]

r_exp_var = np.zeros(n_exp)
g_exp_var = np.zeros(n_exp)

for i in range(n_exp):
    r_exp_var[i] = rgb_sample_line[i,0]/(rgb_sample_line[i,0]+rgb_sample_line[i,1]+rgb_sample_line[i,2])
    g_exp_var[i] = rgb_sample_line[i,1]/(rgb_sample_line[i,0]+rgb_sample_line[i,1]+rgb_sample_line[i,2])

ax1 = plt.subplot(1,1,1)
# ax1.plot(rr, rgb_line[:,0], color='red')
# ax1.plot(rr, rgb_line[:,1], color='green')
# ax1.plot(rr, rgb_line[:,2], color='blue')
ax1.plot(rr, r_exp_var, color='red')
ax1.plot(rr, g_exp_var, color='green')
ax1.set_xlim(0,max_rr)
ax1.set_xlabel('r [mm]')

f_experimental_values = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/experimental_values'
os.chdir(f_experimental_values)

np.savetxt('r_exp_var.txt',r_exp_var)
np.savetxt('g_exp_var.txt',g_exp_var)
np.savetxt('rr_exp.txt',rr)

plt.show()
