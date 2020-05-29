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

################################################################################

f_lateral = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

################################################################################

f_background = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/background_reference'
os.chdir(f_background)

background_intensities = np.loadtxt('RGB_intensity_ratio.txt')
mask = np.loadtxt('mask.txt')

b_R = background_intensities[0]
b_G = background_intensities[1]
b_B = background_intensities[2]

################################################################################

f_vertical_reference = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/vertical_reference1'
os.chdir(f_vertical_reference)

rgb_vertical_reference = io.imread('vertical_reference1__C001H001S0001000001.tif')

ff = 300                                                                        #focal length in millimeters
RR = 154.5

x_vertical_reference  = np.shape(rgb_vertical_reference)[0]
y_vertical_reference  = np.shape(rgb_vertical_reference)[1]
ch_vertical_reference = np.shape(rgb_vertical_reference)[2]

rgb_vertical_reference = np.array(rgb_vertical_reference, dtype=np.float16)
rgb_vertical_reference_norm = np.zeros(np.shape(rgb_vertical_reference), dtype=np.float16)
sum = np.zeros((x_vertical_reference, y_vertical_reference), dtype=np.float16)

for i in range(x_vertical_reference):
    for j in range(y_vertical_reference):
        sum[j,i] = rgb_vertical_reference[j,i,0] + rgb_vertical_reference[j,i,1] + rgb_vertical_reference[j,i,2]
        if sum[j,i] == 0:
            rgb_vertical_reference_norm[j,i,0] = 0
            rgb_vertical_reference_norm[j,i,1] = 0
            rgb_vertical_reference_norm[j,i,2] = 0
        else:
            rgb_vertical_reference_norm[j,i,0] = rgb_vertical_reference[j,i,0]/(sum[j,i])
            rgb_vertical_reference_norm[j,i,1] = rgb_vertical_reference[j,i,1]/(sum[j,i])
            rgb_vertical_reference_norm[j,i,2] = rgb_vertical_reference[j,i,2]/(sum[j,i])

center_coordinates = np.loadtxt('center_coordinates.txt')

xc = int(center_coordinates[0])
yc = int(center_coordinates[1])
radius = int(center_coordinates[2])

rgb_mod = rgb_vertical_reference_norm[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]

ref_R = average_profile(0, 360, 0.5, radius, radius, radius, rgb_mod[:,:,0])
ref_G = average_profile(0, 360, 0.5, radius, radius, radius, rgb_mod[:,:,1])
ref_B = average_profile(0, 360, 0.5, radius, radius, radius, rgb_mod[:,:,2])

plt.plot(ref_R)
plt.plot(ref_G)
plt.plot(ref_B)

rr = np.arange(0,radius*px,px)
hh = (RR - np.sqrt((RR*RR)-(rr*rr)))*1000

np.savetxt('ref_R.txt',ref_R)
np.savetxt('ref_G.txt',ref_G)
np.savetxt('ref_B.txt',ref_B)
np.savetxt('rr.txt',rr)
np.savetxt('hh.txt',hh)

plt.show()
