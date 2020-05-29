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
import cv2

f_lateral = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')

f_background = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/background_reference'
os.chdir(f_background)

background_intensities = np.loadtxt('RGB_intensity_ratio.txt')

f_vertical_reference = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/vertical_reference1'
os.chdir(f_vertical_reference)

rgb_vertical_reference = io.imread('vertical_reference1__C001H001S0001000001.tif')
rgb_vertical_reference_array = (rgb_vertical_reference/((2**8)-1)).astype('uint8')

x_vertical_reference  = np.shape(rgb_vertical_reference)[0]
y_vertical_reference  = np.shape(rgb_vertical_reference)[1]
ch_vertical_reference = np.shape(rgb_vertical_reference)[2]

rgb_vertical_reference_modified = np.zeros(np.shape(rgb_vertical_reference))

for i in range(x_vertical_reference):
    for j in range(y_vertical_reference):
        rgb_vertical_reference_modified[i,j,0] = rgb_vertical_reference[i,j,0]/background_intensities[0]
        rgb_vertical_reference_modified[i,j,1] = rgb_vertical_reference[i,j,1]/background_intensities[1]
        rgb_vertical_reference_modified[i,j,2] = rgb_vertical_reference[i,j,2]/background_intensities[2]

rgb_vertical_reference_modified[:,:,0] = ((rgb_vertical_reference_modified[:,:,0]/rgb_vertical_reference_modified[:,:,0].max())*((2**16)/2))
rgb_vertical_reference_modified[:,:,1] = ((rgb_vertical_reference_modified[:,:,1]/rgb_vertical_reference_modified[:,:,1].max())*((2**16)/2))
rgb_vertical_reference_modified[:,:,2] = ((rgb_vertical_reference_modified[:,:,2]/rgb_vertical_reference_modified[:,:,2].max())*((2**16)/2))

rgb_vertical_reference_modified_array = (rgb_vertical_reference_modified/((2**8)-1)).astype('uint8')

plt.figure(0)
plt.imshow(rgb_vertical_reference_array)

plt.figure(1)
plt.imshow(rgb_vertical_reference_modified_array)

plt.show()
