from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz
from mpl_toolkits.mplot3d import axes3d
from skimage import data

# img = data.astronaut()
# img_xyz = color.rgb2xyz(img)
#
# plt.figure(0)
# plt.imshow(img)
#
# plt.figure(1)
# plt.imshow(img_xyz)
#
# plt.show()

def func_return(a):
    if a > 0.008856:
        b = a**(1/3)
    else:
        b = (7.787*a) + (16/166)
    return b

f_lateral = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

f = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/vertical_reference1'
os.chdir(f)

rgb_01 = io.imread('vertical_reference1_1__C001H001S0001000001.tif')
rgb_arr01 = np.array(rgb_01)
rgb_02 = io.imread('vertical_reference1__C001H001S0001000001.tif')
rgb_arr02 = np.array(rgb_02)

print(rgb_01.dtype)
print(rgb_arr01.dtype)

print(rgb_02.dtype)
print(rgb_arr02.dtype)

print(rgb_arr01[:,:,0].max())
print(rgb_arr02[:,:,0].max())

print(rgb_01[0,0,0] + rgb_02[0,0,0])

input()

# rgb = plt.imread('vertical_reference1_1__C001H001S0001000001.tif')              # 8 bit color depth each in R,G and B

# The min and max RGB values seem to reamin the same after converting the image to 8 or 16 bit
# which gives a hint that the values are with respect to a 12 bit original image

x_px = np.shape(rgb)[0]
y_px = np.shape(rgb)[1]
ch = np.shape(rgb)[2]

# To convert to a 8 bit image, one need to multiply 2^()

# print(rgb[:,:,0].min(), rgb[:,:,0].max())
# print(rgb[:,:,1].min(), rgb[:,:,1].max())
# print(rgb[:,:,2].min(), rgb[:,:,2].max())

# rgb = rgb*((2**8)/(2**12))

# print(rgb[:,:,0].min(), rgb[:,:,0].max())
# print(rgb[:,:,1].min(), rgb[:,:,1].max())
# print(rgb[:,:,2].min(), rgb[:,:,2].max())

print(x_px, y_px, ch)

conv = np.array([[0.49000, 0.31000, 0.20000], [0.17697, 0.81240, 0.01063], [0.00000, 0.01000, 0.99000]])

xyz_ref = color.rgb2xyz(rgb)

plt.figure(0)
plt.imshow(rgb)

plt.figure(1)
plt.imshow(xyz_ref)

plt.show()

print(rgb[0,0,:])
print(xyz_ref[0,0])

input()
