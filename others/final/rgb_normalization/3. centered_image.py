from matplotlib import pyplot as plt
import os
from skimage import io
from FUNC_ import rgb2gray
from FUNC_ import find_center
import numpy as np

f_centered_image = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/vertical_reference1'
os.chdir(f_centered_image)

rgb_centered_image = io.imread('vertical_reference1__C001H001S0001000001.tif')
rgb_centered_image_array = (rgb_centered_image/((2**8)-1)).astype('uint8')

x_centered_image  = np.shape(rgb_centered_image)[0]
y_centered_image  = np.shape(rgb_centered_image)[1]
ch_centered_image = np.shape(rgb_centered_image)[2]

threshold = 75
find_center(rgb_centered_image_array,threshold)

xc = round((222+314)/2)
yc = round((207+300)/2)

# xc = 274
# yc = 264

pixel_numbers = round(min(xc, yc, x_centered_image-1-xc, y_centered_image-1-yc))
center_coordinates = [xc, yc, pixel_numbers]

np.savetxt('center_coordinates.txt',center_coordinates,fmt='%d')
