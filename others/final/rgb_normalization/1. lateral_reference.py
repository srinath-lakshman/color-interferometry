from matplotlib import pyplot as plt
import numpy as np
import os
from FUNC_ import rgb2gray
from FUNC_ import find_center
from skimage import io

f = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191023/exp2/lateral_reference'
os.chdir(f)

rgb = io.imread('lateral_reference__C001H001S0001000001.tif')

threshold = 30000
find_center(rgb,threshold)

delta_x = abs(270-451)
delta_y = abs(186-368)

print(delta_x, delta_y)

avg = (delta_x + delta_y)/2

length = 1/1000                                                                 #reference length in meters

px_meters = [length/avg]

np.savetxt('pixel_length.txt',px_meters,fmt='%.9f')
