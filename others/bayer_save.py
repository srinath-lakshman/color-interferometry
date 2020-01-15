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
from PIL import Image
import cv2
import matplotlib.image as mpimg
# import rawpy

f = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191017/bayer_save'
os.chdir(f)

aa = cv2.imread('bayer_save_sensor_bit_tiff__C001H001S0001000001.tif')

# print(np.shape(aa))
# print(aa.dtype)

cv2.imshow('haha',aa)
cv2.wait(0)
