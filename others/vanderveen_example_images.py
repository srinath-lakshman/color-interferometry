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

f = '/home/devici/Downloads/WLdropAnalysisMM'
os.chdir(f)

img1 = mpimg.imread('CIECAL8.tif')
# img2 = mpimg.imread('CIEIM8.tif')
img3 = mpimg.imread('colcal8nb.tif')

print(img1.dtype)
# print(img2.dtype)
print(img3.dtype)

print(np.shape(img1))
# print(np.shape(img2))
print(np.shape(img3))

print(img1[:,:,0].min(), img1[:,:,0].max())
# print(img2[:,:,0].min(), img2[:,:,0].max())
print(img3[:,:,0].min(), img3[:,:,0].max())

print(img1[:,:,1].min(), img1[:,:,1].max())
# print(img2[:,:,1].min(), img2[:,:,1].max())
print(img3[:,:,1].min(), img3[:,:,1].max())

print(img1[:,:,2].min(), img1[:,:,2].max())
# print(img2[:,:,2].min(), img2[:,:,2].max())
print(img3[:,:,2].min(), img3[:,:,2].max())

input()
