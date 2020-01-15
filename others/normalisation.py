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

# f_background = '/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/background_reference'
# os.chdir(f_background)
#
# background = mpimg.imread('background_reference__C001H001S0001000001.tif')
#
# # background = np.array(background_image)
#
# R_background = background[:,:,0]
# G_background = background[:,:,1]
# B_background = background[:,:,2]
#
# print(R_background.min(), R_background.max())
# print(G_background.min(), G_background.max())
# print(B_background.min(), B_background.max())

f = '/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/vertical_reference3'
os.chdir(f)

sRGB_image = mpim.imread('vertical_reference3__C001H001S0001000001.tif')
sRGB = np.array(sRGB_image)

x_px = np.shape(background)[1]
y_px = np.shape(background)[0]
ch   = np.shape(background)[2]

R = sRGB[:,:,0]
G = sRGB[:,:,1]
B = sRGB[:,:,2]

# plt.subplot(3,2,1)
# plt.imshow(background[:,:,0])
#
# plt.subplot(3,2,2)
# plt.imshow(sRGB[:,:,0]/background[:,:,0])
#
# plt.subplot(3,2,3)
# plt.imshow(background[:,:,1])
#
# plt.subplot(3,2,4)
# plt.imshow(sRGB[:,:,1]/background[:,:,1])
#
# plt.subplot(3,2,5)
# plt.imshow(background[:,:,2])
#
# plt.subplot(3,2,6)
# plt.imshow(sRGB[:,:,2]/background[:,:,2])
#
# plt.show()

R_div = np.zeros((y_px, x_px))
G_div = np.zeros((y_px, x_px))
B_div = np.zeros((y_px, x_px))

for i in range(x_px):
    for j in range(y_px):
        if R_background[j,i] == 0:
            R_div[j,i] = R[j,i]/1
        else:
            R_div[j,i] = R[j,i]/R_background[j,i]
        if G_background[j,i] == 0:
            G_div[j,i] = G[j,i]/1
        else:
            G_div[j,i] = G[j,i]/G_background[j,i]
        if B_background[j,i] == 0:
            B_div[j,i] = B[j,i]/1
        else:
            B_div[j,i] = B[j,i]/B_background[j,i]

R_div_final = (R_div-R_div.min())*100
G_div_final = (G_div-G_div.min())*100
B_div_final = (B_div-B_div.min())*100

# R_div_final = (R_div-R_div.min())*(((R.max())-1)/(R.max()-R.min()))
# G_div_final = (G_div-G_div.min())*(((G.max())-1)/(G.max()-G.min()))
# B_div_final = (B_div-B_div.min())*(((B.max())-1)/(B.max()-B.min()))

final_image_array = np.zeros((y_px, x_px, ch), dtype=np.uint8)

for i in range(x_px):
    for j in range(y_px):
        final_image_array[j,i] = [R_div_final[j,i], G_div_final[j,i], B_div_final[j,i]]

final_image = Image.fromarray(final_image_array, 'RGB')

plt.subplot(2,2,1)
plt.imshow(background)

plt.subplot(2,2,2)
plt.imshow(sRGB)

plt.subplot(2,2,3)
plt.imshow(final_image)

plt.show()

# input()
#
# if str(sRGB_image.dtype) == 'uint16':
#     bit_depth = 16
# elif str(sRGB_image.dtype) == 'uint8':
#     bit_depth = 8
#
# print(sRGB_image.dtype)
# input()
#
# R = sRGB_image[:,:,0]
# G = sRGB_image[:,:,1]
# B = sRGB_image[:,:,2]
#
# Rp = np.zeros((y_px, x_px))
# Gp = np.zeros((y_px, x_px))
# Bp = np.zeros((y_px, x_px))
#
# for i in range(x_px):
#     for j in range(y_px):
#         if R[j,i] <= 0.04045:
#             Rp[j,i] = R[j,i]/12.92
#         else:
#             Rp[j,i] = ((R[j,i]+0.055)/1.055)**2.4
#         if G[j,i] <= 0.04045:
#             Gp[j,i] = G[j,i]/12.92
#         else:
#             Gp[j,i] = ((G[j,i]+0.055)/1.055)**2.4
#         if B[j,i] <= 0.04045:
#             Bp[j,i] = B[j,i]/12.92
#         else:
#             Bp[j,i] = ((B[j,i]+0.055)/1.055)**2.4
#
# X = np.zeros((y_px, x_px))
# Y = np.zeros((y_px, x_px))
# Z = np.zeros((y_px, x_px))
#
# for i in range(x_px):
#     for j in range(y_px):
#         X[j,i] = 0.4360747*Rp[j,i] + 0.3850649*Gp[j,i] + 0.1430804*Bp[j,i]
#         Y[j,i] = 0.2225045*Rp[j,i] + 0.7168786*Gp[j,i] + 0.0606169*Bp[j,i]
#         Z[j,i] = 0.0139322*Rp[j,i] + 0.0971045*Gp[j,i] + 0.7141733*Bp[j,i]
