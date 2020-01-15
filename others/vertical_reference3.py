from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz
from mpl_toolkits.mplot3d import axes3d

f_lateral = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

f_background = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/background_reference'
os.chdir(f_background)

background = plt.imread('background_reference__C001H001S0001000001.tif')

avg_red = np.mean(background[:,:,0])
avg_blue = np.mean(background[:,:,1])
avg_green = np.mean(background[:,:,2])

f = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/vertical_reference3'
os.chdir(f)

vertical_reference = plt.imread('vertical_reference3__C001H001S0001000001.tif')

rgb = vertical_reference
# rgb = vertical_reference - background
# rgb = vertical_reference - [avg_red, avg_blue, avg_green]
x_px = np.shape(rgb)[1]
y_px = np.shape(rgb)[0]
ch = np.shape(rgb)[2]

# fig = plt.gcf()
# fig.set_size_inches(10,10)

# plt.figure(0)
#
# plt.subplot(121)
# plt.imshow(rgb)
# plt.xlabel('X [px]')
# plt.xticks([0,x_px-1],[1,x_px])
# plt.ylabel('Y [px]')
# plt.yticks([0,y_px-1],[y_px,1])
# plt.title('Raw Image')
#
# threshold = 32
# plt.subplot(122)
# plt.imshow(rgb2gray(rgb)>threshold, cmap=plt.get_cmap('gray'))
# plt.xlabel('X [px]')
# # plt.xticks([0,x_px-1],[1,x_px])
# plt.ylabel('Y [px]')
# # plt.yticks([0,y_px-1],[y_px,1])

xc = round((254+294)/2)
yc = round((202+241)/2)

# plt.subplot(121)
# plt.scatter(xc, yc)
#
# plt.show()

radius = min(xc, yc, x_px-1-xc, y_px-1-yc)

rgb_mod = rgb[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]
sq_size = round((np.shape(rgb_mod)[0]+np.shape(rgb_mod)[1])/2)
delta = round((sq_size-1)/2)

fig = plt.figure(1)
fig.set_size_inches(25,15)

plt.subplot(121)
plt.imshow(rgb_mod, extent=[(-radius)*px, (+radius)*px, (-radius)*px, (+radius)*px])
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.title('Reference Image')

ff = 100                                                                        #focal length in millimeters
RR = ff/2
rr = np.arange(0,radius*px,px)
hh = (RR - np.sqrt((RR*RR)-(rr*rr)))*1000

ref_red = average_profile(0, 360, 0.5, delta, delta, delta, rgb_mod[:,:,0])
ref_green = average_profile(0, 360, 0.5, delta, delta, delta, rgb_mod[:,:,1])
ref_blue = average_profile(0, 360, 0.5, delta, delta, delta, rgb_mod[:,:,2])

ax1 = plt.subplot(222)
ax1.plot(np.arange(0,radius*px,px), ref_red, color='red', label='Red')
ax1.plot(np.arange(0,radius*px,px), ref_green, color='green', label='Green')
ax1.plot(np.arange(0,radius*px,px), ref_blue, color='blue', label='Blue')
ax1.set_xlim(0,radius*px)
ax1.set_xlabel('r [mm]')
ax1.set_ylabel('Color Intensity')
ax1.set_title(r'Calibration plot - RGB colorspace')
ax1.legend(loc='center left', bbox_to_anchor=(1.1,0.5))

ax2 = ax1.twinx()
ax2.plot(rr, hh, color='black')
ax2.set_ylabel(r'h [$\mu m$]')

lab_mod = color.rgb2lab(rgb_mod, illuminant='D65')

ref_l = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,0])
ref_a = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,1])
ref_b = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,2])

ax3 = plt.subplot(224)
ax3.plot(np.arange(0,radius*px,px), ref_l, color='red', label='L')
ax3.plot(np.arange(0,radius*px,px), ref_a, color='green', label='a')
ax3.plot(np.arange(0,radius*px,px), ref_b, color='blue', label='b')
ax3.set_xlim(0,radius*px)
ax3.set_xlabel('r [mm]')
ax3.set_ylabel('Color Intensity')
ax3.set_title(r'Calibration plot - Lab colorspace')
ax3.legend(loc='center left', bbox_to_anchor=(1.1,0.5))

ax4 = ax3.twinx()
ax4.plot(rr, hh, color='black', label='spherical lens')
ax4.set_ylabel(r'h [$\mu m$]')

# fig1 = plt.figure(2)
# ax5 = fig1.gca(projection='3d')
# ax5.scatter(ref_a,ref_b,hh)

fig.savefig('vertical_reference3_plot.tif')

np.savetxt('ref_l.txt',ref_l)
np.savetxt('ref_a.txt',ref_a)
np.savetxt('ref_b.txt',ref_b)
np.savetxt('rr.txt',rr)
np.savetxt('hh.txt',hh)

plt.show()
