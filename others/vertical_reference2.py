from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz

f = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/vertical_reference2'
os.chdir(f)

rgb = plt.imread('vertical_reference2__C001H001S0001000001.tif')
x_px = np.shape(rgb)[0]
y_px = np.shape(rgb)[1]
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
# threshold = 100
# plt.subplot(122)
# plt.imshow(rgb2gray(rgb)>threshold, cmap=plt.get_cmap('gray'))
# plt.xlabel('X [px]')
# plt.xticks([0,x_px-1],[1,x_px])
# plt.ylabel('Y [px]')
# plt.yticks([0,y_px-1],[y_px,1])
#
# plt.show()

px = 7.575/1000                                                                 #pixel length in millimeters

xc = round((245+313)/2)
yc = round((255+323)/2)

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

ff = 300                                                                        #focal length in millimeters
# RR = 2*ff
RR = 154.5
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
ax4.grid()

# fig1 = plt.figure(2)
# ax5 = fig1.gca(projection='3d')
# ax5.scatter(ref_a,ref_b,hh)

# fig.savefig('vertical_reference2_plot.tif')
#
# np.savetxt('ref_l.txt',ref_l)
# np.savetxt('ref_a.txt',ref_a)
# np.savetxt('ref_b.txt',ref_b)
# np.savetxt('rr.txt',rr)
# np.savetxt('hh.txt',hh)

plt.show()
