from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz

f_lateral = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

f1 = r'/media/devici/Samsung_T5/color_interferometry/20191004/vertical_reference1'
f2 = r'/media/devici/Samsung_T5/color_interferometry/20191004/vertical_reference2'
f3 = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/vertical_reference3'
os.chdir(f3)

ref_l = np.loadtxt('ref_l.txt')
ref_a = np.loadtxt('ref_a.txt')
ref_b = np.loadtxt('ref_b.txt')

hh = np.loadtxt('hh.txt')
rr = np.loadtxt('rr.txt')

plt.plot(ref_a, ref_b)
plt.show()

len_ref = len(rr)
rgb_ref = np.zeros((len_ref, 3))

for i in np.arange(0,len_ref,1):
    rgb_ref[i,:] = color.lab2rgb([[[ref_l[i], ref_a[i], ref_b[i]]]], illuminant='D65')

plt.imshow(rgb_ref)
plt.show()

ff = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/oil_film_10mum_100_cst_h4_run1'
os.chdir(ff)

rgb = plt.imread('oil_film_10mum_100_cst_h4_run1__C001H001S0001000097.tif')
x_px = np.shape(rgb)[1]
y_px = np.shape(rgb)[0]
ch = np.shape(rgb)[2]

# fig = plt.gcf()
# fig.set_size_inches(10,10)

plt.figure(0)

plt.subplot(121)
plt.imshow(rgb)
plt.xlabel('X [px]')
plt.xticks([0,x_px-1],[1,x_px])
plt.ylabel('Y [px]')
plt.yticks([0,y_px-1],[y_px,1])
plt.title('Raw Image')

threshold = 46
plt.subplot(122)
plt.imshow(rgb2gray(rgb)>threshold, cmap=plt.get_cmap('gray'))
plt.xlabel('X [px]')
plt.xticks([0,x_px-1],[1,x_px])
plt.ylabel('Y [px]')
plt.yticks([0,y_px-1],[y_px,1])

plt.show()

# xc = round((249+302)/2)
# yc = round((216+270)/2)

xc = 282
yc = 172

radius = min(xc, yc, x_px-1-xc, y_px-1-yc)

rgb_mod = rgb[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]
sq_size = round((np.shape(rgb_mod)[0]+np.shape(rgb_mod)[1])/2)
delta = round((sq_size-1)/2)

# plt.figure(1)
#
# plt.subplot(121)
# plt.imshow(rgb_mod, extent=[(-radius)*px, (+radius)*px, (-radius)*px, (+radius)*px])
# plt.xlabel('X [mm]')
# plt.ylabel('Y [mm]')
# plt.title('Reference Image')

lab_mod = color.rgb2lab(rgb_mod, illuminant='D65')

l = average_profile(-90+45, -90-45, 0.5, delta, delta, delta, lab_mod[:,:,0])
a = average_profile(-90+45, -90-45, 0.5, delta, delta, delta, lab_mod[:,:,1])
b = average_profile(-90+45, -90-45, 0.5, delta, delta, delta, lab_mod[:,:,2])

# ax1 = plt.subplot(122)
# ax1.plot(np.arange(0,radius*px,px), l, color='red', label='L')
# ax1.plot(np.arange(0,radius*px,px), a, color='green', label='a')
# ax1.plot(np.arange(0,radius*px,px), b, color='blue', label='b')
# ax1.set_xlim(0,radius*px)
# ax1.set_xlabel('r [mm]')
# ax1.set_ylabel('Color Intensity')
# ax1.set_title(r'Bouncing air film - Lab colorspace')

N_exp = delta
# N_exp = round(0.9/px)
# Array_exp = np.arange(N_exp)

# Array_ref = np.arange(15,200,1)
# N_ref = len(Array_ref)
N_ref = len(hh)

de = np.zeros((N_exp, len(hh)))
h_air = np.zeros(N_exp)
index_final = np.zeros(N_exp)

for i in np.arange(N_exp):
    for j in np.arange(len(hh)):
        de[i,j] = np.sqrt((a[i]-ref_a[j])**2 + (b[i]-ref_b[j])**2)

# haha1 = de[:,15:30]

for i in np.arange(N_exp):
    # index = 15 + np.argmin(haha1[i,:])
    index = np.argmin(de[i,:])
    index_final[i] = index
    h_air[i] = hh[index]
    # h_air[i] = hh[Array_ref[index]]
    # if index < 50:
        # index_final[i] = index
        # h_air[i] = hh[index]
        # h_air[i] = hh[Array_ref[index]]
    # else:
        # index_final[i] = None
        # h_air[i] = None

plt.figure(4)
plt.imshow(np.transpose(de), cmap='gray')

plt.figure(3)
plt.imshow(np.transpose(de), cmap='gray')
plt.scatter(np.arange(0,delta,1), index_final[:])
# plt.scatter(np.arange(0,round(0.9/px),1), index_final[:])
# plt.scatter(np.arange(0,round(0.9/px),1), index_final[:] )

plt.figure(2)
plt.plot(np.arange(0,delta,1), h_air)
# plt.plot(np.arange(0,round(0.9/px),1), h_air)

plt.show()
