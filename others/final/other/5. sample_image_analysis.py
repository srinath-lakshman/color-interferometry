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

################################################################################

f_lateral = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

################################################################################

f_background = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/background_reference'
os.chdir(f_background)

background_intensities = np.loadtxt('RGB_intensity_ratio.txt')

b_R = background_intensities[0]
b_G = background_intensities[1]
b_B = background_intensities[2]

################################################################################

f_vertical_reference = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191020/vertical_reference1'
os.chdir(f_vertical_reference)

ref_l = np.loadtxt('ref_l.txt')
ref_a = np.loadtxt('ref_a.txt')
ref_b = np.loadtxt('ref_b.txt')
rr = np.loadtxt('rr.txt')
hh = np.loadtxt('hh.txt')

################################################################################

f_image = r'/media/devici/328C773C8C76F9A5/color_interferometry/bottom_view/20191009/oil_film_10mum_100_cst_h4_run1'
os.chdir(f_image)

rgb_image = io.imread('oil_film_10mum_100_cst_h4_run1__C001H001S0001000097.tif')
rgb_image_array = (rgb_image/((2**8)-1)).astype('uint8')

x_px  = np.shape(rgb_image)[0]
y_px  = np.shape(rgb_image)[1]
ch_px = np.shape(rgb_image)[2]

xc = int(round((243+321)/2))
yc = int(round((210+131)/2))

radius = round(min(xc, yc, x_px-1-xc, y_px-1-yc))

rgb_centered_mod = rgb_image[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]/[b_R, b_G, b_B]
rgb_centered_mod_array = (rgb_centered_mod/((2**8)-1)).astype('uint8')

ref_R = average_profile(0, 360, 0.5, radius, radius, radius, rgb_centered_mod[:,:,0])
ref_G = average_profile(0, 360, 0.5, radius, radius, radius, rgb_centered_mod[:,:,1])
ref_B = average_profile(0, 360, 0.5, radius, radius, radius, rgb_centered_mod[:,:,2])

sq_side = int(round(radius/np.sqrt(2)))

rgb_mod_axisymmetric = np.zeros((2*sq_side+1, 2*sq_side+1, ch_px))

count_x = -1

for i in range(-sq_side, sq_side+1):
    count_x = count_x + 1
    count_y = -1
    for j in range(-sq_side, sq_side+1):
        count_y = count_y + 1
        dist = int(round(np.sqrt((np.power(i,2)) + (np.power(j,2)))))-2
        rgb_mod_axisymmetric[count_y,count_x,0] = ref_R[dist]
        rgb_mod_axisymmetric[count_y,count_x,1] = ref_G[dist]
        rgb_mod_axisymmetric[count_y,count_x,2] = ref_B[dist]

rgb_mod_axisymmetric_array = (rgb_mod_axisymmetric/((2**8)-1)).astype('uint8')

plt.figure(0)
plt.imshow(rgb_centered_mod_array)

plt.figure(1)
plt.imshow(rgb_mod_axisymmetric_array)

plt.show()

lab = sRGBtoLab(rgb_mod_axisymmetric, 16, 'D50', 'D50')
lab_mod = lab

ref_image_l = average_profile(0, 360, 0.5, sq_side, sq_side, sq_side, lab_mod[:,:,0])
ref_image_a = average_profile(0, 360, 0.5, sq_side, sq_side, sq_side, lab_mod[:,:,1])
ref_image_b = average_profile(0, 360, 0.5, sq_side, sq_side, sq_side, lab_mod[:,:,2])

plt.plot(ref_a, ref_b)
plt.plot(ref_image_a, ref_image_b)
plt.show()

N_exp = sq_side
N_ref = len(rr)

# print(N_exp, N_ref)
# print(len(ref_image_l), len(ref_l), len(hh))
# input()

de = np.zeros((N_exp, N_ref))
h_air = np.zeros(N_exp)
index_final = np.zeros(N_exp)

# print(ref_image_a[0], ref_image_a[sq_side-1])
# print(de[N_exp-1, N_ref-1])
# input()

for i in range(N_exp):
    for j in range(N_ref):
        de[i,j] = np.sqrt((ref_image_a[i]-ref_a[j])**2 + (ref_image_b[i]-ref_b[j])**2)

k = 8

avg_de = np.zeros(N_ref)
avg = 0
count = 0

for j in range(N_ref):
    start_pixel = j
    # print(j)
    for i in range(N_exp):
        if i == 0:
            index = start_pixel
            avg = de[i,index] + avg
            count = count + 1
        else:
            val_start = start_pixel - k
            val_end   = start_pixel + k
            if val_start < 0:
                val_start = 0
            if val_end > len(ref_l):
                val_end = len(ref_l)
            index = np.argmin(de[i,val_start:val_end+1]) + val_start
            avg = de[i,index] + avg
            count = count + 1
    avg_de[j] = avg/count
    # avg_de[j] = np.mean(de[i,val_start:val_end+1]) + de[0,val_start:val_end+1]

# plt.plot(avg_de)
# plt.scatter(range(len(avg_de)),avg_de)
# plt.show()

start_pixel_value = np.argmin(avg_de)

print(start_pixel_value)
# input()

for i in range(N_exp):
    if i == 0:
        index = start_pixel_value
    else:
        val_start = start_pixel_value - k
        val_end   = start_pixel_value + k
        if val_start < 0:
            val_start = 0
        if val_end > len(ref_l):
            val_end = len(ref_l)
        index = np.argmin(de[i,val_start:val_end+1]) + val_start
    index_final[i] = index
    h_air[i] = hh[index]

# plt.figure(0)
# plt.plot(avg_de)

plt.figure(0)
plt.imshow(de.T, cmap='gray')
plt.plot(range(N_exp), index_final)

plt.figure(1)
plt.plot(range(N_exp), h_air)

plt.show()
