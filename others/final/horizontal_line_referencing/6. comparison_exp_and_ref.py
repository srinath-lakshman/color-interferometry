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
from FUNC_ import find_center

################################################################################

f_lateral = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

################################################################################

f_reference_values = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/reference_values'
os.chdir(f_reference_values)

r_ref = np.loadtxt('r_var.txt')
g_ref = np.loadtxt('g_var.txt')
rr_ref = np.loadtxt('rr.txt')
hh_ref = np.loadtxt('hh.txt')

# aa = 360
# bb = 8400
#
# rr_ref = np.loadtxt('r_final_total.txt')[aa:bb]
# hh_ref = np.loadtxt('h_final_total.txt')[aa:bb]
# r_ref = np.loadtxt('r_var_final_total.txt')[aa:bb]
# g_ref = np.loadtxt('g_var_final_total.txt')[aa:bb]

################################################################################

f_experimental_values = r'/home/devici/github/color_interferometry/final/horizontal_line_referencing/experimental_values'
os.chdir(f_experimental_values)

r_exp = np.loadtxt('r_exp_var.txt')
g_exp = np.loadtxt('g_exp_var.txt')
rr_exp = np.loadtxt('rr_exp.txt')

################################################################################

n_ref = len(rr_ref)
n_exp = len(rr_exp)

de = np.zeros((n_exp, n_ref))

for i in range(n_exp):
    for j in range(n_ref):
        de[i,j] = np.sqrt((r_exp[i]-r_ref[j])**2 + (g_exp[i]-g_ref[j])**2)

# val = np.zeros(n_exp)
# index = np.zeros(n_exp)
#
# for i in range(n_exp):
#     val[i] = min(de[i,:])
#     index[i] = np.argmin(de[i,:])

gradients = np.gradient(de)
x_grad = gradients[0]
y_grad = gradients[1]

ax = plt.subplot(1,1,1)
plt.imshow(de.T, cmap=plt.get_cmap('gray'))
plt.gca().invert_yaxis()
# plt.scatter(range(n_exp), index)
ax.set_xlabel('Experiment')
ax.set_ylabel('Reference')
ax.set_aspect('auto')

plt.show()

k = 100                                                                         # search min with k pixels only

avg_de = np.zeros(n_ref)
avg = 0
count = 0

for j in range(n_ref):
    start_pixel = j
    for i in range(n_exp):
        if i == 0:
            index = start_pixel
            avg = de[i,index] + avg
            count = count + 1
        else:
            val_start = start_pixel - k
            val_end   = start_pixel + k
            if val_start < 0:
                val_start = 0
            if val_end > n_ref:
                val_end = n_ref
            index = np.argmin(de[i,val_start:val_end+1]) + val_start
            avg = de[i,index] + avg
            count = count + 1
    avg_de[j] = avg/count

start_pixel_value = np.argmin(avg_de)
# print(start_pixel_value)

h_air = np.zeros(n_exp)
index_final = np.zeros(n_exp)

for i in range(n_exp):
    if i == 0:
        index = start_pixel_value
    else:
        val_start = start_pixel_value - k
        val_end   = start_pixel_value + k
        if val_start < 0:
            val_start = 0
        if val_end > n_ref:
            val_end = n_ref
        index = np.argmin(de[i,val_start:val_end+1]) + val_start
    index_final[i] = index
    h_air[i] = hh_ref[index]

plt.subplot(1,2,1)
plt.imshow(de.T, cmap=plt.get_cmap('gray'))
plt.gca().invert_yaxis()
# plt.scatter(range(n_exp), index)
plt.scatter(range(n_exp), index_final)
plt.xlabel('Experiment')
plt.ylabel('Reference')

plt.subplot(1,2,2)
plt.scatter(rr_exp,h_air)

plt.show()
