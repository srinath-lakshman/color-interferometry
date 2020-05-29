from matplotlib import pyplot as plt
import numpy as np
import os
from skimage import io, color
from FUNC_ import average_profile
from FUNC_ import image_profile
from FUNC_ import rgb2gray
from FUNC_ import srgbtoxyz
from mpl_toolkits.mplot3d import axes3d

px = 7.575/1000                                                                 #pixel length in millimeters

f1 = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/vertical_reference1'
#f1000

os.chdir(f1)

ref_l1 = np.loadtxt('ref_l.txt')
ref_a1 = np.loadtxt('ref_a.txt')
ref_b1 = np.loadtxt('ref_b.txt')

hh1 = np.loadtxt('hh.txt')
rr1 = np.loadtxt('rr.txt')

f2 = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/vertical_reference2'
#f300

os.chdir(f2)

ref_l2 = np.loadtxt('ref_l.txt')
ref_a2 = np.loadtxt('ref_a.txt')
ref_b2 = np.loadtxt('ref_b.txt')

hh2 = np.loadtxt('hh.txt')
rr2 = np.loadtxt('rr.txt')

# fig = plt.figure(0)
# ax = fig.gca()
# # ax = fig.gca(projection='3d')
# ax.scatter(ref_a1,ref_b1,hh1*30)
# ax.scatter(ref_a2,ref_b2,hh2*30)

min_value = -0.025
max_value = 3.00

plt.figure(1)
plt.subplot(121)
plt.scatter(ref_a1,hh1)
plt.scatter(ref_a2,hh2)
plt.ylim(min_value,max_value)
plt.grid()

plt.subplot(122)
plt.scatter(ref_b1,hh1)
plt.scatter(ref_b2,hh2)
plt.ylim(min_value,max_value)
plt.grid()

plt.show()
