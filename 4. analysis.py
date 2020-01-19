import os
from mpl_toolkits import mplot3d as Axes3D
from matplotlib import cm
from FUNC import *
from skimage.graph import route_through_array
import scipy.fftpack
from scipy.signal import savgol_filter

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200114/'

################################################################################

f_ref = hard_disk + project + r'reference/info_f0400'
os.chdir(f_ref)

n_ref, sRGB_ref, Lab_ref, px_ref_microns = analysis_readme()
h_ref_microns = np.loadtxt('h_microns.txt')

f_exp = hard_disk + project + r'experiment/higher_speed_mica_run1/info/higher_speed_mica_run1_000155'
os.chdir(f_exp)

n_exp, sRGB_exp, Lab_exp, px_exp_microns = analysis_readme()
r_exp_mm = range(n_exp)*px_exp_microns*(1/1000.0)

################################################################################

os.chdir('..')
os.chdir('..')

################################################################################

[RR,HH] = np.meshgrid(r_exp_mm, h_ref_microns)

RGB_ref_ratio = np.zeros((1,n_ref,3))
RGB_exp_ratio = np.zeros((1,n_exp,3))

for i in range(n_ref):
    RGB_ref_ratio[0,i,0] = sRGB_ref[0,i,0]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])
    RGB_ref_ratio[0,i,1] = sRGB_ref[0,i,1]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])
    RGB_ref_ratio[0,i,2] = sRGB_ref[0,i,2]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])

for i in range(n_exp):
    RGB_exp_ratio[0,i,0] = sRGB_exp[0,i,0]/(sRGB_exp[0,i,0]+sRGB_exp[0,i,1]+sRGB_exp[0,i,2])
    RGB_exp_ratio[0,i,1] = sRGB_exp[0,i,1]/(sRGB_exp[0,i,0]+sRGB_exp[0,i,1]+sRGB_exp[0,i,2])
    RGB_exp_ratio[0,i,2] = sRGB_exp[0,i,2]/(sRGB_exp[0,i,0]+sRGB_exp[0,i,1]+sRGB_exp[0,i,2])

de_RGB = np.zeros((n_ref, n_exp))
de_Lab = np.zeros((n_ref, n_exp))

for i in range(n_ref):
    for j in range(n_exp):
        de_RGB[i,j] = np.sqrt(((RGB_ref_ratio[0,i,0] - RGB_exp_ratio[0,j,0])**2) + ((RGB_ref_ratio[0,i,1] - RGB_exp_ratio[0,j,1])**2) + ((RGB_ref_ratio[0,i,2] - RGB_exp_ratio[0,j,2])**2))
        de_Lab[i,j] = np.sqrt(((Lab_ref[0,i,1] - Lab_exp[0,j,1])**2) + ((Lab_ref[0,i,2] - Lab_exp[0,j,2])**2))

################################################################################

plt.subplot(2,1,1)
plt.pcolormesh(RR,HH,de_Lab, cmap='gray')
plt.xlabel(r'r $[mm]$')
plt.ylabel(r'h $[\mu m]$')

plt.subplot(2,2,3)
plt.scatter(de_Lab[:,0], h_ref_microns, marker='.', color='black')
plt.title('Start Intensity Profile')

plt.subplot(2,2,4)
plt.scatter(de_Lab[:,n_exp-1], h_ref_microns, marker='.', color='black')
plt.title('End Intensity Profile')

plt.show(block=False)

input()
print('Minimum path length algorithm')
# start_location = np.array(input('Start height range = ').split(',')).astype('float')
# end_location = np.array(input('End height range = ').split(',')).astype('float')

# Lower impact speed
# image 'lower_speed_mica_run1_000092.tif'
# start_location = [2.925, 2.915]
# end_location = [0.601, 0.599]

# Higher impact speed
# image 'higher_speed_mica_run1_000155.tif'

start_location = [2.336, 2.332]
end_location = [0.3575, 0.3570]

# start_location = [2.585, 2.583]
# end_location = [0.608, 0.606]

start_index = int(np.where((h_ref_microns > min(start_location)) & (h_ref_microns < max(start_location)))[0])
end_index = int(np.where((h_ref_microns > min(end_location)) & (h_ref_microns < max(end_location)))[0])

# fully_connected == True means diagonal moves are permitted. If False, only axial (x and y) moves allowed.
# geometric == True means diagonal distances are incorporated. If False, diagonal distances are ignored.
indices, weight = route_through_array(de_Lab, (start_index,0), (end_index,n_exp-1),fully_connected=True, geometric=False)
indices1 = np.asarray(indices)
r_path_minimum_exp = RR[indices1[:,0],indices1[:,1]]
h_path_minimum_exp = HH[indices1[:,0],indices1[:,1]]

h_smooth = h_path_minimum_exp

plt.close()

plt.subplot(2,2,1)
plt.pcolormesh(RR,HH,de_Lab, cmap='gray')
plt.xlabel(r'r $[mm]$')
plt.ylabel(r'h $[\mu m]$')

plt.subplot(2,4,5)
plt.scatter(de_Lab[:,0], h_ref_microns, marker='.', color='black')
plt.axhline(y=h_ref_microns[start_index], linestyle='--', color='black')
plt.title('Start Intensity Profile')

plt.subplot(2,4,6)
plt.scatter(de_Lab[:,n_exp-1], h_ref_microns, marker='.', color='black')
plt.axhline(y=h_ref_microns[end_index], linestyle='--', color='black')
plt.title('End Intensity Profile')

plt.subplot(2,2,2)
plt.pcolormesh(RR,HH,de_Lab, cmap='gray')
plt.plot(r_path_minimum_exp, h_path_minimum_exp, linestyle='-', color='white')
plt.xlabel(r'r $[mm]$')
plt.ylabel(r'h $[\mu m]$')

plt.subplot(2,2,4)
plt.plot(r_path_minimum_exp, h_smooth, linestyle='-', color='black')
plt.xlabel(r'r $[mm]$')
plt.ylabel(r'h $[\mu m]$')

plt.show(block=False)

os.chdir(f_exp)

start_index_path = [0, h_ref_microns[start_index]]
end_index_path = [r_exp_mm[n_exp-1], h_ref_microns[end_index]]

path_endpoints = [start_index_path, end_index_path]
profile_unfiltered = np.array([r_path_minimum_exp, h_path_minimum_exp]).T

np.savetxt('path_endpoints.txt', path_endpoints, fmt='%0.6f')
np.savetxt('profile_unfiltered.txt', profile_unfiltered, fmt='%0.6f')

input('Extracting air profiles done!!')
plt.close()
