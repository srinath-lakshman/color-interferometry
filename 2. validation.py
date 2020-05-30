import os
from mpl_toolkits import mplot3d as Axes3D
from matplotlib import cm
from FUNC import *
from skimage.graph import route_through_array
from scipy.signal import find_peaks

################################################################################

hard_disk   = r'F:/'
project     = r'color_interferometry/bottom_view/20200520/'

################################################################################

fr = hard_disk + project + r'calibration/info_f0400'
os.chdir(fr)

n_ref, sRGB_ref, Lab_ref, px_ref_microns = analysis_readme()
h_ref_microns = np.loadtxt('h_microns.txt')
r_mm = range(n_ref)*px_ref_microns*(1/1000.0)

f_rf0300 = hard_disk + project + r'calibration/info_f0300'
os.chdir(f_rf0300)

n_ref_rf0300, sRGB_ref_rf0300, Lab_ref_rf0300, px_ref_microns_rf0300 = analysis_readme()
h_ref_microns_rf0300 = np.loadtxt('h_microns.txt')
r_mm_rf0300 = range(n_ref_rf0300)*px_ref_microns_rf0300*(1/1000.0)

f_rf1000 = hard_disk + project + r'calibration/info_f1000'
os.chdir(f_rf1000)

n_ref_rf1000, sRGB_ref_rf1000, Lab_ref_rf1000, px_ref_microns_rf1000 = analysis_readme()
h_ref_microns_rf1000 = np.loadtxt('h_microns.txt')
r_mm_rf1000 = range(n_ref_rf1000)*px_ref_microns_rf1000*(1/1000.0)

################################################################################

os.chdir('..')
os.chdir('..')

################################################################################

RGB_ref_ratio = np.zeros((1,n_ref,3))
RGB_rf0300_ratio = np.zeros((1,n_ref_rf0300,3))
RGB_rf1000_ratio = np.zeros((1,n_ref_rf1000,3))

for i in range(n_ref):
    RGB_ref_ratio[0,i,0] = sRGB_ref[0,i,0]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])
    RGB_ref_ratio[0,i,1] = sRGB_ref[0,i,1]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])
    RGB_ref_ratio[0,i,2] = sRGB_ref[0,i,2]/(sRGB_ref[0,i,0]+sRGB_ref[0,i,1]+sRGB_ref[0,i,2])

for i in range(n_ref_rf0300):
    RGB_rf0300_ratio[0,i,0] = sRGB_ref_rf0300[0,i,0]/(sRGB_ref_rf0300[0,i,0]+sRGB_ref_rf0300[0,i,1]+sRGB_ref_rf0300[0,i,2])
    RGB_rf0300_ratio[0,i,1] = sRGB_ref_rf0300[0,i,1]/(sRGB_ref_rf0300[0,i,0]+sRGB_ref_rf0300[0,i,1]+sRGB_ref_rf0300[0,i,2])
    RGB_rf0300_ratio[0,i,2] = sRGB_ref_rf0300[0,i,2]/(sRGB_ref_rf0300[0,i,0]+sRGB_ref_rf0300[0,i,1]+sRGB_ref_rf0300[0,i,2])

for i in range(n_ref_rf1000):
    RGB_rf1000_ratio[0,i,0] = sRGB_ref_rf1000[0,i,0]/(sRGB_ref_rf1000[0,i,0]+sRGB_ref_rf1000[0,i,1]+sRGB_ref_rf1000[0,i,2])
    RGB_rf1000_ratio[0,i,1] = sRGB_ref_rf1000[0,i,1]/(sRGB_ref_rf1000[0,i,0]+sRGB_ref_rf1000[0,i,1]+sRGB_ref_rf1000[0,i,2])
    RGB_rf1000_ratio[0,i,2] = sRGB_ref_rf1000[0,i,2]/(sRGB_ref_rf1000[0,i,0]+sRGB_ref_rf1000[0,i,1]+sRGB_ref_rf1000[0,i,2])

de_RGB_rf0300 = np.zeros((n_ref, n_ref_rf0300))
de_Lab_rf0300 = np.zeros((n_ref, n_ref_rf0300))

de_RGB_rf1000 = np.zeros((n_ref, n_ref_rf1000))
de_Lab_rf1000 = np.zeros((n_ref, n_ref_rf1000))

for i in range(n_ref):
    for j in range(n_ref_rf0300):
        de_RGB_rf0300[i,j] = np.sqrt(((RGB_ref_ratio[0,i,0] - RGB_rf0300_ratio[0,j,0])**2) + ((RGB_ref_ratio[0,i,1] - RGB_rf0300_ratio[0,j,1])**2) + ((RGB_ref_ratio[0,i,2] - RGB_rf0300_ratio[0,j,2])**2))
        de_Lab_rf0300[i,j] = np.sqrt(((Lab_ref[0,i,1] - Lab_ref_rf0300[0,j,1])**2) + ((Lab_ref[0,i,2] - Lab_ref_rf0300[0,j,2])**2))

for i in range(n_ref):
    for j in range(n_ref_rf1000):
        de_RGB_rf1000[i,j] = np.sqrt(((RGB_ref_ratio[0,i,0] - RGB_rf1000_ratio[0,j,0])**2) + ((RGB_ref_ratio[0,i,1] - RGB_rf1000_ratio[0,j,1])**2) + ((RGB_ref_ratio[0,i,2] - RGB_rf1000_ratio[0,j,2])**2))
        de_Lab_rf1000[i,j] = np.sqrt(((Lab_ref[0,i,1] - Lab_ref_rf1000[0,j,1])**2) + ((Lab_ref[0,i,2] - Lab_ref_rf1000[0,j,2])**2))

[RR_rf0300,HH_rf0300] = np.meshgrid(r_mm_rf0300, h_ref_microns)
[RR_rf1000,HH_rf1000] = np.meshgrid(r_mm_rf1000, h_ref_microns)

if max(h_ref_microns) < max(h_ref_microns_rf0300):
    index_values_rf0300 = np.where(h_ref_microns_rf0300 > max(h_ref_microns))[0]
    cutoff_rf0300 = index_values_rf0300[0]
    x_max_index_rf0300 = cutoff_rf0300-1
    y_max_index_rf0300 = n_ref-1

if max(h_ref_microns) > max(h_ref_microns_rf1000):
    index_values_rf1000 = np.where(h_ref_microns > max(h_ref_microns_rf1000))[0]
    cutoff_rf1000 = index_values_rf1000[0]
    x_max_index_rf1000 = n_ref_rf1000-1
    y_max_index_rf1000 = cutoff_rf1000-1

indices, weight = route_through_array(de_Lab_rf0300, (0,0), (y_max_index_rf0300,x_max_index_rf0300))
indices1 = np.asarray(indices)
r_path_minimum_rf0300 = RR_rf0300[indices1[:,0],indices1[:,1]]
h_path_minimum_rf0300 = HH_rf0300[indices1[:,0],indices1[:,1]]

indices, weight = route_through_array(de_Lab_rf1000, (0,0), (y_max_index_rf1000,x_max_index_rf1000))
indices1 = np.asarray(indices)
r_path_minimum_rf1000 = RR_rf1000[indices1[:,0],indices1[:,1]]
h_path_minimum_rf1000 = HH_rf1000[indices1[:,0],indices1[:,1]]

# r_path_minimum_binned_rf0300 = r_path_minimum_rf0300[0:1100].reshape(110, 10).mean(axis=1)
# h_path_minimum_binned_rf0300 = h_path_minimum_rf0300[0:1100].reshape(110, 10).mean(axis=1)
# r_path_minimum_binned_rf1000 = r_path_minimum_rf1000[0:1100].reshape(110, 10).mean(axis=1)
# h_path_minimum_binned_rf1000 = h_path_minimum_rf1000[0:1100].reshape(110, 10).mean(axis=1)

r_path_minimum_binned_rf0300 = r_path_minimum_rf0300
h_path_minimum_binned_rf0300 = h_path_minimum_rf0300
r_path_minimum_binned_rf1000 = r_path_minimum_rf1000
h_path_minimum_binned_rf1000 = h_path_minimum_rf1000

fig, ax = plt.subplots(2,2,figsize=(20,20))

plt.subplot(2,2,1)
plt.pcolormesh(RR_rf0300, HH_rf0300, de_Lab_rf0300, cmap='gray')
plt.plot(r_path_minimum_rf0300, h_path_minimum_rf0300, color='white')
plt.xlabel('r [mm]')
plt.ylabel(r'h $[\mu m]$')
plt.title('Minimum path algorithm')

plt.subplot(2,2,2)
plt.pcolormesh(RR_rf1000, HH_rf1000, de_Lab_rf1000, cmap='gray')
plt.plot(r_path_minimum_rf1000, h_path_minimum_rf1000, color='white')
plt.xlabel('r [mm]')
plt.ylabel(r'h $[\mu m]$')
plt.title('Minimum path algorithm')

plt.subplot(2,2,3)
plt.plot(r_mm_rf0300[0:cutoff_rf0300], h_ref_microns_rf0300[0:cutoff_rf0300], color='red')
plt.scatter(r_path_minimum_binned_rf0300, h_path_minimum_binned_rf0300, marker='o', facecolors='none', edgecolors='black')
plt.xlabel('r [mm]')
plt.ylabel(r'h $[\mu m]$')

plt.subplot(2,2,4)
plt.plot(r_mm_rf1000[0:n_ref_rf1000], h_ref_microns_rf1000[0:n_ref_rf1000], color='red')
plt.scatter(r_path_minimum_binned_rf1000, h_path_minimum_binned_rf1000, marker='o', facecolors='none', edgecolors='black')
plt.xlabel('r [mm]')
plt.ylabel(r'h $[\mu m]$')

f = hard_disk + project + r'calibration'
os.chdir(f)

comparison_folder = os.getcwd() + '/comparison'
if os.path.exists(comparison_folder):
    print('\n**Comparison folder already exists**')
else:
    os.mkdir(comparison_folder)

os.chdir(comparison_folder)

np.savetxt("f0300_lens.txt", np.vstack((r_mm_rf0300[0:cutoff_rf0300],h_ref_microns_rf0300[0:cutoff_rf0300])), fmt='%f')
np.savetxt("f0300_lens_calculated.txt", np.vstack((r_path_minimum_binned_rf0300, h_path_minimum_binned_rf0300)), fmt='%f')
np.savetxt("f1000_lens.txt", np.vstack((r_mm_rf1000[0:n_ref_rf1000], h_ref_microns_rf1000[0:n_ref_rf1000])), fmt='%f')
np.savetxt("f1000_lens_calculated.txt", np.vstack((r_path_minimum_binned_rf1000, h_path_minimum_binned_rf1000)), fmt='%f')

fig.savefig('comparison.png', bbox_inches='tight')

os.chdir('..')

plt.show()

################################################################################
