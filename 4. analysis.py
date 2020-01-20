import os
from FUNC import *
from skimage.graph import route_through_array
from scipy.signal import argrelextrema
from scipy.spatial import distance

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200114/'

################################################################################

f_ref = hard_disk + project + r'reference/info_f0400'
os.chdir(f_ref)

n_ref, sRGB_ref, Lab_ref, px_ref_microns = analysis_readme()
h_ref_microns = np.loadtxt('h_microns.txt')

f_exp = hard_disk + project + r'experiment/lower_speed_mica_run1/info/lower_speed_mica_run1_000092'
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

index_minima_horizontal = np.array(argrelextrema(de_Lab, np.less, axis=0)).T
index_minima_vertical = np.array(argrelextrema(de_Lab, np.less, axis=1)).T

R_minima_horizontal = RR[index_minima_horizontal[:,0], index_minima_horizontal[:,1]]
H_minima_horizontal = HH[index_minima_horizontal[:,0], index_minima_horizontal[:,1]]

R_minima_vertical = RR[index_minima_vertical[:,0], index_minima_vertical[:,1]]
H_minima_vertical = HH[index_minima_vertical[:,0], index_minima_vertical[:,1]]

# R_minima = np.concatenate([R_minima_horizontal,R_minima_vertical])
# H_minima = np.concatenate([H_minima_horizontal,H_minima_vertical])

R_minima = R_minima_horizontal
H_minima = H_minima_horizontal

plt.figure(1)
plt.pcolormesh(RR,HH,de_Lab, cmap='gray')
plt.xlabel(r'r $[mm]$')
plt.ylabel(r'h $[\mu m]$')
plt.xlim(0, max(r_exp_mm))
plt.ylim(0, max(h_ref_microns))
plt.show(block=False)

start_location = np.array(input('Enter start location [radius, height] = ').split(',')).astype('float')
R_start = R_minima[np.argmin((R_minima - start_location[0])**2)]
H_start = H_minima[np.argmin((H_minima - start_location[1])**2)]
start_location = [R_start, H_start]

end_location = np.array(input('Enter end location [radius, height] = ').split(',')).astype('float')
R_end = R_minima[np.argmin((R_minima - end_location[0])**2)]
H_end = H_minima[np.argmin((H_minima - end_location[1])**2)]
end_location = [R_end, H_end]

initial_path, initial_points = calculate_path_minimum_profile(de_Lab, start_location, end_location, r_exp_mm, h_ref_microns)

plt.figure(1)
plt.plot(initial_path[:,0],initial_path[:,1], linestyle='-', color='white')
plt.scatter(initial_points[:,0],initial_points[:,1], marker='o', color='red')
plt.show(block=False)

char = input('Additional locations (y/n)?: ')

points = [[initial_points[0,0],initial_points[0,1]], [initial_points[1,0],initial_points[1,1]]]

char = 'y'

while char == 'y':
    intermediate = np.array(input('Enter additonal location [radius, height] = ').split(',')).astype('float')
    R_intermediate = R_minima[np.argmin((R_minima - intermediate[0])**2)]
    H_intermediate = H_minima[np.argmin((H_minima - intermediate[1])**2)]

    points = np.vstack((points,[[R_intermediate, H_intermediate]]))
    points1 = points.tolist()
    points1.sort(key=lambda x: x[0])
    points = np.array(points1)

    k = np.shape(points)[0]

    r_path = []
    h_path = []

    for i in range(k-1):
        path, _ = calculate_path_minimum_profile(de_Lab, [ points[i,0], points[i,1] ], [ points[i+1,0] ,points[i+1,1] ], r_exp_mm, h_ref_microns)

        r_path = np.append(r_path, path[:,0])
        h_path = np.append(h_path, path[:,1])

    plt.close()
    plt.figure(1)
    plt.pcolormesh(RR,HH,de_Lab, cmap='gray')
    plt.plot(r_path,h_path, linestyle='-', color='white')
    plt.scatter(points[:,0],points[:,1], marker='o', color='red')
    plt.xlabel(r'r $[mm]$')
    plt.ylabel(r'h $[\mu m]$')
    plt.xlim(0, max(r_exp_mm))
    plt.ylim(0, max(h_ref_microns))
    plt.show(block=False)

    char = input('Additional locations (y/n)?: ')

os.chdir(f_exp)

path_points = np.array(points)
profile_unfiltered = np.array([r_path, h_path]).T

np.savetxt('path_points.txt', path_points, fmt='%0.6f')
np.savetxt('profile_unfiltered.txt', profile_unfiltered, fmt='%0.6f')

input('Extracting air profiles done!!')
plt.close()
