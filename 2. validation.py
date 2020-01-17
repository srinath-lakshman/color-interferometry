import os
from mpl_toolkits import mplot3d as Axes3D
from matplotlib import cm
from FUNC import *
from skimage.graph import route_through_array
from scipy.signal import find_peaks

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200114/'

################################################################################

fr = hard_disk + project + r'reference/info'
os.chdir(fr)

n_ref, sRGB_ref, Lab_ref, px_ref_microns = analysis_readme()
h_ref_microns = np.loadtxt('h_microns.txt')

f_rf0300 = hard_disk + project + r'reference/info_f0300'
os.chdir(f_rf0300)

n_ref_rf0300, sRGB_ref_rf0300, Lab_ref_rf0300, px_ref_microns_rf0300 = analysis_readme()
h_ref_microns_rf0300 = np.loadtxt('h_microns.txt')

f_rf1000 = hard_disk + project + r'reference/info_f1000'
os.chdir(f_rf1000)

n_ref_rf1000, sRGB_ref_rf1000, Lab_ref_rf1000, px_ref_microns_rf1000 = analysis_readme()
h_ref_microns_rf1000 = np.loadtxt('h_microns.txt')

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

h_mod = h_ref_microns[0:n_ref]
r_rf0300_mm = range(n_ref_rf0300)*px_ref_microns_rf0300*(1/1000.0)
r_rf1000_mm = range(n_ref_rf1000)*px_ref_microns_rf1000*(1/1000.0)

[RR_rf0300,HH_rf0300] = np.meshgrid(r_rf0300_mm, h_mod)
[RR_rf1000,HH_rf1000] = np.meshgrid(r_rf1000_mm, h_mod)

# plt.subplot(1,2,1)
# # plt.pcolormesh(RR_rf0300, HH_rf0300, de_RGB_rf0300, cmap='gray')
# plt.pcolormesh(RR_rf0300, HH_rf0300, de_Lab_rf0300, cmap='gray')
#
# plt.subplot(1,2,2)
# # plt.pcolormesh(RR_rf1000, HH_rf1000, de_RGB_rf1000, cmap='gray')
# plt.pcolormesh(RR_rf1000, HH_rf1000, de_Lab_rf1000, cmap='gray')
#
# plt.show()

################################################################################

[_, min_start_experiment] = peakdetect(de_Lab_rf0300[:,0], x_axis = None, lookahead = 20, delta=0)
min_start_experiment = np.array(min_start_experiment)

[_, min_end_experiment] = peakdetect(de_Lab_rf0300[:,n_ref_rf0300-1], x_axis = None, lookahead = 20, delta=0)
min_end_experiment = np.array(min_end_experiment)

# print(min_start_experiment)
# print('haha')
# print((int(min_start_experiment[0,0]),0))
# print((int(min_end_experiment[0,0]),0))
# print(n_ref_rf0300-1)
#
# plt.subplot(1,2,1)
# plt.plot(de_Lab_rf0300[:,0])
# plt.scatter(min_start_experiment[:,0], min_start_experiment[:,1])
#
# plt.subplot(1,2,2)
# plt.plot(de_Lab_rf0300[:,n_ref_rf0300-1])
# plt.scatter(min_end_experiment[:,0], min_end_experiment[:,1])
#
# plt.show()

# indices, weight = route_through_array(de_Lab_rf0300, (int(min_start_experiment[1,0]),0), (int(min_end_experiment[0,0]),n_ref_rf0300-1))
indices, weight = route_through_array(de_Lab_rf0300, (0,0), (959,1021))

indices1 = np.asarray(indices)
# input()

plt.figure(2)
# plt.imshow(de_Lab_rf0300, cmap='gray')
plt.pcolormesh(RR_rf0300, HH_rf0300, de_Lab_rf0300, cmap='gray')
plt.scatter(indices1[:,1], indices1[:,0], color='red')
plt.gca().invert_yaxis()

plt.show()

print('haha')
input()

# len_start = np.shape(min_start_experiment)[0]
# len_end = np.shape(min_end_experiment)[0]

sum_weight = 100

for i in range(len_start):
    for j in range(len_end):
        indices, weight = route_through_array(de_RGB, (int(min_start_experiment[i,0]),0), (int(min_end_experiment[j,0]),n_exp-1), fully_connected=False, geometric=False)
        indices = np.array(indices)

        # plt.imshow(de_RGB, cmap=plt.get_cmap('gray'))
        # plt.axis('auto')
        # plt.xlim(0, n_exp-1)
        # plt.ylim(0, n_ref-1)
        # plt.show()

        # plt.imshow(de_RGB, cmap=plt.get_cmap('gray'))
        # plt.plot(new_array, color='red')
        # plt.axis('auto')
        # plt.xlim(0, n_exp-1)
        # plt.ylim(0, n_ref-1)
        # plt.show()

        # sum_avg[i,j] = weight
        pp = weight
        if pp < sum_weight:
            sum_weight = pp
            start_index = i
            end_index = j
        del indices, weight

# print(indices[0])
# print(indices[0,0], indices[0,1])
# input()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(de_RGB, cmap=plt.get_cmap('gray'))
# ax.scatter(range(n_exp),haha, color='Gray')
    # ax.scatter(indices[:,1], indices[:,0], color='red')
# plt.xlim(0, n_exp-1)
# plt.ylim(0, n_ref-1)
ax.set_aspect('auto')
plt.xticks([0, n_exp-1])
plt.yticks([0, n_ref-1])
plt.title('From sRGB model')
# plt.savefig(r'test1.png', dpi=300, format='png')

# ax = fig.add_subplot(122)
# ax.imshow(de_Lab, cmap=plt.get_cmap('gray'))
# plt.gca().invert_yaxis()
# ax.set_aspect('auto')
# plt.title('From Lab model')

plt.show()

################################################################################

# # start_val = np.argmin(de_Lab[:,0])
# # end_val = np.argmin(de_Lab[:,n_exp-1])
#
# plt.figure(1)
# plt.imshow(de_Lab, cmap=plt.get_cmap('gray'))
# plt.xlim(0, n_exp-1)
# plt.ylim(0, n_ref-1)
# # plt.scatter(0, start_val)
# # plt.scatter(n_exp-1, end_val)
# plt.scatter(0,pick)
# plt.axis('auto')
#
# plt.show()

################################################################################

pick = np.argmin(de_Lab[700:740,0]) + 700
# pick = np.argmin(de_RGB[730:765,0]) + 730
x_length = n_exp

yo_mama = np.zeros(x_length, dtype='int')
h_exp_microns = np.zeros(x_length)

points = [0, 175, 225, 250, x_length]

division1 = [points[0],points[1]]
division2 = [points[1]+1, points[2]]
division3 = [points[2]+1, points[3]]
division4 = [points[3]+1, points[4]]

for i in range(x_length):
    if division1[0] <= i <= division1[1]:
        down_variation = 20
        up_variation = 0
    elif division2[0] <= i <= division2[1]:
        down_variation = 10
        up_variation = 0
    elif division3[0] <= i <= division3[1]:
        down_variation = 4
        up_variation = 0
    elif division4[0] <= i <= division4[1]:
        down_variation = 18
        up_variation = 0
    else:
        down_variation = 0
        up_variation = 0
    pick = np.argmin(de_RGB[pick-down_variation:pick+up_variation+1,i]) + pick-down_variation
    yo_mama[i] = pick
    h_exp_microns[i] = h_ref_microns[pick]

plt.subplot(121)
plt.imshow(de_RGB, cmap=plt.get_cmap('gray'))
plt.xlim(0, n_exp-1)
plt.ylim(0, n_ref-1)
plt.axvline(x=points[0], color='green', linestyle='--')
plt.axvline(x=points[1], color='green', linestyle='--')
plt.axvline(x=points[2], color='green', linestyle='--')
plt.axvline(x=points[3], color='green', linestyle='--')
plt.axvline(x=points[4], color='green', linestyle='--')
plt.scatter(range(x_length), yo_mama, color='red')
plt.axis('auto')

plt.subplot(122)
plt.scatter(range(x_length)*px_exp_microns*(1/1000.0), h_exp_microns)
plt.ylim(0,4)

plt.show()

# fig, ax = plt.subplots()
# font_size = 18
# w = 10
# h = 7.5
# fig.set_size_inches(w,h)
# contourf_ = ax.pcolormesh(RR, HH, de_RGB, cmap=cm.gray)
# cbar = fig.colorbar(contourf_)
# cbar.ax.tick_params(labelsize=font_size)
# plt.xlabel(r'r $\left[ mm \right]$', fontsize=font_size)
# plt.ylabel(r'h $\left[ \mu m \right]$', fontsize=font_size)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.title(r'From sRGB model', fontsize=font_size)
# # plt.savefig(r'test.png', dpi=300, format='png')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(de_RGB, cmap=plt.get_cmap('gray'))
# # ax.scatter(range(n_exp),haha, color='Gray')
# plt.xlim(0, n_exp-1)
# plt.ylim(0, n_ref-1)
# ax.set_aspect('auto')
# plt.xticks([0, n_exp-1])
# plt.yticks([0, n_ref-1])
# plt.title('From sRGB model')
# # plt.savefig(r'test1.png', dpi=300, format='png')
