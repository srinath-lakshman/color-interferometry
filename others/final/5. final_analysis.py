import os
from FUNC_ import *

################################################################################

hard_disk = r'/media/devici/328C773C8C76F9A5/'
folder = r'color_interferometry/bottom_view/'
project = r'20191023/exp1/'

################################################################################

f_lateral = hard_disk + folder + project + r'lateral_reference/info'
os.chdir(f_lateral)

px_microns = vertical_reference_lateral()

################################################################################

f_vertical_reference = hard_disk + folder + project + r'vertical_reference2/info'
os.chdir(f_vertical_reference)

[r_ref, h_ref, px_ref, R_ref, G_ref, B_ref] = analysis_vertical_readfile()

f_experimental_reference = hard_disk + folder + project + r'sample_impact_over_dry_glass/info'
os.chdir(f_experimental_reference)

[px_exp, R_exp, G_exp, B_exp] = analysis_experimental_readfile()

px_ref = int(px_ref)
px_exp = int(px_exp)

os.chdir('..')
os.chdir('..')

################################################################################

de = np.zeros((px_ref, px_exp))

for i in range(px_ref):
    for j in range(px_exp):
        # de[i,j] = np.sqrt((R_ref[i]-R_exp[j])**2 + (G_ref[i]-G_exp[j])**2)
        de[i,j] = np.sqrt((G_ref[i]-G_exp[j])**2 + (B_ref[i]-B_exp[j])**2)
        # de[i,j] = np.sqrt((B_ref[i]-B_exp[j])**2 + (R_ref[i]-R_exp[j])**2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(de, cmap=plt.get_cmap('gray'))
plt.gca().invert_yaxis()
ax.set_aspect('auto')
plt.show()
#
#
# #matrix = [[2,3,4],[5,9,8],[7,2,1]]
# matrix = de
# start = [125,0]
# end = [125,161]
# cost=minimumCostPath(matrix,start,end)
#
# for neigh in range(20):
#     count = count + 1
#     [de, start_pixel_value, track]= path_minimum_algorithm(h_ref, px_ref, R_ref, G_ref, B_ref, px_exp, R_exp, G_exp, B_exp, count)
#     plt.figure(0)
#     plt.imshow(de, cmap=plt.get_cmap('gray'))
#     plt.gca().invert_yaxis()
#     plt.plot(track[start_pixel_value,:])
#     # plt.show(block=False)
#     # raw_input()
#     del start_pixel_value
#     del track
#
# plt.show()

# neigh = px_ref
neigh = 20
[de, start_pixel_value, track]= path_minimum_algorithm(h_ref, px_ref, R_ref, G_ref, B_ref, px_exp, R_exp, G_exp, B_exp, neigh)

mat_min = de.min()
# print(mat_min)
mat_index = np.where(de == mat_min)
# print(mat_index[1])
# input()

plt.subplot(121)
plt.imshow(de, cmap=plt.get_cmap('gray'))
plt.gca().invert_yaxis()
# plt.scatter(min_val, range(px_ref))
plt.scatter(0, start_pixel_value)
plt.plot(track[start_pixel_value,:])
plt.scatter(mat_index[1], mat_index[0])

ll = track[start_pixel_value,:].astype('int')
h_exp = h_ref[ll]

plt.subplot(122)
plt.plot(h_exp)

plt.show()
