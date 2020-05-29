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

f_lateral = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/lateral_reference'
os.chdir(f_lateral)

px = np.loadtxt('pixel_length.txt')*1000                                        #pixel length in millimeters

f = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191004/vertical_reference1'
os.chdir(f)

rgb = mpimg.imread('vertical_reference1__C001H001S0001000001.tif')

x_px = np.shape(rgb)[0]
y_px = np.shape(rgb)[1]
ch = np.shape(rgb)[2]

# print(rgb[:,:,1].min(), rgb[:,:,1].max())
# input()

center = np.loadtxt('center.txt')

xc = int(center[0])
yc = int(center[1])
radius = min(xc, yc, x_px-1-xc, y_px-1-yc)plt.figure(1)
plt.plot(range(N_exp), h_air)

lab = sRGBtoLab(rgb/[1,1.6,1], 8, 'D50', 'D50')

lab_mod = lab[yc-radius:yc+radius+1, xc-radius:xc+radius+1,:]
sq_size = round((np.shape(lab_mod)[0]+np.shape(lab_mod)[1])/2)
delta = round((sq_size-1)/2)

ref_l = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,0])
ref_a = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,1])
ref_b = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,2])

plt.plot(ref_a,ref_b)
plt.show()

print('yo mama')
input()

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
# threshold = 100NL

# plt.subplot(122)
# plt.imshow(rgb2gray(rgb)>threshold, cmap=plt.get_cmap('gray'))
# plt.xlabel('X [px]')
# plt.xticks([0,x_px-1],[1,x_px])
# plt.ylabel('Y [px]')
# plt.yticks([0,y_px-1],[y_px,1])
#
# plt.show()

# fig = plt.figure(1)
# fig.set_size_inches(25,15)
#
# plt.subplot(121)
# plt.imshow(rgb_mod, extent=[(-radius)*px, (+radius)*px, (-radius)*px, (+radius)*px])
# plt.xlabel('X [mm]')
# plt.ylabel('Y [mm]')
# plt.title('Reference Image')

ff = 1000                                                                       #focal length in millimeters
RR = 515.1
rr = np.arange(0,radius*px,px)
hh = (RR - np.sqrt((RR*RR)-(rr*rr)))*1000

start_angle = 0
end_angle = 360
delta_angle = 0.5

ref_red   = average_profile(start_angle, end_angle, delta_angle, delta, delta, delta, rgb_mod[:,:,0])
ref_green = average_profile(start_angle, end_angle, delta_angle, delta, delta, delta, rgb_mod[:,:,1])
ref_blue  = average_profile(start_angle, end_angle, delta_angle, delta, delta, delta, rgb_mod[:,:,2])

# ax1 = plt.subplot(222)
# ax1.plot(np.arange(0,radius*px,px), ref_red, color='red', label='Red')
# ax1.plot(np.arange(0,radius*px,px), ref_green, color='green', label='Green')
# ax1.plot(np.arange(0,radius*px,px), ref_blue, color='blue', label='Blue')
# ax1.set_xlim(0,radius*px)
# ax1.set_xlabel('r [mm]')
# ax1.set_ylabel('Color Intensity')
# ax1.set_title(r'Calibration plot - RGB colorspace')
# ax1.legend(loc='center left', bbox_to_anchor=(1.1,0.5))
#
# ax2 = ax1.twinx()
# ax2.plot(rr, hh, color='black')
# ax2.set_ylabel(r'h [$\mu m$]')

################################################################################
#RGB to XYZ conversion matrix

RGB_to_XYZ = np.array([[0.4360747, 0.3850649, 0.1430804],[0.2225045, 0.7168786, 0.0606169],[0.0139322, 0.0971045, 0.7141733]])

################################################################################

################################################################################
#reference white

R_white = 1093
G_white = 1024
B_white = 1306

n = 12

ref_red = ref_red/2**n
ref_green = ref_green/2**n
ref_blue = ref_blue/2**n

R_white = R_white/2**n
G_white = G_white/2**n
B_white = B_white/2**n

RGB_white = np.transpose(np.array([[R_white, G_white, B_white]]))
XYZ_white = np.dot(RGB_to_XYZ, RGB_white)

X_white = XYZ_white[0]
Y_white = XYZ_white[1]
Z_white = XYZ_white[2]

################################################################################

ref_X = np.zeros(delta)
ref_Y = np.zeros(delta)
ref_Z = np.zeros(delta)

ref_ll = np.zeros(delta)
ref_aa = np.zeros(delta)
ref_bb = np.zeros(delta)

for i in range(delta):
    ref_X[i] = RGB_to_XYZ[0,0]*ref_red[i] + RGB_to_XYZ[0,1]*ref_green[i] + RGB_to_XYZ[0,2]*ref_blue[i]
    ref_Y[i] = RGB_to_XYZ[1,0]*ref_red[i] + RGB_to_XYZ[1,1]*ref_green[i] + RGB_to_XYZ[1,2]*ref_blue[i]
    ref_Z[i] = RGB_to_XYZ[2,0]*ref_red[i] + RGB_to_XYZ[2,1]*ref_green[i] + RGB_to_XYZ[2,2]*ref_blue[i]

for i in range(delta):
    if ref_Y[i]/Y_white > 0.008856:
        ref_ll[i] = (116*((ref_Y[i]/Y_white)**(1/3))) - 16
    else:
        ref_ll[i] = 903.3*(ref_Y[i]/Y_white)
    ref_aa[i] = 500*(func_return(ref_X[i]/X_white) - func_return(ref_Y[i]/Y_white))
    ref_bb[i] = 200*(func_return(ref_Y[i]/Y_white) - func_return(ref_Z[i]/Z_white))

# plt.figure(2)

# print(min(ref_X[i]/X_white), max(ref_X[i]/X_white))
# print(min(ref_Y[i]/Y_white), max(ref_Y[i]/Y_white))
# print(min(ref_Z[i]/Z_white), max(ref_Z[i]/Z_white))
# # input()

lab_mod = color.rgb2lab(rgb_mod, illuminant='D65')

ref_l = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,0])
ref_a = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,1])
ref_b = average_profile(0, 360, 0.5, delta, delta, delta, lab_mod[:,:,2])

plt.plot(ref_aa, ref_bb)
plt.plot(ref_a, ref_b)
plt.show()

#
# ax3 = plt.subplot(224)
# ax3.plot(np.arange(0,radius*px,px), ref_l, color='red', label='L')
# ax3.plot(np.arange(0,radius*px,px), ref_a, color='green', label='a')
# ax3.plot(np.arange(0,radius*px,px), ref_b, color='blue', label='b')
# ax3.set_xlim(0,radius*px)
# ax3.set_xlabel('r [mm]')
# ax3.set_ylabel('Color Intensity')
# ax3.set_title(r'Calibration plot - Lab colorspace')
# ax3.legend(loc='center left', bbox_to_anchor=(1.1,0.5))
#
# ax4 = ax3.twinx()
# ax4.plot(rr, hh, color='black', label='spherical lens')
# ax4.set_ylabel(r'h [$\mu m$]')
#
# # fig1 = plt.figure(2)
# # ax5 = fig1.gca(projection='3d')
# # ax5.scatter(ref_a,ref_b,hh)
#
# fig.savefig('vertical_reference1_plot.tif')
#
# np.savetxt('ref_l.txt',ref_l)
# np.savetxt('ref_a.txt',ref_a)
# np.savetxt('ref_b.txt',ref_b)
# np.savetxt('rr.txt',rr)
# np.savetxt('hh.txt',hh)
#
# plt.show()
