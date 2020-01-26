import os
from matplotlib import pyplot as plt
from FUNC_experiment import experiment_circlefit
from FUNC_experiment import experiment_analysis
from FUNC_experiment import experiment_savefile
from FUNC_experiment import color_8bit
from skimage import io
import numpy as np

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5'
project     = r'color_interferometry/bottom_view/20200114'

f = hard_disk + '/' + project + '/' + r'experiment'
os.chdir(f)

################################################################################

# # Length scale information
# image_filename  = r'mag_2x_dot_diameter_1_mm000001.tif'
# approximate_center          = [344, 441]
# approximate_crop            = 250
# approximate_threshold       = -9000
# approximate_radii           = [183,193]
#
# center, radius, radius_max = experiment_circlefit(\
#                                 image_filename  = image_filename, \
#                                 center          = approximate_center, \
#                                 crop            = approximate_crop, \
#                                 threshold       = approximate_threshold, \
#                                 radii           = approximate_radii)
#
# lengthscale_diameter_mm = 1
# lengthscale_diameter_px = 2 * radius
# px_microns = round((lengthscale_diameter_mm*(10**3))/lengthscale_diameter_px,3)
#
# txt_file = open("circlefit_parameters.txt","w")
# txt_file.write("INPUT:\n")
# txt_file.write(f"image_filename = {image_filename}\n")
# txt_file.write(f"approximate_center = {approximate_center}\n")
# txt_file.write(f"approximate_crop = {approximate_crop}\n")
# txt_file.write(f"approximate_threshold = {approximate_threshold}\n")
# txt_file.write(f"approximate_radii = {approximate_radii}\n")
# txt_file.write("\n")
# txt_file.write("OUTPUT:\n")
# txt_file.write(f"center = {center}\n")
# txt_file.write(f"radius = {radius}\n")
# txt_file.write(f"radius_max = {radius_max}\n")
# txt_file.close()
#
# txt_file = open("px_microns.txt","w")
# txt_file.write(f"1 pixel = {px_microns} microns")
# txt_file.close()

px_microns = 2.688

################################################################################

run = r'higher_speed_mica_run1'
os.chdir(f + '/' + run)

################################################################################

# Center information
image_filename              = r'higher_speed_mica_run1_000144.tif'
approximate_center          = [409, 227]
approximate_crop            = 110
approximate_threshold       = +75
approximate_radii           = [70,90]

center, radius, radius_max = experiment_circlefit(\
                                image_filename  = image_filename, \
                                center          = approximate_center, \
                                crop            = approximate_crop, \
                                threshold       = approximate_threshold, \
                                radii           = approximate_radii)

image = io.imread(image_filename)

channel_R = image[:,:,0]
channel_G = image[:,:,1]
channel_B = image[:,:,2]

for f_subfolder in ['info', 'center', image_filename.split('.')[0]]:
    if os.path.exists(f_subfolder):
        print(f"{f_subfolder} folder already exist!")
    else:
        print(f"{f_subfolder} folder does not exist!")
        os.mkdir(f_subfolder)
        print(f"{f_subfolder} folder created!")
    os.chdir(f_subfolder)

txt_file = open("circlefit_parameters.txt","w")
txt_file.write("INPUT:\n")
txt_file.write(f"image_filename = {image_filename}\n")
txt_file.write(f"approximate_center = {approximate_center}\n")
txt_file.write(f"approximate_crop = {approximate_crop}\n")
txt_file.write(f"approximate_threshold = {approximate_threshold}\n")
txt_file.write(f"approximate_radii = {approximate_radii}\n")
txt_file.write("\n")
txt_file.write("OUTPUT:\n")
txt_file.write(f"center = {center}\n")
txt_file.write(f"radius = {radius}\n")
txt_file.write(f"radius_max = {radius_max}\n")
txt_file.close()

txt_file = open("px_microns.txt","w")
txt_file.write(f"1 pixel = {px_microns} microns")
txt_file.close()

txt_file = open("center.txt","w")
txt_file.write(f"center = {center}")
txt_file.close()

txt_file = open("radius_max.txt","w")
txt_file.write(f"radius_max = {radius_max}")
txt_file.close()

np.savetxt("channel_R.txt", channel_R, fmt='%d')
np.savetxt("channel_G.txt", channel_G, fmt='%d')
np.savetxt("channel_B.txt", channel_B, fmt='%d')

radius_max = 500

r_mm, rgb_colors, ref_colors, image_axi = experiment_analysis(270-5, 270+5, center, radius_max, px_microns)

experiment_savefile(image_filename, r_mm, ref_colors, rgb_colors, px_microns, image_axi)

################################################################################
