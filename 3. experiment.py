import os
from matplotlib import pyplot as plt
from FUNC_experiment import experiment_readimage, color_8bit
from FUNC_experiment import experiment_circlefit
from FUNC_experiment import experiment_savecircle, experiment_analysis
from FUNC_experiment import experiment_savefile

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200114/'

fl = hard_disk + project + r'experiment'
os.chdir(fl)

################################################################################

experiment_lengthscale_file = r'mag_2x_dot_diameter_1_mm000001.tif'

px_microns = 2.717
# px_microns = length_analysis(experiment_lengthscale_file)

fe = fl + r'/lower_speed_mica_run1'
os.chdir(fe)
experiment_image_file = r'lower_speed_mica_run1_000110.tif'
experiment_image = experiment_readimage(experiment_image_file)

center_px, radius_px = experiment_circlefit(experiment_image, center=[380,383], crop=75, threshold=100, radii=[70,80])

char = input("Correct (y/n)? ")
if char == 'y':
    experiment_savecircle(experiment_image_file, center_px, radius_px, px_microns)

r_mm, rgb_colors, ref_colors, image_axi = experiment_analysis(experiment_image, 270-5, 270+5, center_px, radius_px, px_microns)

experiment_savefile(experiment_image_file, radius_px, r_mm, ref_colors, rgb_colors, px_microns, image_axi)

################################################################################
