import os
from FUNC import *

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
experiment_image_file = r'lower_speed_mica_run1_000108.tif'
experiment_image = image_readfile(experiment_image_file)

# center = [377, 402]           #from the code
# radius_px = 365               #from the code
center, radius_px = find_center(experiment_image)

threshold = experiment_threshold(experiment_image)

################################# GLOBAL LIMIT #################################
# lower speed
radius_px = 300
#
# # higher speed
# radius_px = 500
################################################################################

# threshold = 750
# left-top corner = [330, 143]
# right-bottom corner = [492, 308]
# left extents = [0, 5]
# right extents = [157, 161]
# top extents = [2, 10]
# bottom extents = [160, 164]
_, rgb_colors, ref_colors, image_axi = analysis_experiment(experiment_image, 270-22.5, 270+22.5, center, radius_px, px_microns)
radius_px_mod = analysis_drop_extents(image_axi, radius_px, rgb_colors, ref_colors)
r_mm, rgb_colors_mod, ref_colors_mod, image_axi_mod = analysis_experiment(experiment_image, 270-22.5, 270+22.5, center, radius_px_mod, px_microns)

savefile_experimental(experiment_image_file, radius_px_mod, r_mm, ref_colors_mod, rgb_colors_mod, px_microns, image_axi)

################################################################################
