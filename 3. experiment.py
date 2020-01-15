import os
from FUNC import *

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200112/'

f = hard_disk + project + r'experiment/lower_impact_speed_run1'
os.chdir(f)

################################################################################

# experiment_lengthscale_file = r'experiment_lengthscale000001.tif'
experiment_image_file = r'lower_impact_run1_000131.tif'

experiment_image = image_readfile(experiment_image_file)

px_microns = 2.543
# px_microns = length_analysis(experiment_lengthscale_file)

center = [392, 405]
radius_px = 362
# center, radius_px = find_center(experiment_image)

r_mm, rgb_colors, ref_colors, image_axi = analysis_experiment(experiment_image, 270-22.5, 270+22.5, center, radius_px, px_microns)

savefile_experimental(experiment_image_file, radius_px, r_mm, ref_colors, rgb_colors, px_microns)

################################################################################