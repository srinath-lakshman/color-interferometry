import os
# from FUNC import *
from FUNC_experiment import experiment_readimage
from FUNC_experiment import experiment_threshold, experiment_crop, experiment_circlefit
from FUNC_experiment import experiment_analysis, experiment_dropextents
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
experiment_image_file = r'lower_speed_mica_run1_000092.tif'
experiment_image = experiment_readimage(experiment_image_file)

# center = [377, 402]           #from the code
# radius_px = 365               #from the code
# center, radius_px = find_center(experiment_image)

gray, binary = experiment_threshold(experiment_image,threshold=125)
binary_cropped, crop = experiment_crop(binary, crop=[[250,250],[500,500]])
center, radius_px = experiment_circlefit(gray, binary_cropped, crop, diameter_extents=[[20,30],[225,235],[40,50],[245,255]])
# center = experiment_circlefit(gray, binary_cropped, crop)

r_mm, rgb_colors, ref_colors, image_axi = experiment_analysis(experiment_image, 270-22.5, 270+22.5, center, radius_px, px_microns)
radius_px_mod = experiment_dropextents(image_axi, radius_px, rgb_colors, ref_colors)
r_mm_mod, rgb_colors_mod, ref_colors_mod, image_axi_mod = experiment_analysis(experiment_image, 270-22.5, 270+22.5, center, radius_px_mod, px_microns)

experiment_savefile(experiment_image_file, radius_px_mod, r_mm_mod, ref_colors_mod, rgb_colors_mod, px_microns, image_axi)

################################################################################
