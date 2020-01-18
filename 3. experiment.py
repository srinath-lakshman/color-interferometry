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

fe = fl + r'/higher_speed_mica_run1'
os.chdir(fe)
experiment_image_file = r'higher_speed_mica_run1_000155.tif'
experiment_image = image_readfile(experiment_image_file)

center = [414, 220]           #from the code
# radius_px = 220               #from the code
radius_px = 410
# center, radius_px = find_center(experiment_image)

r_mm, rgb_colors, ref_colors, image_axi = analysis_experiment(experiment_image, 270-22.5, 270+22.5, center, radius_px, px_microns)

# plt.imshow(color_8bit(image_axi))
# plt.show()

savefile_experimental(experiment_image_file, radius_px, r_mm, ref_colors, rgb_colors, px_microns)

################################################################################
