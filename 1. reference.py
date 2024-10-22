import os
from FUNC import *

################################################################################

hard_disk   = r'F:/'
project     = r'color_interferometry/bottom_view/20200520/'

f = hard_disk + project + r'calibration'
os.chdir(f)

################################################################################

reference_lengthscale_file = r'reference_lengthscale.tif'
reference_image_file = r'lens_focal_length_1000_mm_la1464.tif'

reference_image = image_readfile(reference_image_file)

px_microns = 1.131
# px_microns = length_analysis(reference_lengthscale_file)

# center = [1022, 1026]
# radius_px = 1021
center, radius_px = find_center(reference_image)

# r_mm = 0
# h_microns = 0
# rgb_colors = 0
# ref_colors = 0
# image_axi = 0
# lens_config = 'f0300'
r_mm, h_microns, rgb_colors, ref_colors, image_axi, lens_config = analysis_reference(reference_image_file, reference_image, center, radius_px, px_microns)

savefile_reference(lens_config, radius_px, r_mm, h_microns, ref_colors, rgb_colors, px_microns)

################################################################################
