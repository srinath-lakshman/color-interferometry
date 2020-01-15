import os
from FUNC import *

################################################################################

hard_disk   = r'/media/devici/328C773C8C76F9A5/'
project     = r'color_interferometry/bottom_view/20200114/'

f = hard_disk + project + r'reference'
os.chdir(f)

################################################################################

reference_lengthscale_file = r'reference_lengthscale.tif'
reference_image_file = r'lens_focal_length_300_mm_la1484.tif'

reference_image = image_readfile(reference_image_file)

px_microns = 1.385
# px_microns = length_analysis(reference_lengthscale_file)

center = [1022, 1024]
radius_px = 1022
# center, radius_px = find_center(reference_image)

r_mm, h_microns, rgb_colors, ref_colors, image_axi = analysis_reference(reference_image, center, radius_px, px_microns)

savefile_reference(radius_px, r_mm, h_microns, ref_colors, rgb_colors, px_microns)

################################################################################
