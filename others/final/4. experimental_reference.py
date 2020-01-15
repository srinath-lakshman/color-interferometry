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

f_background = hard_disk + folder + project + r'background_reference/info'
os.chdir(f_background)

[mask, mean_val, Rb_ch, Gb_ch, Bb_ch] = vertical_reference_background()
image_b = np.dstack((Rb_ch, Gb_ch, Bb_ch))

################################################################################

f = hard_disk + folder + project + r'sample_impact_over_dry_glass'
os.chdir(f)

image_file = r'sample_impact_over_dry_glass__C001H001S0001000048.tif'

info_file = f + '/info'
if os.path.exists(info_file):
    print('Info folder already exists!')
    [threshold, center, radius_px, Re_ch, Ge_ch, Be_ch] = image_info_readfile(info_file)
else:
    [threshold, center, radius_px, Re_ch, Ge_ch, Be_ch] = image_analysis1(image_file)

image_e = np.dstack((Re_ch, Ge_ch, Be_ch))

mod_image_color = image_color_corrected1(image_b, mask, mean_val, image_e)
mod_image_centered = image_centered_and_cropped(mod_image_color, center, radius_px)

# axisymmetric, len = radius_px, mask included
[R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric] = extract_color_channels1(mod_image_centered, 0-10, 0+10)

# # along a line with angle = theta, len = radius_px, mask included
# [R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric] = extract_color_channels2(mod_image_centered, 0)

if os.path.exists(info_file):
    os.chdir(info_file)
else:
    os.mkdir(info_file)
    os.chdir(info_file)

experimental_savefile(threshold, center, radius_px, Re_ch, Ge_ch, Be_ch, l, a, b)

################################################################################
