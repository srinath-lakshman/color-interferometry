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

f = hard_disk + folder + project + r'vertical_reference2'
os.chdir(f)

image_file = image_readfile()

info_file = f + '/info'
if os.path.exists(info_file):
    print('Info folder already exists!')
    [threshold, center, radius_px, Rv_ch, Gv_ch, Bv_ch] = image_info_readfile(info_file)
else:
    [threshold, center, radius_px, Rv_ch, Gv_ch, Bv_ch] = image_analysis1(image_file)

image_v = np.dstack((Rv_ch, Gv_ch, Bv_ch))

mod_image_color = image_color_corrected1(image_b, mask, mean_val, image_v)
mod_image_centered = image_centered_and_cropped(mod_image_color, center, radius_px)

# axisymmetric, len = radius_px, mask included
# [R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric] = extract_color_channels1(mod_image_centered, 0, 360)
# [r_microns, r_mm, h_microns] = planoconvex_readfile(radius_px, px_microns)

# # along a line with angle = theta, len = radius_px, mask included
# [R_avg, G_avg, B_avg, l, a, b, side_len, image_axisymmetric] = extract_color_channels2(mod_image_centered, 0)
# [r_microns, r_mm, h_microns] = planoconvex_readfile(radius_px, px_microns)

# aa = np.array([2,4,8,6,1,0])
# bb = sort(aa)
# print(aa)
# print(bb)
# input()

# # including all image coordinates, len is orders of magnitude larger than radius_px, mask included
[_, _, _, R_lens] = planoconvex_readfile(radius_px, px_microns)
[R_avg, G_avg, B_avg, l, a, b, radius_px, r_mm, h_microns] = extract_color_channels3(mod_image_centered, px_microns, R_lens)

if os.path.exists(info_file):
    os.chdir(info_file)
else:
    os.mkdir(info_file)
    os.chdir(info_file)

vertical_reference_savefile(threshold, center, radius_px, Rv_ch, Gv_ch, Bv_ch, r_mm, h_microns, l, a, b)

################################################################################
