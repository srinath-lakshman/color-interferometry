import os
from FUNC_ import *

################################################################################

hard_disk = r'/media/devici/328C773C8C76F9A5/'
folder = r'color_interferometry/bottom_view/'
project = r'20191023/exp1/'

f = hard_disk + folder + project + r'background_reference'
os.chdir(f)

################################################################################

info_file = f + '/info'
if os.path.exists(info_file):
    print('Info folder already exists!')
else:
    image_file = image_readfile()
    [threshold, mask, mean_intensities, R_ch, G_ch, B_ch] = background_reference_analysis(image_file)
    background_reference_savefile(threshold, mask, mean_intensities, R_ch, G_ch, B_ch, info_file)

################################################################################
