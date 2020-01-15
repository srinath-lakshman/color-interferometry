import os
from FUNC_ import *

################################################################################

hard_disk = r'/media/devici/328C773C8C76F9A5/'
folder = r'color_interferometry/bottom_view/'
project = r'20191023/exp1/'

f = hard_disk + folder + project + r'lateral_reference'
os.chdir(f)

################################################################################

info_file = f + '/info'
if os.path.exists(info_file):
    print('Info folder already exists!')
else:
    len = lateral_reference_readfile()
    image_file = image_readfile()
    [avg_px, threshold, extents] = lateral_reference_analysis(image_file)
    lateral_reference_savefile(len, avg_px, threshold, extents, info_file)

################################################################################
