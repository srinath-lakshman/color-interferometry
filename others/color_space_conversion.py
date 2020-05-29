import os
import matplotlib.image as mpimg
from FUNC_ import sRGBtoLab

f = r'/media/devici/Samsung_T5/color_interferometry/bottom_view/20191009/vertical_reference3'
os.chdir(f)

RGB_1 = mpimg.imread('vertical_reference3__C001H001S0001000001.tif')

hi = sRGBtoLab(RGB_1, 8, 'D65', 'D65')

print(RGB_1)
print(hi)
print('hi')
