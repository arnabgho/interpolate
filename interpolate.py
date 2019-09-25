import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

img_orig = plt.imread('epoch003_sparse_real_A.png')
img_ds_align = plt.imread('small.png')
img_ds_no_align = plt.imread('small_no_align.png')
img_ds_nearest = plt.imread('small_nearest.png')

img_rs = imresize(img_orig, .25, interp='bilinear')

plt.figure()
plt.subplot(1,5,1)
plt.imshow(img_orig)
plt.title('Original')
plt.subplot(1,5,2)
plt.imshow(img_ds_no_align)
plt.title('No Align')
plt.subplot(1,5,3)
plt.imshow(img_ds_align)
plt.title('Align')
plt.subplot(1,5,4)
plt.imshow(img_ds_nearest)
plt.title('Nearest')
plt.subplot(1,5,5)
plt.imshow(img_rs)
plt.title('Resized')
plt.show()

