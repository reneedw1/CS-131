# https://ieeexplore.ieee.org/document/5329404
# Modified version of above algorithm (no edge thinning b/c it looked worse)

import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io
from color_edge1_helpers import *

plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

img = io.imread(r'C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\iguana.png')#, as_gray=True)
noisy_img = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\noisy_image.jpeg")
solid_color = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\solid_color.jpeg")
colored_blocks = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\colored_blocks.jpeg")

low_2 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\2.png")

#filtered = adaptive_median_kernel(img, 3, 9)
#filtered = adaptive_median_kernel(noisy_img, 3, 9)
#filtered = adaptive_median_kernel(solid_color, 3, 9)
filtered = adaptive_median_kernel(low_2, 3, 9)
#filtered = adaptive_median_kernel(colored_blocks, 3, 9)
weighted_gray = weighted_average_image(filtered)
directional_difference = max_directional_difference(weighted_gray)
thresholded = threshold(directional_difference)

#edge_thinned = edge_thinning(thresholded)

canny_output = canny(rgb2gray(low_2), kernel_size=5, sigma=1.4, high=1, low=0.5)
canny_output2 = cv.Canny((thresholded).astype(np.uint8), 10, 50, L2gradient=True)


plt.subplot(1, 4, 1)
plt.imshow(filtered)
plt.axis('off')
plt.title('image')


plt.subplot(1, 4, 2)
plt.imshow(thresholded)
plt.axis('off')
plt.title('Filtered')

plt.subplot(1, 4, 3)
plt.imshow(canny_output)
plt.axis('off')
plt.title('output homemade')

plt.subplot(1, 4, 4)
plt.imshow(canny_output2)
plt.axis('off')
plt.title('output2')



plt.show()
#plot_images(img, filtered, 4)
#plot_images(noisy_img, filtered, 3)
#plot_images(solid_color, filtered, 3)