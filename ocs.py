# https://ieeexplore.ieee.org/document/5329404
# Modified version of above algorithm (no edge thinning b/c it looked worse)

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2hsv
from color_edge1_helpers import rgb2gray
#from classiccolorcanny import weighted_average, max_edge_channels, mean_edge_channels

plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


img = io.imread(r'C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\iguana.png')#, as_gray=True)
noisy_img = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\noisy_image.jpeg")
solid_color = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\solid_color.jpeg")
colored_blocks = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\colored_blocks.jpeg")

low_2 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\2.png")
low_107 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\107.png")


def rgb2ocs(img):
    output_img = np.zeros(img.shape)
    H, W, _ = img.shape
    for i in range(H):
        for j in range(W):
            r, g, b = img[i, j]
            output_img[i, j, 0] = np.uint8(0.5 * (255 + g - r))
            output_img[i, j, 1] = np.uint8(0.25 * (510 + r + g - 2 * b))
            output_img[i, j, 2] = np.uint8(1 / 3 * (r + g + b))
    return output_img.astype(np.uint8)

def weighted_average(img):
    shape = img.shape
    output = np.zeros((shape[0], shape[1]))
    for row in range(shape[0]):
        for col in range(shape[1]):
            red, green, blue = 0, 0, 0
            if img[row][col].shape == (4,):
                [red, green, blue, _] = img[row][col]
            else:
                [red, green, blue] = img[row, col]
            output[row, col] = (np.uint16(red) + np.uint16(green) + np.uint16(blue)) / 3
    return (output).astype(np.uint8)


def plot_channels(imgs, title):
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    plt.show()


def max_edge_channels(img, low_threshold, high_threshold):
    edge_detections = []
    for color in range(3): # 3 because RGB
        color_img = img[:, :, color]
        edge_detections.append(cv.Canny(color_img, low_threshold, high_threshold, L2gradient=True))
    edge_detections = np.array(edge_detections)
    #plot_channels(edge_detections, "max")
    output_edge = np.max(edge_detections, axis=0)
    return output_edge.astype(np.uint8)

def mean_edge_channels(img, low_threshold, high_threshold):
    edge_detections = []
    for color in range(3): # 3 because RGB
        color_img = img[:, :, color]
        edge_detections.append(cv.Canny(color_img, low_threshold, high_threshold, L2gradient=True))
    edge_detections = np.array(edge_detections)
    #plot_channels(edge_detections, "mean")
    output_edge = np.mean(edge_detections, axis=0)
    return output_edge.astype(np.uint8)



img_in_use = low_107

low_523 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\523.png")
img_in_use = low_523

ocs_version = rgb2ocs(img_in_use)

# plt.subplot(1, 3, 1)
# plt.imshow(ocs_version[:, :, 0])
# plt.axis('off')
# plt.title('image')

# plt.subplot(1, 3, 2)
# plt.imshow(ocs_version[:, :, 1])
# plt.axis('off')
# plt.title('Hue channel')

# plt.subplot(1, 3, 3)
# plt.imshow(ocs_version[:, :, 2])
# plt.axis('off')
# plt.title('Saturation channel')

# plt.show()



#low_threshold = 10
high_threshold = 10
low_threshold = high_threshold * 0.4

# Regular grayscale
grayscale_version = (rgb2gray(img_in_use)).astype(np.uint8)
#cv_canny = cv.Canny(grayscale_version, 10, 50, L2gradient=True)

# Combine Channels First
weighted_gray_version = weighted_average(ocs_version)
# plt.imshow(weighted_gray_version)
# plt.show()
cv_mean_canny = cv.Canny(weighted_gray_version, low_threshold, high_threshold, L2gradient=True)

# Channels treated independently
max_channel_version = max_edge_channels(ocs_version, low_threshold, high_threshold)
#max_cv_channel_canny = cv.Canny(max_channel_version, low_threshold, high_threshold, L2gradient=True)\

mean_channel_version = mean_edge_channels(ocs_version, low_threshold, high_threshold)
#mean_cv_channel_canny = cv.Canny(mean_channel_version, low_threshold, high_threshold, L2gradient=True)

# plt.subplot(1, 4, 1)
# plt.imshow(cv_canny)
# plt.axis('off')
# plt.title('cv')

plt.subplot(1, 2, 1)
plt.imshow(cv_mean_canny)
plt.axis('off')
plt.title('mean combined')

plt.subplot(1, 2, 2)
plt.imshow(max_channel_version)
plt.axis('off')
plt.title('max channel')

# plt.subplot(1, 2, 4)
# plt.imshow(mean_channel_version)
# plt.axis('off')
# plt.title('mean channel')

plt.show()