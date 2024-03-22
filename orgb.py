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
low_107 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\477.png")
high_107 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\high\107.png")


def rgb2orgb(img):
    output_img = np.zeros(img.shape)
    H, W, _ = img.shape
    for i in range(H):
        for j in range(W):
            #b, g, r = img[i, j]
            r, g ,b = img[i, j]
            b = (b / 255) ** (1 / 2.2)
            g = (g / 255) ** (1 / 2.2)
            r = (r / 255) ** (1 / 2.2)
            #L' 
            output_img[i, j, 0] = 0.2990 * r + 0.5870 * g + 0.1140 * b
            #C1'
            #output_img[i, j, 1] = 0.5000 * r + 0.5000 * g - 1.0000 * b
            C1 = 0.5000 * r + 0.5000 * g - 1.0000 * b
            #C2'
            #output_img[i, j, 2] = 0.8660 * r - 0.8660 * g + 0.0000 * b
            C2 = 0.8660 * r - 0.8660 * g + 0.0000 * b

            theta = np.arctan2(C2, C1)
            theta_new = 0
            
            if theta < np.pi / 3:
                theta_new = 3/2 * theta
            elif theta >= np.pi / 3 and theta <= np.pi:
                theta_new = np.pi / 2 + 3 / 4 * (theta - np.pi / 3)
            else:
                temp = 0

            theta_diff = theta - theta_new
            R = np.array([[np.cos(theta_diff), -np.sin(theta_diff)], 
                         [np.sin(theta_diff), np.cos(theta_diff)]])
            
            Cs = np.array([C1, C2])

            [Cyb, Crg] = np.matmul(R, np.transpose(Cs))

            output_img[i, j, 1] = Cyb
            output_img[i, j, 2] = Crg

    temp1 = output_img[:, :, 0]
    temp1 = output_img[:, :, 1]
    temp1 = output_img[:, :, 2]
    output_img[:, :, 0] = (output_img[:, :, 0] - np.min(output_img[:, :, 0]))/(np.max(output_img[:, :, 0]) - np.min(output_img[:, :, 0])) * 255
    output_img[:, :, 1] = (output_img[:, :, 1] - np.min(output_img[:, :, 1]))/(np.max(output_img[:, :, 1]) - np.min(output_img[:, :, 1])) * 255
    output_img[:, :, 2] = (output_img[:, :, 2] - np.min(output_img[:, :, 2]))/(np.max(output_img[:, :, 2]) - np.min(output_img[:, :, 2])) * 255

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
            output[row, col] = (red + green + blue) / 3
            #output[row, col] = (red + 0.5 * green + 0.5 * blue) / 2
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
    plot_channels(edge_detections, "max")
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


#img_in_use = high_107
#img_in_use = low_107


# plt.imshow(ocs_version)
# plt.show()

low_523 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\523.png")
img_in_use = low_523

ocs_version = rgb2orgb(img_in_use)

print(np.min(ocs_version[:, :, 0]), np.mean(ocs_version[:, :, 0]), np.max(ocs_version[:, :, 0]))
print(np.min(ocs_version[:, :, 1]), np.mean(ocs_version[:, :, 1]), np.max(ocs_version[:, :, 1]))
print(np.min(ocs_version[:, :, 2]), np.mean(ocs_version[:, :, 2]), np.max(ocs_version[:, :, 2]))


plt.subplot(1, 3, 1)
plt.imshow(ocs_version[:, :, 0])
plt.axis('off')
plt.title('dim1')

plt.subplot(1, 3, 2)
plt.imshow(ocs_version[:, :, 1])
plt.axis('off')
plt.title('dim2')

plt.subplot(1, 3, 3)
plt.imshow(ocs_version[:, :, 2])
plt.axis('off')
plt.title('dim3')

plt.show()


high_threshold = 200
low_threshold = high_threshold * 0.4

# Regular grayscale
grayscale_version = (rgb2gray(img_in_use)).astype(np.uint8)
cv_canny = cv.Canny(grayscale_version, 12, 30, L2gradient=True)

# Combine Channels First
weighted_gray_version = weighted_average(ocs_version)
cv_mean_canny = cv.Canny(weighted_gray_version, low_threshold, high_threshold, L2gradient=True)

# Channels treated independently
max_channel_version = max_edge_channels(ocs_version, low_threshold, high_threshold)
#max_cv_channel_canny = cv.Canny(max_channel_version, 100, 200, L2gradient=True)\

mean_channel_version = mean_edge_channels(ocs_version, low_threshold, high_threshold)
#mean_cv_channel_canny = cv.Canny(mean_channel_version, low_threshold * 4, high_threshold * 4, L2gradient=True)

# plt.subplot(1, 3, 1)
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

# plt.subplot(1, 4, 4)
# plt.imshow(mean_channel_version)
# plt.axis('off')
# plt.title('mean channel')

plt.show()