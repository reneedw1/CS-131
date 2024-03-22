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
low_107 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\487.png")

def plot_channels(imgs, title):
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    plt.show()

#w1 = 0.3, w2 = 0.6, w3 = 0.1
# def hsv_range(hue_img, saturation_img, value_img):
#     w1 = 0
#     while w1 <= 1:
#         display = []
#         w2 = 0
#         while w2 <= 1 - w1:
#             w3 = 1 - w1 - w2 
#             output_image = w1 * value_img + w2 * value_img * saturation_img + w3 * value_img * saturation_img * hue_img
#             output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)
#             cv_canny = cv.Canny(output_image, 0.3 * 0.4 * 255, 0.3 * 255, L2gradient=True)
#             display.append(cv_canny)
#             w2 += 0.1
#         display_len = len(display)
#         for i, img in enumerate(display):
#             plt.subplot(1, display_len, i + 1)
#             plt.imshow(img)
#             plt.axis('off')
#             plt.title(f'{round(w1, 2)}, {round(i * 0.1, 2)}')
#         plt.show()
#         w1 += 0.1

def hsv(hue_img, saturation_img, value_img, low_threshold, high_threshold):
    w1 = 0.7
    w2 = 0.3
    w3 = 0.0
    output_image = w1 * value_img + w2 * value_img * saturation_img + w3 * value_img * saturation_img * hue_img
    output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)
    cv_canny = cv.Canny(output_image, low_threshold, high_threshold, L2gradient=True)
    #cv_canny = cv.Canny(output_image, 75, 100, L2gradient=True)
    return cv_canny      


# def hsv_flat(hue_img, saturation_img, value_img):
#     w1 = 0.3
#     w2 = 0.6
#     w3 = 0.1
#     output_image = w1 * value_img + w2 * saturation_img + w3 * hue_img
#     output_image = (output_image / np.max(output_image) * 255).astype(np.uint8)
#     cv_canny = cv.Canny(output_image, 0.3 * 0.4 * 255, 0.3 * 255, L2gradient=True)
#     return cv_canny   


def hsv_max_channels(img, low_threshold, high_threshold):
    hue_img = img[:, :, 0]
    saturation_img = img[:, :, 1]
    value_img = img[:, :, 2]

    w1 = 0.3
    w2 = 0.6
    w3 = 0.1

    channel1 = w1 * value_img
    channel2 = w1 * value_img + w2 * value_img * saturation_img
    channel3 = w1 * value_img + w2 * value_img * saturation_img + w3 * value_img * saturation_img * hue_img

    channel1 = (channel1 / np.max(channel1) * 255).astype(np.uint8)
    channel2 = (channel2 / np.max(channel2) * 255).astype(np.uint8)
    channel3 = (channel3 / np.max(channel3) * 255).astype(np.uint8)

    channel1_canny = cv.Canny(channel1, low_threshold, high_threshold, L2gradient=True)
    channel2_canny = cv.Canny(channel2, low_threshold, high_threshold, L2gradient=True)
    channel3_canny = cv.Canny(channel3, low_threshold, high_threshold, L2gradient=True)

    # channel1_canny = cv.Canny(channel1, low_threshold, high_threshold, L2gradient=True)
    # channel2_canny = cv.Canny(channel2, low_threshold, high_threshold, L2gradient=True)
    # channel3_canny = cv.Canny(channel3, low_threshold, high_threshold, L2gradient=True)

    edge_detections = np.array([channel1_canny, channel2_canny, channel3_canny])
    output_edge = np.max(edge_detections, axis=0)
    return output_edge.astype(np.uint8)

def hsv_range_channels(hue_img, saturation_img, value_img):
    w1 = 0
    while w1 <= 1:
        display = []
        w2 = 0
        while w2 <= 1 - w1:
            w3 = 1 - w1 - w2 
            channel1 = w1 * value_img
            channel2 = w1 * value_img + w2 * value_img * saturation_img
            channel3 = w1 * value_img + w2 * value_img * saturation_img + w3 * value_img * saturation_img * hue_img

            channel1 = (channel1 / np.max(channel1) * 255).astype(np.uint8)
            channel2 = (channel2 / np.max(channel2) * 255).astype(np.uint8)
            channel3 = (channel3 / np.max(channel3) * 255).astype(np.uint8)
            
            channel1_canny = cv.Canny(channel1, 0.25 * 0.4 * 255, 0.25 * 255, L2gradient=True)
            channel2_canny = cv.Canny(channel2, 0.25 * 0.4 * 255, 0.25 * 255, L2gradient=True)
            channel3_canny = cv.Canny(channel3, 0.25 * 0.4 * 255, 0.25 * 255, L2gradient=True)
            
            channel1_canny = cv.Canny(channel1, 75, 100, L2gradient=True)
            channel2_canny = cv.Canny(channel2, 75, 100, L2gradient=True)
            channel3_canny = cv.Canny(channel3, 75, 100, L2gradient=True)

            edge_detections = np.array([channel1_canny, channel2_canny, channel3_canny])
            output_edge = np.max(edge_detections, axis=0)
            display.append(output_edge)
            w2 += 0.1
        display_len = len(display)
        for i, img in enumerate(display):
            plt.subplot(1, display_len, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{round(w1, 2)}, {round(i * 0.1, 2)}')
        plt.show()
        w1 += 0.1


def hsv_mean_channels(img, low_threshold, high_threshold):
    hue_img = img[:, :, 0]
    saturation_img = img[:, :, 1]
    value_img = img[:, :, 2]

    w1 = 0.3
    w2 = 0.6
    w3 = 0.1

    channel1 = w1 * value_img
    channel2 = w1 * value_img + w2 * value_img * saturation_img
    channel3 = w1 * value_img + w2 * value_img * saturation_img + w3 * value_img * saturation_img * hue_img

    channel1 = (channel1 / np.max(channel1) * 255).astype(np.uint8)
    channel2 = (channel2 / np.max(channel2) * 255).astype(np.uint8)
    channel3 = (channel3 / np.max(channel3) * 255).astype(np.uint8)

    channel1_canny = cv.Canny(channel1, low_threshold, high_threshold, L2gradient=True)
    channel2_canny = cv.Canny(channel2, low_threshold, high_threshold, L2gradient=True)
    channel3_canny = cv.Canny(channel3, low_threshold, high_threshold, L2gradient=True)

    # channel1_canny = cv.Canny(channel1, 75, 100, L2gradient=True)
    # channel2_canny = cv.Canny(channel2, 75, 100, L2gradient=True)
    # channel3_canny = cv.Canny(channel3, 75, 100, L2gradient=True)

    edge_detections = np.array([channel1_canny, channel2_canny, channel3_canny])
    output_edge = np.mean(edge_detections, axis=0)
    return output_edge.astype(np.uint8)

low_477 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\477.png")
low_523 = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\our485\low\523.png")

img_in_use = low_523

grayscale_version = (rgb2gray(img_in_use)).astype(np.uint8)
cv_canny = cv.Canny(grayscale_version, 12, 30, L2gradient=True)

hsv_img = rgb2hsv(img_in_use)
hue_img = hsv_img[:, :, 0]
saturation_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]
    
# plt.subplot(1, 4, 1)
# plt.imshow(img_in_use)
# plt.axis('off')
# plt.title('image')

# plt.subplot(1, 4, 2)
# plt.imshow(hue_img, cmap='hsv')
# plt.axis('off')
# plt.title('Hue channel')

# plt.subplot(1, 4, 3)
# plt.imshow(saturation_img)
# plt.axis('off')
# plt.title('Saturation channel')

# plt.subplot(1, 4, 4)
# plt.imshow(value_img)
# plt.axis('off')
# plt.title('Value channel')

# plt.show()

high_threshold = 200
low_threshold = high_threshold * 0.4

hsv_canny = hsv(hue_img, saturation_img, value_img, low_threshold, high_threshold)
#hsv_temp = hsv_range_channels(hue_img, saturation_img, value_img)
hsv_max_channels_canny = hsv_max_channels(hsv_img, low_threshold, high_threshold)
hsv_mean_channels_canny = hsv_mean_channels(hsv_img, low_threshold, high_threshold)

# plt.subplot(1, 4, 1)
# plt.imshow(cv_canny)
# plt.axis('off')
# plt.title('cv')

plt.subplot(1, 2, 1)
plt.imshow(hsv_canny)
plt.axis('off')
plt.title('combined')

plt.subplot(1, 2, 2)
plt.imshow(hsv_max_channels_canny)
plt.axis('off')
plt.title('hsv max channels')

# plt.subplot(1, 3, 3)
# plt.imshow(hsv_mean_channels_canny)
# plt.axis('off')
# plt.title('hsv mean channels')

plt.show()