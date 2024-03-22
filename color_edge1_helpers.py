from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage

plt.rcParams['figure.figsize'] = (15.0, 12.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from original_canny import canny

img = io.imread(r'C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\iguana.png')#, as_gray=True)
noisy_img = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\noisy_image.jpeg")
solid_color = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\solid_color.jpeg")
colored_blocks = io.imread(r"C:\Users\jimdw\OneDrive-Stanford\Documents\Junior Year\Winter Quarter\CS 131\CS131_release\winter_2024\final_project\colored_blocks.jpeg")


def get_median(input):
    indeces = np.argsort(input)
    mid = len(input) // 2
    return indeces[mid], input[indeces[mid]]


def levelB(z_min, z_med, z_max, z_xy):
    b_1 = z_xy - z_min
    b_2 = z_xy - z_max
    if b_1 > 0 and b_2 < 0:
        return z_xy
    else:
        return z_med


def levelA(z_min, z_med, z_max, z_xy, s_xy, s_max):
    a_1 = z_med - z_min
    a_2 = z_med - z_max 
    if a_1 > 0 and a_2 < 0:
        return levelB(z_min, z_med, z_max, z_xy)
    else:
        s_xy += 2
        if s_xy <= s_max:
            return levelA(z_min, z_med, z_max, z_xy, s_xy, s_max)
        else:
            return z_xy

def adaptive_median_kernel(img, initial_window, max_window):
    grayscale_image = rgb2gray(img) 
    x_length, y_length = grayscale_image.shape

    s_xy = initial_window
    s_max = max_window

    s_offset = int(s_xy // 2)

    pad_width0 = s_offset
    pad_width1 = s_offset
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(grayscale_image, pad_width, mode='edge')

    output = img.copy()

    for row in range(x_length - 1):
        for col in range(y_length - 1):
            cur_window = padded[row: row + s_xy, col: col + s_xy]
            reshaped = cur_window.reshape(-1)
            z_min = np.min(reshaped)
            index, z_med, = get_median(reshaped) # different than numpy medium implementation
            z_max = np.max(reshaped)
            z_xy = padded[row, col]

            new_val = levelA(z_min, z_med, z_max, z_xy, s_xy, s_max)
            if new_val != z_xy:
                x, y = index // 3, index % 3
                output[row][col] = output[row + x - 1][col + y - 1]
    return output

def rgb2gray(rgb):
    if(len(rgb.shape) == 3):
        return np.float64(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    else:#already a grayscale
        return rgb
    

def find_difference(img, filtered, length):
    count = 0
    new_image = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if np.any(img[row, col] - filtered[row][col] != 0):
                count += 1
                temp = img[row, col]
                if length == 3:
                    new_image[row][col][:] = [255, 255, 255]
                else:
                    new_image[row][col][:] = [0, 0, 0, 255]
    print(count)
    return new_image

def plot_images(img, filtered, length):
    new_image = find_difference(img, filtered, length)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('image')


    plt.subplot(1, 3, 2)
    plt.imshow(filtered)
    plt.axis('off')
    plt.title('Filtered')

    plt.subplot(1, 3, 3)
    plt.imshow(new_image)
    plt.axis('off')
    plt.title('Difference')

    plt.show()


def weighted_average_image(filtered):
    shape = filtered.shape
    output = np.zeros((shape[0], shape[1]))

    for row in range(shape[0]):
        for col in range(shape[1]):
            red, green, blue = 0, 0, 0
            if filtered[row][col].shape == (4,):
                [red, green, blue, _] = filtered[row][col]
            else:
                [red, green, blue] = filtered[row, col]
            output[row, col] = 2 * red + 3 * green + 4 * blue
    return output


def max_directional_difference(filtered):
    shape = filtered.shape
    output = np.zeros(shape)

    pad_width = ((0,2),(0,2))
    padded = np.pad(filtered, pad_width, mode='edge')

    for row in range(shape[0]):
        for col in range(shape[1]):
            # zero_rot = abs(filtered[row + 1, col] - filtered[row + 1, col + 2])
            # forty_five_rot = abs(filtered[row + 2, col] - filtered[row, col + 2])
            # ninty_rot = abs(filtered[row, col + 1] - filtered[row + 2, col + 1])
            # hundred_thirty_five_rot = abs(filtered[row, col] - filtered[row + 2, col + 2])
            zero_rot = abs(padded[row + 1, col] - padded[row + 1, col + 2])
            forty_five_rot = abs(padded[row + 2, col] - padded[row, col + 2])
            ninty_rot = abs(padded[row, col + 1] - padded[row + 2, col + 1])
            hundred_thirty_five_rot = abs(padded[row, col] - padded[row + 2, col + 2])
            output[row, col] = max([zero_rot, forty_five_rot, ninty_rot, hundred_thirty_five_rot])
    return output

def threshold(img):
    t = np.sum(img) / (img.shape[0] * img.shape[1])
    T = 1.2 * t

    output = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col] >= T:
                output[row][col] = img[row][col]
    output = output / output.max() * 255
    return output


def edge_thinning(img):
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_y = sobel_x.T
    
    x = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
    y = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
    
    laplacian = np.array([[0, -1, 0], 
                           [-1, 4, -1], 
                           [0, -1, 0]])
    
    Ix = ndimage.filters.convolve(img, x)
    Iy = ndimage.filters.convolve(img, y)
    
    G = np.hypot(Ix, Iy)
    return G

