import numpy as np
#Canny Edge Detector
#1. Smoothing
#2. Finding Gradients
#3. Non-maximum Suppression
#4. Double Thresholding
#5. Edge Tracking by Hysterisis


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.
    """

    kernel = np.zeros((size, size))
    coeff = 1 / (2 * np.pi * (sigma ** 2))
    k = (size - 1) / 2
    for row in range(size):
        for col in range(size):
            inside = - ((row - k) ** 2 + (col - k) ** 2)/(2 * sigma ** 2)
            exponential = np.exp(inside)
            kernel[row][col] = coeff * exponential
    return kernel


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    for row in range(Hi):
        for col in range(Wi):
            temp = np.multiply(kernel, padded[row: row + Hk, col: col + Wk])
            out[row][col] = np.sum(temp)
    return out


def partial_x(img):
    """ Computes partial x-derivative of input img.
    """
    Dx = np.array([[-1/2, 0, 1/2]])
    out = conv(img, Dx)
    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.
    """
    Dy = np.array([[-1/2], [0], [1/2]])
    out = conv(img, Dy)
    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.
    """
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx * Gx + Gy * Gy)
    theta = np.degrees(np.arctan2(Gy, Gx))
    theta[theta<0] += 360
    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    for i in range(H):
        for j in range(W):
            neighbor_vals = []
            if theta[i][j] in [0, 180]:
                if j - 1 in range(W):
                    neighbor_vals.append(G[i][j-1])
                if j + 1 in range(W):
                    neighbor_vals.append(G[i][j + 1])
            elif theta[i][j] in [45, 225]:
                if i + 1 in range(H) and j + 1 in range(W):
                    neighbor_vals.append(G[i + 1][j + 1])
                if i - 1 in range(H) and j - 1 in range(W):
                    neighbor_vals.append(G[i - 1][j - 1])
            elif theta[i][j] in [90, 270]:
                if i - 1 in range(H):
                    neighbor_vals.append(G[i - 1][j])
                if i + 1 in range(H):
                    neighbor_vals.append(G[i + 1][j])
            else:
                if i + 1 in range(H) and j - 1 in range(W):
                    neighbor_vals.append(G[i + 1][j - 1])
                if i - 1 in range(H) and j + 1 in range(W):
                    neighbor_vals.append(G[i - 1][j + 1])
            if all (x < G[i][j] for x in neighbor_vals):   
                out[i][j] = G[i][j]
    return out


def double_thresholding(img, high, low):
    """
    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """
    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > high:
                strong_edges[i][j] = True
            elif img[i][j] > low and img[i][j] < high:
                weak_edges[i][j] = True
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)
    """
    neighbors = []
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))
    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    added_edges = set()
    queue = []
    for [i, j] in indices:
        queue.append([i, j])
        while len(queue) != 0:
            elem = queue.pop(0)
            neighbor_indices = get_neighbors(elem[0], elem[1], H, W)
            for x, y in neighbor_indices:
                if weak_edges[x][y] == True:
                    edges[x][y] = True
                    if (x, y) not in added_edges:
                        queue.append([x, y])
                        added_edges.add((x, y))
    return edges

import matplotlib.pyplot as plt

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    
    plt.show()
    return edge