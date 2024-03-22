import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import matplotlib.image as mpimg

path = r"C:\Users\jimdw\OneDrive-Stanford\Pictures\Screenshots\Screenshot 2024-03-20 230614.png"

img = mpimg.imread(path)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

plt.imshow(rgb2gray(img), cmap=plt.get_cmap('gray'))
plt.show()