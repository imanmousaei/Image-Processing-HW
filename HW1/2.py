import numpy as np
import cv2
from sklearn.cluster import KMeans


def index_compress(img, num_colors=8):
    # Reshape each channel to a 1D array of pixels
    pixels = img.reshape(-1, 3)

    # Use KMeans to find most common colors
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto').fit(pixels)

    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    # Replace each pixel with the closest color
    new_pixels = np.array([kmeans.cluster_centers_[label] for label in kmeans.labels_])
    new_img = new_pixels.reshape(img.shape)
    
    # print(new_img)

    return new_img


def convert_to_HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def convert_to_LAB(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return lab

def convert_to_YCrCb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return ycrcb

if __name__ == '__main__':
    image_path = 'img/image2.jpg'
    img = cv2.imread(image_path)
    num_colors = 25
    
    compressed_img = index_compress(img, num_colors)
    cv2.imwrite(f'output/2-compressed_image2-colors-{num_colors}.jpg', compressed_img)

    lab = convert_to_LAB(compressed_img)
    hsv = convert_to_HSV(compressed_img)
    ycrcb = convert_to_YCrCb(compressed_img)