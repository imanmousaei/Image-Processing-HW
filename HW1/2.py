import numpy as np
import cv2
from sklearn.cluster import KMeans


def index_compress(image_path, num_colors=8):
    img = cv2.imread(image_path)

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


if __name__ == '__main__':
    image_path = 'img/image2.jpg'
    compressed_img = index_compress(image_path)
    cv2.imwrite(f'output/2-compressed_image2.jpg', compressed_img)
