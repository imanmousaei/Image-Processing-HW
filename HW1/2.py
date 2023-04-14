import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def index_compress(image_path, num_colors=8):
    # img = cv2.imread(image_path)
    img = Image.open(image_path)
    img_array = np.array(img)

    # Reshape each channel to a 1D array of pixels
    pixels = img_array.reshape(-1, 3)

    # Use KMeans to find most common colors
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto').fit(pixels)

    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)

    # Replace each pixel with the closest color
    new_pixels = np.array([kmeans.cluster_centers_[label] for label in kmeans.labels_])
    new_img_array = new_pixels.reshape(img_array.shape)
    
    # print(new_img_array)
    
    new_img = Image.fromarray(new_img_array.astype('uint8'), 'RGB')
    new_img.save('output/2-compressed_image1.jpg')


if __name__ == '__main__':
    image_path = 'img/image1.jpg'
    index_compress(image_path)
