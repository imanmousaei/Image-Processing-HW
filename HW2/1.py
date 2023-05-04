import cv2


def resize_image(img, scale_percent):
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def sobel_edge_detection(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    
    mag = cv2.magnitude(sobelx, sobely)
    
    mag_scaled = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return mag_scaled


def canny_edge_detection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    edges = cv2.Canny(image=img_blur, threshold1=70, threshold2=250)

    return edges


def threshold_edge_detection(img):
    # Detect the edges using adaptive thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median blur to reduce noise
    blur = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    return edges


def blur_image(img, kernel_size=(3, 3)):
    blur = cv2.GaussianBlur(img, kernel_size, 0)
    return blur


def smooth_image(img):
    smooth = cv2.bilateralFilter(img, 9, 250, 250)
    return smooth


def blacken_edges(img, threshold=200):
    # Apply binary threshold to create mask
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Invert mask
    mask = cv2.bitwise_not(mask)

    return mask


def cartoonize1(img, edges_weight=10):
    edges = canny_edge_detection(img)
    edges = blacken_edges(edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    blur = blur_image(img, (3, 3))

    cartoon = blur + edges*edges_weight
    return cartoon


def cartoonize2(img, edges_weight=2):
    edges = threshold_edge_detection(img) * edges_weight

    smooth = smooth_image(img)

    cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)

    return cartoon

def cartoonize3(img, edges_weight=10):
    edges = sobel_edge_detection(img)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    smooth = smooth_image(img)

    cartoon = smooth + edges*edges_weight

    return cartoon


def show_image(img, label):
    cv2.imshow(label, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('img/img1.jpg')
    # img = cartoonize1(img, edges_weight=10)
    img = cartoonize2(img, edges_weight=2)
    # img = cartoonize3(img, edges_weight=10)
    
    cv2.imwrite('outputs/cartoon/2.png', img)
    
    # img = resize_image(img, 70)
    # show_image(img, 'cartoon')
