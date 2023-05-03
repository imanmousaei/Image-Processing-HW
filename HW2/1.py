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
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    
    return sobelxy

def canny_edge_detection(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    edges = cv2.Canny(image=img_blur, threshold1=70, threshold2=250)
    
    return edges

def smooth_image(img, kernel_size=(5, 5)):
    blur = cv2.GaussianBlur(img, kernel_size, 0)
    return blur

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

    smooth = smooth_image(img, (3,3))
    
    cartoon = smooth + edges*edges_weight
    return cartoon
    

def show_image(img, label):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    img = cv2.imread('img/img1.jpg')
    img = cartoonize1(img)
    
    img = resize_image(img, 70)
    show_image(img, 'cartoon')
    