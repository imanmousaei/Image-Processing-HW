import cv2


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=[100, 100])
    return faces


def draw_rectangle(img, contours):
    # Draw rectangle around the faces
    for (x, y, w, h) in contours:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def resize_image(img, scale_percent):
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    image_path = 'image1.jpg'
    img = cv2.imread(image_path)

    faces = detect_faces(img)
    img = draw_rectangle(img, faces)
    img = resize_image(img, 20)
    show_image(img)
