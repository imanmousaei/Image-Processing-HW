import cv2


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def draw_rectangle(img, contours):
    # Draw rectangle around the faces
    for (x, y, w, h) in contours:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    image_path = 'image1.jpg'
    img = cv2.imread(image_path)

    faces = detect_faces(img)
    draw_rectangle(img, faces)
    show_image(img)
