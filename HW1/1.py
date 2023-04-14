import cv2
import numpy as np


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces larger than 100x100
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=[100, 100])
    return faces


def draw_rectangle(img, contour):
    # Draw rectangle around the faces
    for (x, y, w, h) in contour:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def resize_image(img, scale_percent):
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def show_image(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_contour(img, contour):
    # Crop and save the detected face region
    for i, (x, y, w, h) in enumerate(contour):
        cropped = img[y:y+h, x:x+w]
        cv2.imwrite(f'output/face{i}.jpg', cropped)


def mul_contour(contour, mul):
    contour *= mul

    return contour


def flip_contour(contour):
    # flip up and down
    contour = np.flipud(contour)
    return contour


def rotate_rgb_image(image, angle):
    theta = np.deg2rad(angle)

    # We consider the rotated image to be of the same size as the original
    rot_img = np.uint8(np.zeros(image.shape))

    # Finding the center point of rotated (or original) image.
    height, width, channels = rot_img.shape
    mid_x, mid_y = (width//2, height//2)

    for i in range(height):
        for j in range(width):
            x = (i - mid_x) * np.cos(theta) + (j - mid_y) * np.sin(theta)
            y = -(i - mid_x) * np.sin(theta) + (j - mid_y) * np.cos(theta)

            x = round(x) + mid_x
            y = round(y) + mid_y

            if (x >= 0 and y >= 0 and x < height and y < width):
                rot_img[i, j, :] = image[x, y, :]

    return rot_img


def add_contour(contour, add):
    contour = contour + add

    return contour


def shift_contour(contour, shift_x=0, shift_y=0):
    shifted_image = np.roll(contour, shift_x, axis=1)
    shifted_image = np.roll(shifted_image, shift_y, axis=0)

    return shifted_image


def transform_contours(img, contours, transformation='rotate'):
    # mul and add are not Geometric Transformations but I did them for fun

    for (x, y, w, h) in contours:
        cropped = img[y:y+h, x:x+w]

        if transformation == 'rotate':
            img[y:y+h, x:x+w] = rotate_rgb_image(cropped, angle=45)
        elif transformation == 'flipud':
            img[y:y+h, x:x+w] = flip_contour(cropped)
        elif transformation == 'shift':
            img[y:y+h, x:x+w] = shift_contour(cropped, shift_x=200)
        elif transformation == 'mul':
            img[y:y+h, x:x+w] = mul_contour(cropped, mul=2)
        elif transformation == 'add':
            img[y:y+h, x:x+w] = add_contour(cropped, add=20)

    return img


if __name__ == '__main__':
    image_path = 'img/image1.jpg'
    img = cv2.imread(image_path)

    faces = detect_faces(img)
    # img = draw_rectangle(img, faces)

    transformation = 'add'
    img = transform_contours(img, faces, transformation)
    cv2.imwrite(f'output/1-{transformation}.jpg', img)

    # resize to fit in my screen
    img = resize_image(img, 20)
    show_image(img)
