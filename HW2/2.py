import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(img, scale_percent):
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def show_image(img, label):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    
def MSE(img1, img2):
    diff = img1 - img2

    squared_diff = diff ** 2

    mse_loss = np.mean(squared_diff)
    return mse_loss

def PSNR(img1, img2):
    mse = MSE(img1, img2)
    l = 255
    psnr = 10 * np.log10(l*l/mse)
    return psnr
    
    
def gaussian_filter(img, kernel_size):
    # apply Gaussian blur with kernel size of kernel_size x kernel_size and standard deviation of 0
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur


filenames = [
    'lena.tif', 
    'cameraman.tif', 
    'baboon.bmp'
]

loss_function = 'psnr'

if __name__ == '__main__':
    for imagename in filenames:
        losses = []
        for kernel_size in range(1,102,2):
            img = cv2.imread(f'img/{imagename}')
            
            blur = gaussian_filter(img, kernel_size)
            
            if loss_function.lower() == 'mse':
                loss = MSE(img, blur)
            else:
                loss = PSNR(img, blur)
            
            losses.append(loss)
            print(f'loss of {imagename} with kernel_size = {kernel_size} is {loss}')

        plt.plot(losses)
        plt.title(imagename)
        plt.xlabel('MSE')
        plt.ylabel('Kernel Size')
        plt.show()