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


def add_noise(img, mean=0, variance=50):
    sigma = np.sqrt(variance)
    gaussian = np.random.normal(mean, sigma, img.shape)

    # Add Gaussian noise to image
    noisy_img = img + gaussian.astype('uint8')
    return noisy_img


def plot_losses(losses):
    lists = sorted(losses.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.title(imagename)
    plt.xlabel('Kernel Size')
    plt.ylabel(loss_function)
    plt.savefig(f'outputs/noise={variance}/{loss_function}-{imagename}')
    # plt.show()
    plt.clf() # clear plot


filenames = [
    'lena.tif', 
    'cameraman.tif', 
    'baboon.bmp'
]


if __name__ == '__main__':
    for filename in filenames:
        for variance in [10,50,100]:
            for loss_function in ['mse', 'psnr']:
                losses = {}
                for kernel_size in range(1,102,2):
                    img = cv2.imread(f'img/{filename}')
                    imagename = filename.split('.')[0]
                    
                    noisy = add_noise(img, variance=variance)
                    # show_image(noisy, 'noisy')
                    cv2.imwrite(f'outputs/noise={variance}/noisy-{imagename}.png', noisy)
                    
                    blur = gaussian_filter(noisy, kernel_size)
                    
                    if loss_function.lower() == 'mse':
                        loss = MSE(img, blur)
                    else:
                        loss = PSNR(img, blur)
                                    
                    losses[kernel_size] = loss
                    print(f'{loss_function} of {imagename} with kernel_size = {kernel_size} is {loss}')

                plot_losses(losses)