import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def gen_negative(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i][j][k] = 255 - img[i][j][k]


def color_to_grayscale(orig_img, gray_img):
    for i in range(orig_img.shape[0]):
        for j in range(orig_img.shape[1]):
            gray_img[i][j] = max(orig_img[i][j])/2 + min(orig_img[i][j])/2


def print_histogram(img):
    plt.figure()
    if len(img.shape) == 3:
        plt.subplot(311)
        hist, bins, patches = plt.hist(img[:, :, 0].flatten(), 256,[0,255])
        plt.title("B Channel")
        plt.subplot(312)
        hist, bins, patches = plt.hist(img[:, :, 1].flatten(), 256,[0,255])
        plt.title("G Channel")
        plt.subplot(313)
        hist, bins, patches = plt.hist(img[:, :, 2].flatten(), 256,[0,255])
        plt.title("R Channel")
    else:
        hist, bins, patches = plt.hist(img.flatten(), 256,[0,255])
        plt.title("Mono Channel")


def contrast_stretch(inp_img):
    if inp_img.shape[2] ==3:
        stretched_img = np.zeros((orig_img.shape[0], orig_img.shape[1], 3), dtype=np.uint8)
        for k in range(3):
            flattened_img_dim = inp_img[:,:,k].flatten()
            max_val = max(flattened_img_dim)
            min_val = min(flattened_img_dim)
            print (max_val,min_val,k)
            for i in range(inp_img.shape[0]):
                for j in range(inp_img.shape[1]):
                    stretched_img[i][j][k] = int(255 * ((inp_img[i][j][k] - min_val) * 1.0 / (max_val - min_val)))

    else:
        stretched_img = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.uint8)
        flattened_img = inp_img.flatten()
        max_val = max(flattened_img)
        min_val = min(flattened_img)
        for i in range(inp_img.shape[0]):
            for j in range(inp_img.shape[1]):
                stretched_img[i][j] = int(255 * ((inp_img[i][j] - min_val) * 1.0 / (max_val - min_val)))

    return stretched_img


def histogram_equilize_grayscale(inp_img):
    equilised_img = np.zeros((inp_img.shape[0], inp_img.shape[1]), dtype=np.uint8)
    num_pixels = inp_img.shape[0]*inp_img.shape[1]
    flattend_img = inp_img.flatten()
    hist, bins = np.histogram(flattend_img,256,[0,255])
    cdf = hist.cumsum()/float(num_pixels)
    for i in range(inp_img.shape[0]):
        for j in range(inp_img.shape[1]):
            equilised_img[i][j] = cdf[inp_img[i][j]] * 255
    return equilised_img


def histogram_equilize_color(inp_img):
    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    equilised_img = np.zeros((inp_img.shape[0], inp_img.shape[1],inp_img.shape[2]), dtype=np.uint8)
    num_pixels = inp_img.shape[0] * inp_img.shape[1]
    flattend_img = inp_img[:,:,2].flatten()
    hist, bins = np.histogram(flattend_img, 256, [0, 255])
    cdf = hist.cumsum() / float(num_pixels)
    for i in range(inp_img.shape[0]):
        for j in range(inp_img.shape[1]):
            equilised_img[i][j][0] = inp_img[i][j][0]
            equilised_img[i][j][1] = inp_img[i][j][1]
            equilised_img[i][j][2] = cdf[inp_img[i][j][2]] * 255
    return cv2.cvtColor(equilised_img,cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    imName = 'Awnings.jpg'
    imPath = os.path.join('./Images', imName)
    orig_img = cv2.imread(imPath)

    # gray_img = np.zeros((orig_img.shape[0], orig_img.shape[1]), dtype=np.uint8)
    # color_to_grayscale(orig_img, gray_img)

    # neg_img = np.copy(orig_img)
    # gen_negative(neg_img)

    # stretched_img = contrast_stretch(orig_img)
    equilised_img = histogram_equilize_color(orig_img)



    cv2.namedWindow('orig_window')

    cv2.namedWindow('neg_window')
    cv2.imshow('orig_window', orig_img)
    cv2.imshow('neg_window', equilised_img)

    print_histogram(orig_img)
    print_histogram(equilised_img)
    plt.show()

    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite("equilised"+imName+".jpg",equilised_img)