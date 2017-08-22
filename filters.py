import cv2
import os
import numpy as np

def uniform_blur(inp_img):
    border_img = cv2.copyMakeBorder(inp_img,1,1,1,1,borderType=cv2.BORDER_CONSTANT,value=0)
    size = 4
    kernel = np.ones((size, size), np.float32) / (size**2)
    blur = cv2.filter2D(border_img,ddepth=-1,kernel=kernel)
    return blur[1:blur.shape[0]-1,1:blur.shape[1]-1,:]


if __name__ == '__main__':

    imName = 'Awnings.jpg'
    imPath = os.path.join('./Images', imName)
    orig_img = cv2.imread(imPath)

    uniform_blurred_img = uniform_blur(orig_img)
    cv2.namedWindow('Window 1')
    cv2.namedWindow('Window 2')
    print (orig_img.shape)
    print (uniform_blurred_img.shape)
    cv2.imshow('Window 1', orig_img)
    cv2.imshow('Window 2', uniform_blurred_img)

    cv2.waitKey(0)