import cv2
import numpy as np
import os


def display_zoom(event, x, y, flags, param):
    """ mouse callback function"""
    global img
    global orig_img
    global zoom_img_crop
    global zoom_img
    global strata
    img = np.copy(orig_img)

    if event == cv2.EVENT_LBUTTONDOWN:
        if strata == 'REPEAT':
            strata = 'BILINEAR'
            zoom_img = resize_im(orig_img, strategy=strata)
        else:
            strata = 'REPEAT'
            zoom_img = resize_im(orig_img, strategy=strata)
    if event == cv2.EVENT_MOUSEMOVE:
        if x+50 < img.shape[1] and x > 50 and y > 50 and y+50 < img.shape[0]:
            zoom_img_crop = zoom_img[5*y - 250 : 5*y + 250, 5*x - 250 : 5*x + 250]
        cv2.rectangle(img, (x-50, y-50), (x+50, y+50), (0, 0, 0))


def resize_im(temp, strategy='BILINEAR'):
    """ Takes in temp : orig_img and returns new numpy array of  expanded size according to chosen strategy"""
    mat = np.ones((temp.shape[0]*5, temp.shape[1] * 5, 3), dtype=np.uint8)

    if strategy == 'REPEAT':
        for row_num in range(temp.shape[0]):
            for col_num in range(temp.shape[1]):
                mat[row_num*5:row_num*5 + 5, col_num*5:col_num*5+5]=temp[row_num][col_num]

    if strategy == 'BILINEAR':
        for row_num in range(temp.shape[0] - 1):
            for col_num in range(temp.shape[1] - 1):
                for i in range(5):
                    for j in range(5):
                        p_i1 = (1 - i / 5.0) * temp[row_num][col_num] + (i / 5.0) * temp[row_num][col_num + 1]
                        p_i2 = (1 - i / 5.0) * temp[row_num + 1][col_num] + (i / 5.0) * temp[row_num + 1][col_num + 1]
                        p = (1 - j / 5.0) * p_i1 + (j / 5.0) * p_i2
                        mat[row_num * 5 + j, col_num * 5 + i] = np.uint8(p)

    return mat


# Create a black image, a window and bind the function to window
imName = 'wizard.jpg'
imPath = os.path.join('./Images', imName)
orig_img = cv2.imread(imPath)
img = np.copy(orig_img)


zoomWin = cv2.namedWindow('zoom')
rootWin = cv2.namedWindow('image')
cv2.moveWindow('image', 0, 0)
cv2.moveWindow('zoom', img.shape[1], 0)
cv2.setMouseCallback('image', display_zoom)

strata = 'BILINEAR'
zoom_img = resize_im(orig_img, strategy=strata)
#zoom_img = cv2.resize(orig_img, (orig_img.shape[1]*5, orig_img.shape[0]*5), interpolation=cv2.INTER_LINEAR)
zoom_img_crop = np.zeros((500, 500, 3), dtype=np.uint8)

while True:

    cv2.imshow('zoom', zoom_img_crop)
    cv2.imshow('image', img)
    key = cv2.waitKey(20)
    if key & 0xFF == 27 or key & 0xFF == 13:
        break
    if key & 0xFF == ord('s'):
        cv2.imwrite('out_'+strata+'.jpg',zoom_img_crop)
        break
cv2.destroyAllWindows()


