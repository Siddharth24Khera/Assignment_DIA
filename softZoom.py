import cv2
import numpy as np
import os


def display_zoom_in(event, x, y, flags, param):
    """ mouse callback function"""
    global img
    global orig_img
    global zoom_in_img_crop
    global zoom_in_img
    global strata_in
    img = np.copy(orig_img)

    if event == cv2.EVENT_LBUTTONDOWN:
        if strata_in == 'REPEAT':
            strata_in = 'BILINEAR'
            zoom_in_img = resize_im_in(orig_img, strategy=strata_in)
        else:
            strata_in = 'REPEAT'
            zoom_in_img = resize_im_in(orig_img, strategy=strata_in)
    if event == cv2.EVENT_MOUSEMOVE:
        if x+50 < img.shape[1] and x > 50 and y > 50 and y+50 < img.shape[0]:
            zoom_in_img_crop = zoom_in_img[5*y - 250 : 5*y + 250, 5*x - 250 : 5*x + 250]
        cv2.rectangle(img, (x-50, y-50), (x+50, y+50), (0, 0, 0))


def resize_im_in(orig_img, strategy='BILINEAR'):
    """ Takes in orig_img : orig_img and returns new numpy array of  expanded size according to chosen strategy"""
    mat = np.ones((orig_img.shape[0]*5, orig_img.shape[1] * 5, 3), dtype=np.uint8)

    if strategy == 'REPEAT':
        for row_num in range(orig_img.shape[0]):
            for col_num in range(orig_img.shape[1]):
                mat[row_num*5:row_num*5 + 5, col_num*5:col_num*5+5]=orig_img[row_num][col_num]

    if strategy == 'BILINEAR':
        for row_num in range(orig_img.shape[0] - 1):
            for col_num in range(orig_img.shape[1] - 1):
                for i in range(5):
                    for j in range(5):
                        p_i1 = (1 - i / 5.0) * orig_img[row_num][col_num] + (i / 5.0) * orig_img[row_num][col_num + 1]
                        p_i2 = (1 - i / 5.0) * orig_img[row_num + 1][col_num] + (i / 5.0) * orig_img[row_num + 1][col_num + 1]
                        p = (1 - j / 5.0) * p_i1 + (j / 5.0) * p_i2
                        mat[row_num * 5 + j, col_num * 5 + i] = np.uint8(p)

    return mat

def display_zoom_out(event, x, y, flags, param):
    global img
    global orig_img
    global zoom_out_img_crop
    global zoom_out_img
    global strata_out
    img = np.copy(orig_img)

    if event == cv2.EVENT_LBUTTONDOWN:
        if strata_out == 'DROP':
            strata_out = 'AVG'
            zoom_out_img = resize_im_out(orig_img, strategy=strata_out)
        else:
            strata_out = 'DROP'
            zoom_out_img = resize_im_out(orig_img, strategy=strata_out)
    if event == cv2.EVENT_MOUSEMOVE:
        if x + 150 < img.shape[1] and x > 150 and y > 150 and y + 150 < img.shape[0]:
            pass
            zoom_out_img_crop = zoom_out_img[y / 3 - 50: y/3 + 50, x/3 - 50: x/3 + 50]
        cv2.rectangle(img, (x - 150, y - 150), (x + 150, y + 150), (0, 0, 0))

def resize_im_out(orig_img, strategy='DROP'):
    reduceBy = 3       # Use odd numbers only
    mat = np.ones((orig_img.shape[0] / reduceBy, orig_img.shape[1] / reduceBy, 3), dtype=np.uint8)

    if strategy == 'DROP':
        for row_num in range(mat.shape[0]):
            for col_num in range(mat.shape[1]):
                mat[row_num,col_num] = orig_img[reduceBy/2 + row_num*reduceBy][reduceBy/2 + col_num*reduceBy]

    if strategy == 'AVG':
        temp_slice = orig_img[0:orig_img.shape[0] - (orig_img.shape[0]%3),0:orig_img.shape[1] - (orig_img.shape[1]%3),:]

        for row_num in range(mat.shape[0]):
            for col_num in range(mat.shape[1]):
                mat[row_num, col_num,0] = temp_slice[row_num*reduceBy : row_num*reduceBy+reduceBy,col_num*reduceBy : col_num*reduceBy+reduceBy,0].mean()
                mat[row_num, col_num, 1] = temp_slice[row_num * reduceBy: row_num * reduceBy + reduceBy,col_num * reduceBy: col_num * reduceBy + reduceBy, 1].mean()
                mat[row_num, col_num, 2] = temp_slice[row_num * reduceBy: row_num * reduceBy + reduceBy,col_num * reduceBy: col_num * reduceBy + reduceBy, 2].mean()

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

zoom_status = 0   # 0 for zoomIn; 1 for zoomOut
cv2.setMouseCallback('image', display_zoom_in)

strata_in = 'REPEAT'        # REPEAT OR BILINEAR
strata_out = 'DROP'       # DROP OR AVG

zoom_in_img = resize_im_in(orig_img, strategy=strata_in)
zoom_out_img = resize_im_out(orig_img, strategy=strata_out)
#zoom_img = cv2.resize(orig_img, (orig_img.shape[1]*5, orig_img.shape[0]*5), interpolation=cv2.INTER_LINEAR)
zoom_in_img_crop = np.zeros((500, 500, 3), dtype=np.uint8)
zoom_out_img_crop = np.zeros((100, 100, 3), dtype=np.uint8)

while True:
    cv2.imshow('image', img)
    if zoom_status == 0:
        cv2.imshow('zoom', zoom_in_img_crop)
    else:
        cv2.imshow('zoom', zoom_out_img_crop)
    key = cv2.waitKey(20)
    if key & 0xFF == 32:
        if zoom_status == 0:
            cv2.setMouseCallback('image', display_zoom_out)
        else :
            cv2.setMouseCallback('image', display_zoom_in)
        zoom_status = 1 - zoom_status

    if key & 0xFF == 27 or key & 0xFF == 13:
        break
    if key & 0xFF == ord('s'):
        cv2.imwrite('out_'+strata_in+'.jpg',zoom_in_img_crop)
        break
cv2.destroyAllWindows()


