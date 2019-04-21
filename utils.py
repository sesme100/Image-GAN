import cv2
import numpy as np

from Define import *

def Save_Images(data, filename = 'save.jpg', show_img_cnt = SHOW_IMG_COUNT):
    assert len(data) == show_img_cnt
    
    show_img = np.zeros([(28 * SHOW_IMG_H), (28 * SHOW_IMG_W)], dtype = np.float32)

    for i in range(SHOW_IMG_H):
        for j in range(SHOW_IMG_W):
            show_img[i*28:(i+1)*28, j*28:(j+1)*28] = data[i*SHOW_IMG_H + j]

    show_img = show_img.astype(np.uint8)
    cv2.imwrite(filename, show_img)

