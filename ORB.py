import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_binary_images_GS(path1, v11, v12, path2, v21, v22):
    global bn_img1, bn_img2
    img_1 = cv.imread(path1, 2)
    img_2 = cv.imread(path2, 2)

    r1, c1 = img_1.shape
    for i in range(r1):
        for j in range(c1):
            if img_1[i, j] <= v11 or img_1[i, j] >= v12: img_1[i, j] = 255
            else: img_1[i, j] = 0
    bn_img1 = img_1

    r2, c2 = img_2.shape
    for i in range(r2):
        for j in range(c2):
            if img_2[i, j] >= v21 and img_2[i, j] <= v22: img_2[i, j] = 255
            else: img_2[i, j] = 0
    bn_img2 = img_2

def get_binary_images_RGB(path1, v11r, v12r, v11g, v12g, v11b, v12b, path2, v21r, v22r, v21g, v22g, v21b, v22b):
    global bn_img1, bn_img2
    img_1 = cv.imread(path1, cv.IMREAD_COLOR)
    img_1_b, img_1_g, img_1_r = cv.split(img_1)

    img_2 = cv.imread(path2, cv.IMREAD_COLOR)
    img_2_b, img_2_g, img_2_r = cv.split(img_2)

    r1, c1 = img_1_r.shape
    for i in range(r1):
        for j in range(c1):
            if (img_1_r[i, j] >= v11r and img_1_r[i, j] <= v12r) and\
                (img_1_g[i, j] >= v11g and img_1_g[i, j] <= v12g) and\
                (img_1_b[i, j] >= v11b and img_1_b[i, j] <= v12b):
                img_1_r[i, j] = 0
                img_1_g[i, j] = 0
                img_1_b[i, j] = 0
    img_1 = cv.merge((img_1_r, img_1_g, img_1_b))
    bn_img1 = img_1

    r2, c2 = img_2_r.shape
    for i in range(r2):
        for j in range(c2):
            if (img_2_r[i, j] >= v21r and img_2_r[i, j] <= v22r) and\
                (img_2_g[i, j] >= v21g and img_2_g[i, j] <= v22g) and\
                (img_2_b[i, j] >= v21b and img_2_b[i, j] <= v22b):
                img_2_r[i, j] = 0
                img_2_g[i, j] = 0
                img_2_b[i, j] = 0
    img_2 = cv.merge((img_2_r, img_2_g, img_2_b))
    bn_img2 = img_2

def pixels_filter(img1, img2):
    global filter_img1, filter_img2
    n = 0
    while n<10:
        r1, c1 = img1.shape
        for i in range(r1-1):
            for j in range(c1-1):
                k = 0
                if img1[i, j]==0:
                    if img1[i, j-1]==0: k+=1
                    if img1[i, j+1]==0: k+=1
                    if img1[i-1, j]==0: k+=1
                    if img1[i+1, j]==0: k+=1
                if k<2: img1[i, j] = 255
        filter_img1 = img1

        r2, c2 = img2.shape
        for i in range(r2-1):
            for j in range(c2-1):
                k = 0
                if img2[i, j]==0:
                    if img2[i, j-1]==0: k+=1
                    if img2[i, j+1]==0: k+=1
                    if img2[i-1, j]==0: k+=1
                    if img2[i+1, j]==0: k+=1
                if k<2: img2[i, j] = 255
        filter_img2 = img2
        n += 1

def void_filling(img1, img2):
    global fill_img1, fill_img2
    r1, c1 = img1.shape
    for i in range(r1-1):
        for j in range(c1-1):
            k = 0
            if img1[i, j]==0:
                if img1[i, j-1]==1: k+=1
                if img1[i, j+1]==1: k+=1
                if img1[i-1, j]==1: k+=1
                if img1[i+1, j]==1: k+=1
            if k>2: img1[i, j] = 255
    fill_img1 = img1

    r2, c2 = img2.shape
    for i in range(r2-1):
        for j in range(c2-1):
            k = 0
            if img2[i, j]==0:
                if img2[i, j-1]==1: k+=1
                if img2[i, j+1]==1: k+=1
                if img2[i-1, j]==1: k+=1
                if img2[i+1, j]==1: k+=1
            if k>2: img2[i, j] = 255
    fill_img2 = img2

def clustering (path1, path2, k_value):
    paths = [path1, path2]
    global cl_img_gr, cl_img

    i = 0
    for p in paths:
        image = cv.imread(p)
        pixel_vals = image.reshape((-1,3))
        pixel_vals = np.float32(pixel_vals)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = k_value
        retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)

        step = 256/k
        value = 0
        for ar in centers:
            for v in ar:
                segmented_image[segmented_image == v] = value
            value += step

        if i == 0: cl_img_gr = segmented_image
        elif i == 1: cl_img = segmented_image

        i += 1

get_binary_images_GS('Data/yandex_2gis_moscow/yandex_msc_gr.tif', 160, 210,
                  'Data/yandex_2gis_moscow/2gis_msc.tif', 200, 230)

orb = cv.ORB_create(patchSize=256)
gp1, des1 = orb.detectAndCompute(bn_img1, None)
gp2, des2 = orb.detectAndCompute(bn_img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img_rslt = cv.drawMatches(bn_img1, gp1, bn_img2, gp2, matches[:], None)

plt.imshow(img_rslt)
plt.axis('off')
plt.show()