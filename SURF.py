import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

surf = cv.xfeatures2d.SURF_create(800)

path1 = 'Data/yandex_2gis_moscow/yandex_msc_gr.tif'
path2 = 'Data/yandex_2gis_moscow/2gis_msc.tif'

def clustering(path1, path2, k_value):
  paths = [path1, path2]
  global cl_img_gr, cl_img

  i = 0
  for p in paths:
    image = cv.imread(p)
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = k_value
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    step = 256 / k
    value = 0
    for ar in centers:
      for v in ar:
        segmented_image[segmented_image == v] = value
      value += step

    if i == 0:
      cl_img_gr = segmented_image
    elif i == 1:
      cl_img = segmented_image

    i += 1
clustering(path1, path2 , 6)

# img_1 = cv.imread(path1, 2)
# img_2 = cv.imread(path2, 2)

gp1, des1 = surf.detectAndCompute(cl_img_gr, None)
gp2, des2 = surf.detectAndCompute(cl_img, None)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck = False)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x : x.distance)

result = cv.drawMatches(cl_img_gr, gp1, cl_img, gp2, matches[:10], None, flags = 2)

plt.imshow(result)
plt.axis('off')
plt.show()