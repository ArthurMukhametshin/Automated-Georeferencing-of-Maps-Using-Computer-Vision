import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path1 = 'Data/yandex_2gis_moscow/yandex_msc_gr.tif'
path2 = 'Data/yandex_2gis_moscow/2gis_msc.tif'

img_1 = cv.imread(path1, 2)
img_2 = cv.imread(path2, 2)

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

# clustering(path1, path2 , 6)

sift = cv.xfeatures2d.SIFT_create()
gp1, des1 = sift.detectAndCompute(img_1, None)
gp2, des2 = sift.detectAndCompute(img_2, None)
FLAN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLAN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch (des1, des2, k=2)

good_matches = []
for m1, m2 in matches:
  if m1.distance < m2.distance:
    good_matches.append([m1])

flann_matches = cv.drawMatchesKnn(img_1, gp1, img_2, gp2, good_matches, None, flags=2)

plt.imshow(flann_matches)
plt.axis('off')
plt.show()