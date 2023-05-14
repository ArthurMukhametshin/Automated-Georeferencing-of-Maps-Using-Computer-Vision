import os
import sys
import time
import torch
import kornia
import cv2 as cv
import numpy as np
from osgeo import gdal, osr
import kornia.feature as KF
import matplotlib.pyplot as plt
from kornia_moons.feature import *

# ИСХОДНЫЕ ПАРАМЕТРЫ
path_georef = 'Data/yandex_2gis_moscow/yandex_msc_gr.tif' # Путь к привязанной карте
path_notgeoref = 'Data/yandex_2gis_moscow/2gis_msc.tif' # Путь к непривязанной карте
save_path = 'Data/yandex_2gis_moscow/2gis_msc_gr.tif' # Путь для сохранения привязанной карты
n_clusters = 0 # Число кластеров для сегментации карт (в случае ненадобности ввести значение 0)
tr_type = 'aff' # Тип трансформации: aff, poly2, poly3, spline

# ФУНКЦИЯ ДЛЯ КЛАСТЕРИЗАЦИИ КАРТ
def clustering (path1, path2, k_value):
    paths = [path1, path2]
    global cl_img_gr, cl_img

    i = 0
    for p in paths:
        image = cv.imread(p)
        pixel_values = image.reshape(-1, 3)
        pixel_values = np.float32(pixel_values)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        retval, labels, centers = cv.kmeans(pixel_values, k_value, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)

        step = 256/k_value
        value = 0
        for ar in centers:
            for v in ar:
                segmented_image[segmented_image == v] = value
            value += step

        if i == 0: cl_img_gr = segmented_image
        elif i == 1: cl_img = segmented_image

        i += 1

# ВЫБОР ИСПОЛЬЗОВАНИЯ КЛАСТЕРИЗЦИИ
st = time.time()

if n_clusters > 0:
    clustering(path_georef, path_notgeoref, n_clusters)

    # ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ И ПРЕОБРАЗОВАНИЯ КАРТ В ТЕНЗОРЫ
    def load_torch_image(image):
        img = kornia.image_to_tensor(image, False).float() / 255.
        img = kornia.color.bgr_to_rgb(img)

        return img

    x = cl_img_gr
    y = cl_img

elif n_clusters == 0:

    # ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ И ПРЕОБРАЗОВАНИЯ КАРТ В ТЕНЗОРЫ
    def load_torch_image(image):
        img = kornia.image_to_tensor(cv.imread(image), False).float() / 255.
        img = kornia.color.bgr_to_rgb(img)
        return img

    x = path_georef
    y = path_notgeoref

else:
    print('Введите число кластеров >= 0!')
    sys.exit()

et = time.time()
print ('Время выполнения кластеризации: ', round(et-st, 2), ' с')

# ПРЕОБРАЗОВАНИЕ КАРТ В ТЕНЗОРЫ
st = time.time()

img1 = load_torch_image(x)
img2 = load_torch_image(y)

et = time.time()
print ('Время преобразования карт в тензоры: ', round(et-st, 2), ' с')

# ОБНАРУЖЕНИЕ ТОЧЕК
st = time.time()

matcher = KF.LoFTR(pretrained='outdoor')
input_dict = {'image0': kornia.color.rgb_to_grayscale(img1),
              'image1': kornia.color.rgb_to_grayscale(img2)}

et = time.time()
print ('Время обнаружения точек: ', round(et-st, 2), ' с')

# СОПОСТАВЛЕНИЕ ТОЧЕК
st = time.time()

with torch.inference_mode():
    correspondences = matcher(input_dict)
    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()

et = time.time()
print ('Время сопоставления точек: ', round(et-st, 2), ' с')

# УТОЧНЕНИЕ ТОЧЕК ДО СУБПИКСЕЛЬНОГО УРОВНЯ
st = time.time()

Fm, inliers = cv.findFundamentalMat(mkpts1,
                                    mkpts2,
                                    method=cv.FM_RANSAC,
                                    ransacReprojThreshold=0.1,
                                    confidence=0.999,
                                    maxIters=100000)
inliers = inliers > 0
print('Выделено соответственных точек: ', len(inliers))

et = time.time()
print ('Время уточнения точек до субпиксельного уровня: ', round(et-st, 2), ' с')

# ОТРИСОВКА ТОЧЕК
draw_LAF_matches(
    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts1.shape[0]).view(1, -1, 1)),

    KF.laf_from_center_scale_ori(torch.from_numpy(mkpts2).view(1, -1, 2),
                                 torch.ones(mkpts2.shape[0]).view(1, -1, 1, 1),
                                 torch.ones(mkpts2.shape[0]).view(1, -1, 1)),
    torch.arange(mkpts1.shape[0]).view(-1, 1).repeat(1, 2),
    kornia.tensor_to_image(img1),
    kornia.tensor_to_image(img2),
    inliers,
    draw_dict={'inlier_color': (0.2, 1, 0.2),
               'tentative_color': None,
               'feature_color': (0.2, 0.5, 1), 'vertical': False})

plt.axis('off')
plt.show()

# ПОЛУЧЕНИЕ ПИКСЕЛЬНЫХ КООРДИНАТ ТОЧЕК
st = time.time()

list_kp1 = []
list_kp2 = []

n = 0
for i in inliers:
    if i == True:
        list_kp1.append(mkpts1[n])
        list_kp2.append(mkpts2[n])
    n += 1

et = time.time()
print ('Время получения пиксельных координат точек: ', round(et-st, 2), ' с')

# ПОЛУЧЕНИЕ ГЕОГРАФИЧЕСКИХ КООРДИНАТ ТОЧЕК
st = time.time()

img_gr = gdal.Open(path_georef)
gt = img_gr.GetGeoTransform()
xmin = gt[0]
ymax = gt[3]
res = gt[1]

list_gpc = []
list_coords = []
i = 0
for kp in list_kp1:
    gpc_x = xmin + kp[0] * res
    gpc_y = ymax - kp[1] * res
    list_gpc.append(gdal.GCP(gpc_x, gpc_y, 0, list_kp2[i][0].astype('float64'),  list_kp2[i][1].astype('float64')))
    list_coords.append('{0},{1},{2},{3},1'.format(gpc_x, gpc_y, list_kp2[i][0], -1*list_kp2[i][1]))
    i += 1

et = time.time()
print ('Время получения географических координат точек: ', round(et-st, 2), ' с')

# ОКОНЧАТЕЛЬНАЯ ПРИВЯЗКА КАРТЫ
st = time.time()

img = gdal.Open(path_notgeoref, gdal.GA_Update)
wkt_gr = img_gr.GetProjection()
epsg = osr.SpatialReference(wkt=wkt_gr).GetAttrValue('AUTHORITY', 1)

img.SetGCPs(list_gpc, wkt_gr)

if tr_type == 'aff':
    gdal.Warp(save_path, img, options=gdal.WarpOptions(dstSRS='EPSG:{0}'.format(str(epsg)),
                                                       format='gtiff',
                                                       errorThreshold=0,
                                                       polynomialOrder=1))
elif tr_type == 'poly2':
    gdal.Warp(save_path, img, options=gdal.WarpOptions(dstSRS='EPSG:{0}'.format(str(epsg)),
                                                       format='gtiff',
                                                       errorThreshold=0,
                                                       polynomialOrder=2))
elif tr_type == 'poly3':
    gdal.Warp(save_path, img, options=gdal.WarpOptions(dstSRS='EPSG:{0}'.format(str(epsg)),
                                                       format='gtiff',
                                                       errorThreshold=0,
                                                       polynomialOrder=3))
elif tr_type == 'spline':
    gdal.Warp(save_path, img, options=gdal.WarpOptions(dstSRS='EPSG:{0}'.format(str(epsg)),
                                                       format='gtiff',
                                                       errorThreshold=0,
                                                       tps=True))

et = time.time()
print ('Время привязки карты: ', round(et-st, 2), ' с')

# ОЦЕНКА ТОЧНОСТИ ПРИВЯЗКИ
path_gcp = os.path.dirname(save_path) + '/' + 'GCP.points'

with open(path_gcp, 'w', encoding='ANSI') as f:
    f.write('mapX,mapY,pixelX,pixelY,enable' + '\n')
    for coords in list_coords:
        f.write(coords + '\n')
