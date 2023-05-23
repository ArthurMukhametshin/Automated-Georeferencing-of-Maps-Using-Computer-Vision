# Automated-Georeferencing-of-Maps-Using-Computer-Vision
This repository contains a script for automated georeferencing of geoimages - **Automated_georeferencing_with_LoFTR.py**. There are also 3 scripts in which the possibility of using other algorithms for finding corresponding points was tested:

  **SIFT.py** - script for finding corresponding points on two geoimages using the SIFT algorithm.
  
  **SURF.py** - script for finding corresponding points on two geoimages using the SURF algorithm.
  
  **ORB.py** - script for finding corresponding points on two geoimages using the ORB algorithm.
  
Each of the scripts contains not only a block related to a specific model, but also additional tools to possibly increase the efficiency of searching for the corresponding points.

To implement the automated georeferencing model, one universal program code was written and stored in a PY file - Automated_georeferencing_with_LoFTR.py. Separate blocks of code presented in the file carry out successive stages of processing the source data. Explanations of the actions of the blocks are given in the comments to each of them.

**Importing Software Libraries.**
At the beginning of the script, libraries are imported (os, sys, time, torch, kornia, OpenCV, numpy, gdal, osr, matplotlib, kornia_moons), which will later be used to call the necessary functions. They will be described in more detail in the next paragraph on software libraries and their role in the script.

**Specifying Options.**
This is followed by a block for specifying the main parameters, including: paths to linked and non-linked geoimages, the path to save the resulting linked geoimage, the number of clusters for segmentation (clustering will not be performed if the value is 0) and the type of raster transformation (affine, polynomial second and third degrees, thin-walled spline). These types were chosen due to the fact that they are presented in the parameters of the Warp function, which performs image transformation.

**Function for clustering geoimages.**
Then the clustering() function is written for clustering geoimages, the 3 parameters of which are: paths to the attached and unattached geoimages (from which the list is formed), the number of clusters. The variables cl_img_gr and cl_img (which will contain the segmented images) are declared global for their use outside the function later. Further, in the loop, images are read in turn (the paths of which are sorted out from the created list). The dimension of the resulting numpy arrays is changed, which are then converted to the float32 type. After that, the kmeans function is called, which takes as arguments: arrays of values, number of clusters, criteria, number of iterations, and the place where the initial centers come from. The output values of the function are 3 objects: retval, labels, centers. In this case, we are interested in the last two variables corresponding to the label arrays and cluster centers. They are used for final image segmentation. Moreover, the values of the clusters also lie in the range from 0 to 255, since the values are assigned to different classes by successively adding to 0 the step value equal to the ratio 256/k, where k is the number of clusters (done to obtain a contrast image).

**Selecting the use of clustering and converting the image to a tensor.**
Depending on the entered number of clusters, the next step is selected. If an integer value greater than 0 was specified, then the clustering function is called, otherwise this step is skipped. The load_torch_image() function is written, which translates images into tensors, normalizes them and converts them to the RGB color space.

**Point detection and matching.**
After calling this function, the converted images are converted to grayscale and fed into the dictionary, where each image is encoded with its own key. Next, a dictionary is placed in the previously created LoFTR module. The result of processing is the variable correspondences, where the correspondences between points on two images are written in the form of a dictionary. Further, it is decomposed into two variables by keys keypoints0 and keypoints1.

**Refinement of points to the subpixel level.**
Then the obtained points are refined to the subpixel level, for which the findFundamentalMat() function is called, which takes the following parameters as input: points from the first image, points from the second image, the method for calculating the fundamental matrix (RANSAC was chosen, since the experience of the authors from the work considered earlier [Luft, Schiewe, 2021]), the maximum distance from a point to the epipolar line (connecting point) in pixels, the level of the desired matrix reliability, the maximum number of iterations. The output variables of the function are a matrix and epipolar lines labeled True and False depending on the filtering results (True - a line is drawn between two refined points, False - a line is not drawn between the filtered points).

**Drawing points.**
Further, the obtained points and their connecting lines are displayed on the screen. At this stage, it is possible to conduct an intermediate quality control of points selection. If the result is unsatisfactory, you can change the number of clusters or the parameters of the findFundamentalMat() function.

**Getting pixel coordinates of points.**
Upon reaching an acceptable result, the procedure for extracting the pixel coordinates of the points begins. This is done by checking the presence of a connecting line between two points. If this is confirmed, then they are added to the list_kp1 and list_kp2 lists.

**Getting geographic coordinates of points.**
Then the received pixel coordinates are converted into geographic ones. To do this, using the gdal library, the attached geo-image is read and its transformation parameters are read, from which the coordinates of the upper left corner of the image and the spatial resolution are extracted. Next, geographic coordinates are calculated and added to the list, where reference points are created using the GCP function for binding (the following are specified in sequence: geographic coordinate x, geographic coordinate y, pixel coordinate x, pixel coordinate y). Thus, a one-to-one correspondence is established between the pixel coordinates of the unattached image and the geographic coordinates obtained from the attached image.

**Final geoimage binding.**
Then comes the geoimage binding block. Here, using gdal, the unattached image is read, the parameters of the coordinate system (CS) or the projection of the attached image are obtained, and its EPSG code is read. Further, using the SetGCPs function, where the list of control points and parameters of the GC of the linked raster are specified, the previously unlinked map is bound. After that, using the Warp function, the attached geo-image is transformed in accordance with the previously selected type of transformation.

**Anchor Accuracy Assessment.**
At the last stage, a file with the POINTS extension is generated, which is used in QGIS to store information about anchor points. Column names are: mapX (geographic x coordinate), mapY (geographic y coordinate), pixelX (pixel x coordinate), pixelY (pixel y coordinate), enable (usage status). Each line corresponds to one point.
![image](https://github.com/ArthurMukhametshin/Automated-Georeferencing-of-Maps-Using-Computer-Vision/assets/104223492/078be0cc-cf1a-4782-b4ed-0c22bd5372bf)
After completing the binding procedure, this file can be loaded into the QGIS georeferencing module to estimate the residual value both in general and at individual points, in order to then make a more detailed assessment of the transformation accuracy.

**Examples of the result of the model:**
![image](https://github.com/ArthurMukhametshin/Automated-Georeferencing-of-Maps-Using-Computer-Vision/assets/104223492/6e2149d7-6fc5-4a48-969c-8497ff669a74)
![image](https://github.com/ArthurMukhametshin/Automated-Georeferencing-of-Maps-Using-Computer-Vision/assets/104223492/aeb27b78-438a-4d5f-973f-8c319d8ee756)

![image](https://github.com/ArthurMukhametshin/Automated-Georeferencing-of-Maps-Using-Computer-Vision/assets/104223492/0f749fa8-4113-4fe6-986e-4fccffe475e2)
![image](https://github.com/ArthurMukhametshin/Automated-Georeferencing-of-Maps-Using-Computer-Vision/assets/104223492/9bfdbb9b-2e14-4396-a559-1fb20bf7fa6f)
