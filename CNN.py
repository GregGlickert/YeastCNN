from PIL import Image
from plantcv import plantcv as pcv
import cv2
import numpy as np
import os.path,os
import shutil
import glob
import scipy
from openpyxl import load_workbook
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import OPTICS
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False,
                help="path to input directory of images")
args = vars(ap.parse_args())
imagePath = sorted(list(paths.list_images(args["images"])))
counter1 = 0
featurelist = []
counter3 = 0

#for i in range(len(imagePath)):
for i in range(0, 1):
    img = Image.open(imagePath[i])
    #img.show()
    counter1 = (i*96)
    counter3 = (i*384)
    base = os.path.basename(imagePath[i])

    def initcrop(img):  # change line 70 for threshold adj
        left = 1875  # was 2050
        top = 730  # was 870
        right = 5680
        bottom = 3260  # was 3280
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        try:
            os.makedirs(path)
        except OSError:
            pass
        img_crop = img.crop((left, top, right, bottom))
        # img_crop.show()
        img_crop.save(os.path.join(path, 'Cropped_full_yeast.png'))
        cropped_img = cv2.imread(
            os.path.join(path, "Cropped_full_yeast.png"))  # changed from Yeast_Cluster.%d.png  %counter
        blue_image = pcv.rgb2gray_lab(cropped_img, 'b')  # can do l a or b
        gaussian_blue = cv2.adaptiveThreshold(blue_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 241,
                                              -1)  # For liz's pictures 241
        cv2.imwrite(os.path.join(path, "blue_test.png"), gaussian_blue)
        blur_image = pcv.median_blur(gaussian_blue, 10)
        heavy_fill_blue = pcv.fill(blur_image, 400)  # value 400
        cv2.imwrite(os.path.join(path, "Cropped_Threshold.png"), heavy_fill_blue)


    def cluster_maker(counter1,counter3):
        dire = os.getcwd()
        path = dire + '/Classifyer_dump'
        path1 = dire + '/Yeast_cluster_inv'
        path2 = dire + '/Yeast_cluster'
        path3 = dire + '/Cells'
        print(path1)
        try:
            os.makedirs(path3)
        except OSError:
            pass
        counter = 0
        im = Image.open(os.path.join(path, "Cropped_Threshold.png"))  # was "Cropped_full_yeast.png"
        sizeX, sizeY = im.size
        im_sizeX = round(sizeX / 12)
        im_sizeY = round(sizeY / 8)
        for h in range(0, im.height, im_sizeY):
            nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
            nim.save(os.path.join(path, "Yeast_Row." + str(counter) + ".png"))
            counter += 1
        anotherCounter = 0
        for i in range(0, 8):
            columnImage = (os.path.join(path, "Yeast_Row.%d.png" % anotherCounter))
            Each_Image = Image.open(columnImage)
            sizeX2, sizeY2 = Each_Image.size
            Each_Image_sizeX = round(sizeX2 / 12)
            Each_Image_sizeY = round(sizeY2 / 8)
            anotherCounter += 1
            widthCounter1 = 0
            widthCounter2 = Each_Image_sizeX
            for w in range(0, 12):
                Wim = Each_Image.crop(
                    (widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
                Wim.save(os.path.join(path1, "Yeast_Cluster." + str(counter1) + ".png"))
                counter1 += 1
                widthCounter1 = widthCounter1 + Each_Image_sizeX
                widthCounter2 = widthCounter2 + Each_Image_sizeX
                print(counter1)
        row_counter_for_save = 0
        row_counter_for_open = 0
        for i in range(0, 96):
            im = Image.open(os.path.join(path2, "Yeast_Cluster.%d.png" % i))
            sizeX, sizeY = im.size
            im_sizeX = round(sizeX / 2)
            im_sizeY = round(sizeY / 2)
            for h in range(0, im.height, im_sizeY):
                nim = im.crop((0, h, im.width - 1, min(im.height, h + im_sizeY) - 1))
                nim.save(os.path.join(path, "ROW_SMALL." + str(row_counter_for_save) + ".png"))
                row_counter_for_save += 1
                if (h >= im_sizeY):
                    break
            for i in range(0, 2):
                rowImage = (os.path.join(path, "ROW_SMALL.%d.png" % row_counter_for_open))
                Each_Image = Image.open(rowImage)
                sizeX2, sizeY2 = Each_Image.size
                Each_Image_sizeX = round(sizeX2 / 2)
                Each_Image_sizeY = round(sizeY2 / 2)
                row_counter_for_open += 1
                widthCounter1 = 0
                widthCounter2 = Each_Image_sizeX
                for w in range(0, 2):
                    Wim = Each_Image.crop(
                        (widthCounter1, w, widthCounter2, min(Each_Image.height, w + Each_Image_sizeX) - 1))
                    Wim.save(os.path.join(path3, "SMALL_CELL." + str(counter3) + ".png"))
                    counter3 += 1
                    widthCounter1 = widthCounter1 + Each_Image_sizeX
                    widthCounter2 = widthCounter2 + Each_Image_sizeX


    def connected_comps_liz(counter):
        dire = os.getcwd()
        path = dire + '/Yeast_cluster_inv'
        cropped_img = cv2.imread(os.path.join(path, 'Yeast_Cluster.%d.png' % counter),
                                 cv2.IMREAD_UNCHANGED)  # changed from Yeast_Cluster.%d.png  %counter
        circle_me = cv2.imread(os.path.join(path, "Yeast_Cluster.%d.png" % counter))

        connected_counter = 0

        connectivity = 8

        connectivity = 8  # either 4 or 8
        output = cv2.connectedComponentsWithStats(cropped_img, connectivity)  # Will determine the size of our clusters

        num_labels = output[0]
        labels = output[1]
        stats = output[2]  # for size
        centroids = output[3]  # for location
        area_array = []

        print("Currently on cell %d" % counter)
        cc_size_array = []
        print(centroids)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                print("%d is in 1" % i)
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 40 and centroids[i][1] <= 110):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 2" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 40 and centroids[i][0] <= 110 and centroids[i][1] >= 200 and centroids[i][1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 3" % i)
        for i in range(0, (len(stats)), 1):
            if (centroids[i][0] >= 200 and centroids[i][0] <= 270 and centroids[i][1] >= 200 and centroids[i][
                1] <= 270):
                cc_size_array.append(stats[i, cv2.CC_STAT_AREA])
                print("%d is in 4" % i)

        if (len(stats) < 4):
            print("too few decteted on %d" % counter)
            print((len(stats)))
            for i in range((len(stats)), 5, 1):
                cc_size_array.append(0)

        if (len(cc_size_array) >= 5):
            print("problem on cell %d" % counter)
            exit(-1)

        # total_size_array = total_size_array + cc_size_array
        print("size data")
        print(cc_size_array)
        avg_size = np.average(cc_size_array)
        print(avg_size)
        std = np.std(np.array(cc_size_array))
        print(std)
        Zscore_array = abs(scipy.stats.zscore(cc_size_array))
        print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8: # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(cc_size_array)):
            if cc_size_array[i] <= 3500:  # for liz she wants small?
                above_size_ther += 1

        temp = ((10*above_size_ther)-(mod * Z_avg))  # simply alg to tell is positive gets normalized later
        print(temp)


        print("end of size data")



        return cc_size_array, avg_size, std, Zscore_array, Z_avg, above_size_ther, mod, temp


    def image_colorfulness(image):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype("float"))
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)


    def colorful_writer(color_counter):
        dire = os.getcwd()
        path = dire + '/Cells'
        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        color_array = []
        for i in range(0, 4):
            print("THIS IS COLOR COUNTER")
            print(color_counter)
            image = cv2.imread(os.path.join(path, "SMALL_CELL.%d.png" % color_counter))
            C = image_colorfulness(image)
            # display the colorfulness score on the image
            color_array.append(C)
            #cv2.putText(image, "{:.2f}".format(C), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            #cv2.imwrite(os.path.join(path, "SMALL_CELL.%d.png" % color_counter), image)
            color_counter = color_counter + 1

        # total_color_array = total_color_array + color_array
        avg_color = (color_array[0] + color_array[1] + color_array[2] + color_array[3]) / 4
        std_color = np.std(np.array(color_array))

        if (len(color_array) < 4):
            print((len(color_array)))
            for i in range((len(color_array)), 5, 1):
                color_array.append(0)

        Zscore_array = abs(scipy.stats.zscore(color_array))
        print(Zscore_array)
        Z_avg = np.average(Zscore_array)
        print(Z_avg)
        mod = 1.5  # if avg zscore is less than .5 or 40% that is rewarded
        if Z_avg >= .8:  # completely random number lol
            mod = 1
        above_size_ther = 0
        for i in range(0, len(color_array)):
            if color_array[i] <= 23: #pretty white ones???
                above_size_ther += 1
        temp = ((10*above_size_ther) - (mod * Z_avg))


        #print(color_array)
        return color_array, avg_color, std_color, color_counter, Zscore_array, Z_avg, above_size_ther, mod, temp


    #initcrop(img)
    #cluster_maker(counter1, counter3)
    featurelist = []
    color_counter = 0
    for i in range(0, 1056):
        returned_array = connected_comps_liz(i)
        size = returned_array[0]
        avg_size = returned_array[1]
        std = returned_array[2]
        Zscore = returned_array[3]
        Z_avg = returned_array[4]
        test = np.append([avg_size], [std])
        #test = np.append([test],[Z_avg])
        #features = np.concatenate([size, Zscore, test])
        returned_color = colorful_writer(color_counter)
        color_counter = returned_color[3]
        color = returned_color[0]
        Zcolor = returned_color[4]
        avg_color = returned_color[1]
        std_color = returned_color[2]
        Z_color_avg = returned_color[5]
        color_stuff = np.append([avg_color], [std_color])
        #color_stuff = np.append([color_stuff], [Z_color_avg])
        features = np.concatenate([test,color_stuff])
        #print("features\n")
        #print(features)
        featurelist.append(features)
        print(featurelist)

#print(np.array(featurelist))




dire = os.getcwd()
imdir = dire + '/Yeast_cluster'
targetdir = dire + '/kmeans'
number_clusters = 5
"""
# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())
#print("feature list np")
#print(np.array(featurelist))
#print(len(featurelist))

"""

"""
# Clustering
#kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 60)
for k in K:
    X = np.array(featurelist)
    # Building and fitting the model
    kmeanModel = MiniBatchKMeans(n_clusters=k).fit(np.array(featurelist))
    kmeanModel.fit(np.array(featurelist))

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key,val in mapping1.items():
    print(str(key)+' : '+str(val))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key,val in mapping2.items():
    print(str(key)+' : '+str(val))

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()
"""



dire = os.getcwd()
imdir = dire + '/Yeast_cluster'
targetdir = dire + '/kmeans'
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
kmeans = MiniBatchKMeans(n_clusters=24, random_state=0).fit(np.array(featurelist))
#optics = OPTICS(cluster_method='dbscan', eps=2).fit((np.array(featurelist)))


# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_): #changed from kmeans to dbscan
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r") #same here
    print("KMEANS")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
    shutil.move(targetdir + str(m) + "_" + str(i) + ".jpg", targetdir)

#https://stackoverflow.com/questions/39123421/image-clustering-by-its-similarity-in-python

