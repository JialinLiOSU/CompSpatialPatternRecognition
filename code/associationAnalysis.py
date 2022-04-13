import pickle
import os
import numpy as np
import json
from PIL import Image,ImageDraw
import sys

import cv2
import matplotlib.pyplot as plt  
import matplotlib.patches as patches

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.measure import find_contours

from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
from random import sample

import math
import random
from sklearn.neighbors import KDTree
import alphashape
import statistics
from scipy.stats import ranksums
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def bgr2rgb(value):
    return value[2],value[0],value[1]
def valueEqualColor(value,color):
    colorR, colorG, colorB = color[0], color[1], color[2]
    valueR, valueG, valueB = value[0], value[1], value[2]
    if abs(colorR - valueR) > 10 or abs(colorB - valueB) > 10 or abs(colorG - valueG) > 10:
        return False
    else:
        return True
def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey
def distance(coord1, coord2):
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

def silKmeans(kmax, x):
    sil = []
    difSilList = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax):
    #     print('K: '+str(k))
        kmeans = KMeans(n_clusters = k,random_state=0).fit(x)
        labels = kmeans.labels_
        silScore = silhouette_score(x, labels, metric = 'euclidean')
        difSil = 0
        if len(sil) > 0:
            difSil = silScore - sil[-1]
        sil.append(silScore)
        difSilList.append(difSil)

    # find the best index of difSil
    index = 0
    maxValue = max(sil)
    maxIndex = sil.index(maxValue)
    maxDif = max(difSilList)
    while maxIndex >= 3:
        maxValue = sil[maxIndex]
        i = maxIndex - 1
        if difSilList[maxIndex] < maxDif / 10:
            maxIndex = maxIndex - 1
        else:
            break
    numClusters = maxIndex + 2
    kmeansResults = KMeans(n_clusters = numClusters).fit(x)
    return kmeansResults

def moransi(z, w):
    """
    Input
      z: list of values
      w: weight matrix
    Output
      I: Moran's I measure
    """
    n = len(z)
    mean = float(sum(z))/n
    S0 = 0
    d = 0
    var = 0
    for i in range(n):
        var += (z[i]-mean)**2
        for j in range(i):
            if w[i][j]:
                S0 += w[i][j]
                d += w[i][j] * (z[i]-mean) * (z[j]-mean)
    I = n*d/S0/var
    return I

def findDetectResult(detectResults, imageName):
    for dr in detectResults:
        name = dr[0]
        if name == imageName:
            return dr
    return None
# 1/(distance(pixelCoordList_sample[i], pixelCoordList_sample[j]) + 0.0000001)
def calculateKNeighWeight(i, coordList, k = 6): 
    # i is the currently target, k includes the point itself
    # k nearest neighborhood, only consider the k nearest neighbors
    # based on distance to calcuate weight, and normalize weight
    distList = []
    for j in range(len(coordList)):
        dist = distance(coordList[i], coordList[j])
        distList.append(dist)
    
    neighIndice = sorted(range(len(distList)), key = lambda sub: distList[sub])[:k]
    distNeigh = [distList[ind] for ind in neighIndice]

    neighIndice = neighIndice[1:]
    distNeigh = distNeigh[1:]
    wNeigh = [1/(distNeigh[j]+ 0.00000001) for j in range(len(distNeigh))]
    
    # normalize weights
    wNeighNorm = (np.asarray(wNeigh)/sum(wNeigh)).tolist()
    
    # put weights together
    wList = [0 for j in range(len(coordList))]
    for ind, j in enumerate(neighIndice):
        wList[j] = wNeighNorm[ind]
        
    return wList

def calculateKNeigh(targetCoord, coordList, k = 6): 
    # targetCoord is the coordinate of current point
    # k nearest neighborhood, only consider the k nearest neighbors
    # CoordList is the set of coordinates to compare
    distList = []
    for j in range(len(coordList)):
        dist = distance(targetCoord, coordList[j])
        distList.append(dist)
    neighIndice = sorted(range(len(distList)), key = lambda sub: distList[sub])[:k]
    return neighIndice

def most_common(lst):
    return max(set(lst), key=lst.count)

def mostCommonListHLine(y,xMin, xMax, pixelCoordList_sample,zList):
    mostCommonList = []
    for x in range(xMin, xMax + 1):
        coord = (x,y)
        neighIndice = calculateKNeigh(coord, pixelCoordList_sample)
        # find the most common class
        zListNeighIndice = [zList[ind] for ind in neighIndice]
        mostCommon = most_common(zListNeighIndice)
        mostCommonList.append(mostCommon)
    return mostCommonList

def mostCommonListVLine(x,yMin, yMax, pixelCoordList_sample,zList):
    mostCommonList = []
    for y in range(yMin, yMax + 1):
        coord = (x,y)
        neighIndice = calculateKNeigh(coord, pixelCoordList_sample)
        # find the most common class
        zListNeighIndice = [zList[ind] for ind in neighIndice]
        mostCommon = most_common(zListNeighIndice)
        mostCommonList.append(mostCommon)
    return mostCommonList

def countSegmentationFun(mostCommonList):
    countSegmentation = 1
    currentValue = mostCommonList[0]
    countDifferentTemp = 0
    for i in range(len(mostCommonList)):
        if mostCommonList[i] ==  currentValue:
            countDifferentTemp = 0
            continue
        else:
            countDifferentTemp += 1
            if countDifferentTemp > 20:
                currentValue = mostCommonList[i]
                countSegmentation += 1
    return countSegmentation

def main():
    # read detection results from pickle file
    detectResultsPath = r'D:\OneDrive - The Ohio State University\choroColorRead'
    detectResultFileName = 'detectResultSpatialPattern.pickle'
    # detection from Mask R-CNN
    with open(detectResultsPath + '\\' + detectResultFileName, 'rb') as f:
        detectResults = pickle.load(f)

    # recognized colors
    with open(r'C:\Users\jiali\Desktop\choroColorRead\colorRecognitionEvaluation' + '\\' + 'colorsOrderedImagesRemoveGrey3.pickle', 'rb') as f:
        colorsOrderedImages = pickle.load(f)

    imagePath = r'C:\Users\jiali\Desktop\choroColorRead\generatedMaps\classifiedQuantiles\pos_small'
    # imageName = 'ohio_Blues_4_neg1.jpg'
    testImages = os.listdir(imagePath)
    for imageName in testImages:
        print('imageName: ' + imageName)
        detectResult = findDetectResult(detectResults, imageName)
        property = detectResult[1]
        boxes = property['rois']
        masks = property['masks']
        class_ids = property['class_ids']

        # extract mask for mapping area
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        image = cv2.imread(imagePath + '\\' + imageName)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height = image.shape[0]
        width = image.shape[1]

        # get the polygon for mapping area
        for i in range(N):
            if class_ids[i] != 3:
                continue
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            # Mask
            mask = masks[:, :, i]
            # print('shape of mask: ')
            # print(mask.shape)

        # rgb color list
        # colorList = [[191, 214, 230], [107, 174, 216], [33, 114, 180], [239, 243, 255]]
        # colorList = spColorExtraction[imageName][0] # 0: detected colors, 1: missed colors
        colorList = colorsOrderedImages[imageName]

        # traverse the whole image
        # colorSumRGBList = [sum(color) for color in colorList]
        colorSumRGBList = [i for i in range(len(colorList))]
        colorGreyList = [rgb2Grey(color) for color in colorList]

        indexMinGreyColor = colorGreyList.index(min(colorGreyList))
        # colorGrey0 = rgb2Grey(colorList[0])
        # colorGrey1 = rgb2Grey(colorList[1])
        # colorGrey2 = rgb2Grey(colorList[2])
        # colorGrey3 = rgb2Grey(colorList[3])
        pixelCoordsList = [[] for i in range(len(colorGreyList)) ] # a set of coords for each color class
        pixelCoordList0, pixelCoordList1, pixelCoordList2, pixelCoordList3 = [],[],[],[]
        for i in range(height):
            # if i % 100 == 0:
            #     print('i: ' + str(i))   
            for j in range(mask.shape[1]):
                # point = Point(j,i)
                if mask[i,j] == True:
                    value =  imageGray[i,j]
                    b, g, r = image[i,j]
                    # if not (abs(b - g) < 10 and abs(b-r) < 10 and abs(g - r)<10):
                    if not (abs(int(b) - int(g)) < 10 and abs(int(b)-int(r)) < 10 and abs(int(g) - int(r))<10):
                        # if j == 879 and i == 105:
                        #     print('debug')
                        for k in range(len(colorList)):
                            colorR, colorG, colorB = colorList[k]
                            if (abs(int(b) - colorB) < 10 and abs(int(g) - colorG) < 10 and abs(int(r) - colorR)<10):
                                pixelCoordsList[k].append((j,i))
                                break  
        pixelCoordListSampleList = [[] for i in range(len(pixelCoordsList))]
        pixelCoordListSampleAll = []
        for i, pixelCoords in enumerate(pixelCoordsList):
            pixelCoordListSampleList[i] = sample(pixelCoordsList[i],int(len(pixelCoordsList[i])/100))
            pixelCoordListSampleAll += pixelCoordListSampleList[i]

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleList[0]]
        # yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleList[0]]
        # ax.scatter(xList, yList, s = 5,c = np.asarray(colorList[0])/255.0)

        # xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleList[1]]
        # yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleList[1]]
        # ax.scatter(xList, yList, s = 5,c = np.asarray(colorList[1])/255.0)

        # xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleList[2]]
        # yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleList[2]]
        # ax.scatter(xList, yList, s = 5,c = np.asarray(colorList[2])/255.0)

        # xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleList[3]]
        # yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleList[3]]
        # ax.scatter(xList, yList, s = 5,c = np.asarray(colorList[3])/255.0)
        # plt.show()

        # clustering
        kmax = 100
        silKmeansResultList = []
        zListList = []
        zListAll = []
        for i, pixelCoordsSample in enumerate(pixelCoordListSampleList):
            kmax = min(100,len(pixelCoordsSample))
            silKmeansResult = silKmeans(kmax, pixelCoordsSample)
            silKmeansResultList.append(silKmeansResult)
            zList = [colorSumRGBList[i] for j in range(len(pixelCoordListSampleList[i]))]
            zListList.append(zList)
            zListAll += zList 

        # clustering
        # kmax = 100
        # # kmeans0 = silKmeans(kmax, pixelCoordList0_sample)
        # # kmeans1 = silKmeans(kmax, pixelCoordList1_sample)
        # kmeansHighestClass = silKmeans(kmax, pixelCoordListSampleList[indexMinGreyColor])
        # # kmeans3 = silKmeans(kmax, pixelCoordList3_sample)
        zCentersList = []
        zCentersAll = []
        coordCentersList = []
        coordCentersAll = []
        needLineGrid = False
        for i,silKmeansResult in enumerate(silKmeansResultList):
            # print('debug')
            zCentersTemp = [colorSumRGBList[i] for j in range(silKmeansResult.cluster_centers_.shape[0])]
            zCentersList.append(zCentersTemp)
            if len(zCentersTemp) <= 10:
                needLineGrid = True
            zCentersAll += zCentersTemp
            coordCenters = silKmeansResult.cluster_centers_.tolist()
            coordCentersList.append(coordCenters)
            coordCentersAll += coordCenters
        
        
        ### any number of cluster centers not larger than 10, conduct grid line algorithm
        if needLineGrid:
            xList = [pixelCoord[0] for pixelCoord in pixelCoordListSampleAll]
            yList = [pixelCoord[1] for pixelCoord in pixelCoordListSampleAll]
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            yMid = int((yMin + yMax) / 2)

            mostCommonListLines = []
            for y in range(yMin, yMax + 1, int((yMax + 1 - yMin) / 10)):
                # print(y)
                mostCommonList = mostCommonListHLine(y, xMin, xMax, pixelCoordListSampleAll,zListAll)
                mostCommonListLines.append(mostCommonList)
            
            for x in range(xMin, xMax + 1, int((xMax + 1 - xMin) / 10)):
                # print(x)
                mostCommonList = mostCommonListVLine(x, yMin, yMax, pixelCoordListSampleAll,zListAll)
                mostCommonListLines.append(mostCommonList)


            countSegmentationList = []
            for mostCommonList in mostCommonListLines:
                countSegmentation = countSegmentationFun(mostCommonList)
                countSegmentationList.append(countSegmentation)

            countSegmentation = max(countSegmentationList)
            totalNumCluster = countSegmentation**2
            numClusterEach = math.ceil(totalNumCluster / len(colorGreyList))
            print('new numClusterEach: '+str(numClusterEach))

            kmeansResultList = []
            coordCentersList = []
            for i, pixelCoordsSample in enumerate(pixelCoordListSampleList):

                if len(pixelCoordListSampleList[i]) < numClusterEach:
                    continue
                else:
                    kmeansResult = KMeans(n_clusters = numClusterEach).fit(pixelCoordListSampleList[i])
                kmeansResultList.append(kmeansResult)
                zList = [colorSumRGBList[i] for j in range(len(pixelCoordListSampleList[i]))]
                zListList.append(zList)
                zListAll += zList 

            # kmeansHighestClass = KMeans(n_clusters = numClusterEach).fit(pixelCoordListSampleList[indexMinGreyColor])
            zCentersList = []
            zCentersAll = []
            coordCentersList = []
            coordCentersAll = []
            for i,kmeansResult in enumerate(kmeansResultList):
                # print('debug')
                zCentersTemp = [colorSumRGBList[i] for j in range(kmeansResult.cluster_centers_.shape[0])]
                zCentersList.append(zCentersTemp)
                
                zCentersAll += zCentersTemp
                coordCenters = kmeansResult.cluster_centers_.tolist()
                coordCentersList.append(coordCenters)
                coordCentersAll += coordCenters

        ### Calculate moran's I for cluster centers
        numCenters = len(zCentersAll)
        wMatrix = np.zeros(shape=(numCenters, numCenters))
        for i in range(numCenters):
            wMatrix[i] = calculateKNeighWeight(i, coordCentersAll)
            # print(wMatrix[i])
        w = wMatrix.tolist()
        moranIClusterCenters = moransi(zCentersAll, w) # use line grid to decide number of clusters
        print('moranIClusterCenters:' + str(moranIClusterCenters))


        ### Calculate Moran's I distribution using Monte Carlo Simulation on attribute value of each cluster center
        zMonteCarlo = zCentersAll
        numMonteCarlo = 10000
        moranIList = []
        for i in range(numMonteCarlo):
            random.shuffle(zMonteCarlo)
            moranIClusterCentersMont = moransi(zMonteCarlo, w) # use line grid to decide number of clusters
            moranIList.append(moranIClusterCentersMont)
        moranIMontArray = np.asarray(moranIList)
        moranIMontArrayPercentage25 = np.percentile(moranIMontArray, 2.5)
        moranIMontArrayPercentage975 = np.percentile(moranIMontArray, 97.5)
        moranIMontArrayPercentage5 = np.percentile(moranIMontArray, 5)
        moranIMontArrayPercentage95 = np.percentile(moranIMontArray, 95)
        thresholdLower = moranIMontArrayPercentage25
        thresholdUpper = moranIMontArrayPercentage975
        print('thresholdLower25: ' + str(thresholdLower))
        print('thresholdUpper25: ' + str(thresholdUpper))

        print('thresholdLower5: ' + str(moranIMontArrayPercentage5))
        print('thresholdUpper5: ' + str(moranIMontArrayPercentage95))

        if moranIClusterCenters < thresholdLower:
            print('Negative spatial autocorrelation')
        elif moranIClusterCenters > thresholdUpper:
            print('Positive spatial autocorrelation')
        else:
            print('No clear spatial autocorrelation')



        ### calculate theoretical expected value and variance of Moran's I
        # expected value
        # N = numCenters
        # print(N)
        # expI = -(1/(N - 1))
        # print('expI: ' + str(expI))

        # # variance
        # s0 = sum([sum(wRow) for wRow in w])
        # s1 = 0
        # for i in range(len(w)):
        #     for j in range(len(w[0])):
        #         s1 += (w[i][j] + w[j][i])**2
        # s1 = s1 /2
        # s2 = 0
        # for k in range(N):
        #     s2kj = 0
        #     s2ik = 0
        #     for j in range(N):
        #         s2kj += w[k][j]
        #     for i in range(N):
        #         s2ik += w[i][k]
        #     s2 += (s2kj + s2ik)**2
        # varI = ((N**2)*s1 - N*s2 + 3*(s0**2))/((N**2 -1)*(s0**2)) - expI**2
        # sigma = varI**(0.5)
        # thresholdLower = expI - 2 * sigma
        # print('thresholdLower: ' + str(thresholdLower))
        # thresholdUpper = expI + 2 * sigma
        # print('thresholdUpper: ' + str(thresholdUpper))

        # if moranIClusterCenters < thresholdLower:
        #     print('Negative spatial autocorrelation')
        # elif moranIClusterCenters > thresholdUpper:
        #     print('Positive spatial autocorrelation')
        # else:
        #     print('No clear spatial autocorrelation')



    

if __name__ == "__main__":    main()