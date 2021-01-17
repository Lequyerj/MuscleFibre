import numpy as np
import os
from tifffile import imread, imwrite
import cv2
from skimage import morphology
import sys
import shutil
from random import randint

#paramters
directoryin = "mask/"
directoryout = "output/"
directoryboundaries = "contours/"
directoryferet = "feret/"
directoryoriginal = "input/"
minregionsize = 50 #ignore detected fibres smaller than this many pixels
minholesize = 100 #fill in holes smaller than this many pixels
framesize = 10 #ignore detected fibres that come within this many pixels of frames edge
cutoff = 128 #minimum probability cutoff for how certain we require nerual network to be that pixel is muscle fibre (/255)
minconvexity = 0 #chucks bizzarely shaped fibres that are probably erroneous detection, set to 0 to turn off this filter

#delete any existing contents of output directories
folder = directoryout
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
folder = directoryferet
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#go to work
for file in os.listdir(directoryoriginal):
     filename = os.fsdecode(file)
     print(filename)
     manualmask = imread(directoryboundaries+filename[:-4]+'.tif')[0,:,:,:]
     manualmask = cv2.cvtColor(manualmask, cv2.COLOR_BGR2GRAY)
     manualmask = cv2.threshold(manualmask, 254, 255, cv2.THRESH_BINARY)[1]
     manualerase = imread(directoryboundaries+filename[:-4]+'.tif')[0,:,:,:]
     manualerase = cv2.cvtColor(manualerase, cv2.COLOR_BGR2GRAY)
     manualerase = 255-manualerase
     manualerase = cv2.threshold(manualerase, 254, 255, cv2.THRESH_BINARY)[1]
     manualerase = 255-manualerase
     img = imread(directoryin+filename[:-4]+'.tif')
     img = np.maximum(img,manualmask)
     
     img = np.minimum(img,manualerase)
     img = cv2.bitwise_not(img)
     img = cv2.threshold(img, cutoff, 255, cv2.THRESH_BINARY)[1]
     img = img>0
     img = morphology.remove_small_objects(img, min_size=minregionsize)
     img = morphology.remove_small_holes(img, area_threshold=minholesize)
     img = 255*img.astype(np.uint8)
     contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     minRect = [None]*len(contours)
     Rect = [None]*len(contours)
     for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        a,b,w,h = cv2.boundingRect(c)
        Rect[i] = (a,b,a+w,b+h)
     drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
     GTOG = cv2.imread(directoryoriginal+filename)
     if len(GTOG.shape)>2:
         GT = GTOG[:,:,1]
     GT = GT/2
     GT = GT.astype(np.uint8)
     drawing[:,:,0] = GT
     drawing[:,:,1] = GT
     drawing[:,:,2] = GT
     newmask = 255*np.ones(drawing.shape,dtype=np.uint8)
     for i, c in enumerate(contours):
        if Rect[i][0] >= 10 and Rect[i][1] >= 10 and Rect[i][2] <= img.shape[1]-10 and Rect[i][3] <= img.shape[0]-10 and hierarchy[0,i,3] == -1:
            color = (17, 138, 178)
            box = cv2.boxPoints(minRect[i])
            (x, y), (width, height), angle = minRect[i]
            feret = min(width,height)
            area = cv2.contourArea(c)
            (p,q), radius = cv2.minEnclosingCircle(c)
            area2 = cv2.contourArea(cv2.convexHull(contours[i]))
            #convexity = area/area2
            cv2.drawContours(drawing, contours, i, color)
            cv2.drawContours(newmask, contours, i, (0,0,0),-1)
            file = open(directoryout+str(filename[:-4])+".csv", "a")
            file.write(str(area)+","+str(feret)+"\n")    
            box = np.intp(box)
            #cv2.drawContours(drawing, [box], 0, color)
     manualedit = np.zeros((2,GTOG.shape[0],GTOG.shape[1],GTOG.shape[2]),dtype=np.uint8)
     manualedit[0] = drawing
     manualedit[1] = GTOG
     imwrite(directoryboundaries+str(filename)[:-4]+'.tif',manualedit, imagej=True)
     imwrite(directoryin+str(filename)[:-4]+'.tif',newmask[:,:,0])
     drawing[:,:,0] = GT
     drawing[:,:,1] = GT
     drawing[:,:,2] = GT
     for i, c in enumerate(contours):
        if Rect[i][0] >= 10 and Rect[i][1] >= 10 and Rect[i][2] <= img.shape[1]-10 and Rect[i][3] <= img.shape[0]-10 and hierarchy[0,i,3] == -1:
            color = (17, 138, 178)
            box = cv2.boxPoints(minRect[i])
            (x, y), (width, height), angle = minRect[i]
            feret = min(width,height)
            area = cv2.contourArea(c)
            (p,q), radius = cv2.minEnclosingCircle(c)
            area2 = cv2.contourArea(cv2.convexHull(contours[i]))
            #convexity = area/area2
            cv2.drawContours(drawing, contours, i, color)
            file = open(directoryout+str(filename[:-4])+".csv", "a")
            file.write(str(area)+","+str(feret)+"\n")    
            cv2.putText(drawing, str(int(feret)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            box = np.intp(box)
	    #cv2.drawContours(drawing, [box], 0, color)   
     imwrite(directoryferet+str(filename[:-4])+".tif",drawing)
        

 
