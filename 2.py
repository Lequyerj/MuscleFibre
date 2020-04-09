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
directoryboundaries = "node/"
directoryoriginal = "input/"
minregionsize = 50 #ignore detected fibres smaller than this many pixels
framesize = 10 #ignore detected fibres that come within this many pixels of frames edge
cutoff = 241 #minimum probability cutoff for how certain we require nerual network to be that pixel is muscle fibre (/255)

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
folder = directoryboundaries
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
for file in os.listdir(directoryin):
     filename = os.fsdecode(file)
     print(filename)
     img = imread(directoryin+filename)
     img = cv2.bitwise_not(img)
     img = cv2.threshold(img, cutoff, 255, cv2.THRESH_BINARY)[1]
     img = img>0
     img = morphology.remove_small_objects(img, min_size=minregionsize)
     img = morphology.remove_small_holes(img, min_size=minregionsize)
     img = 255*img.astype(np.uint8)
     contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     minRect = [None]*len(contours)
     Rect = [None]*len(contours)
     for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        a,b,w,h = cv2.boundingRect(c)
        Rect[i] = (a,b,a+w,b+h)
     drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
     GT = imread(directoryoriginal+filename[0]+'.tif')
     GT = GT/2
     GT = GT.astype(np.uint8)
     drawing[:,:,0] = GT
     drawing[:,:,1] = GT
     drawing[:,:,2] = GT
     for i, c in enumerate(contours):
        if Rect[i][0] >= 10 and Rect[i][1] >= 10 and Rect[i][2] <= img.shape[1]-10 and Rect[i][3] <= img.shape[0]-10 and hierarchy[0,i,3] == -1:
            color = (128+randint(0,128), 128+randint(0,128), 128+randint(0,128))
            box = cv2.boxPoints(minRect[i])
            (x, y), (width, height), angle = minRect[i]
            feret = min(width,height)
            area = cv2.contourArea(c)
            (p,q), radius = cv2.minEnclosingCircle(c)
            cv2.drawContours(drawing, contours, i, color)
            file = open(directoryout+str(filename[0])+".csv", "a")
            file.write(str(area)+","+str(feret)+"\n")    
            #cv2.putText(drawing, str('%.2f' % feret), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            box = np.intp(box)
            #cv2.drawContours(drawing, [box], 0, color)              
     imwrite(directoryboundaries+str(filename[0])+"_boundaries",drawing)     
        

 