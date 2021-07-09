import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from tifffile import imread, imwrite
import sys
import torch.utils.data as utils_data
import numpy as np
import cv2
from skimage import morphology
import shutil
from random import randint


def clearfolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

clearfolder("mask/")
clearfolder("output/")
clearfolder("contours/")
clearfolder("feret/")


class TwoCon(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = TwoCon(in_channels,out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,out_channels,2,2)
        self.conv = TwoCon(in_channels,out_channels)

    def forward(self, x, y):
        x = self.upconv(x)
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = Down(64, 128)
        self.conv3 = Down(128, 256)
        self.conv4 = Down(256, 512)
        self.conv5 = Down(512, 1024)   
        self.upconv1 = Up(1024, 512)
        self.upconv2 = Up(512, 256)
        self.upconv3 = Up(256, 128)
        self.upconv4 = Up(128, 64)
        self.conv6 = nn.Conv2d(64,1,1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv5(x4)
        x = self.upconv1(x,x4)
        x = self.upconv2(x,x3)
        x = self.upconv3(x,x2)
        x = self.upconv4(x,x1)
        x = torch.sigmoid(self.conv6(x))
        return x
class MyTestSet(utils_data.Dataset):
  def __init__(self, transform=None):
      self.mask_file_list = [f for f in os.listdir('input/')]
      
  def __len__(self):
      return len(self.mask_file_list)
  
  def __getitem__(self, index):
      
      file_name =  self.mask_file_list[index]
      img = cv2.imread('input/'+file_name)
      img = img[:,:,1]
      img -= np.amin(img)
      img = img/np.amax(img)
      img = img.astype(np.float32)
      shape = img.shape
      xpixoffset = (16-shape[0]%16)%16
      ypixoffset = (16-shape[1]%16)%16
      result = np.ones((shape[0]+xpixoffset,shape[1]+ypixoffset),dtype=np.float32)
      result[:shape[0],:shape[1]] = img
      img = result
      img = torch.from_numpy(img)
      img = torch.unsqueeze(img,0)
      img = torch.unsqueeze(img,0)
      return img, file_name, shape
#hey = MyTestSet()

      
testloader=MyTestSet()
net = Net()
net.eval()
net.load_state_dict(torch.load('weights.pth',map_location=torch.device('cpu')))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        image, file_name, shipshape = data[0].to(device), data[1], data[2]
        print(file_name)
        image = net(image)
        #image = torch.cat((torch.cat((image[0,0,:,:],image[1,0,:,:]),dim=1),torch.cat((image[2,0,:,:],image[3,0,:,:]),dim=1)),dim=0)
        out = image.cpu().detach().numpy()
        out = 255*out[:,:,:shipshape[0],:shipshape[1]]
        out = out.astype(np.uint8)
        imwrite('mask/'+file_name[:-4]+'.tif', out, imagej=True)



#parameters for contour detection
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
     img = imread(directoryin+filename[:-4]+'.tif')
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
            color = (128+randint(0,128), 128+randint(0,128), 128+randint(0,128))
            box = cv2.boxPoints(minRect[i])
            (x, y), (width, height), angle = minRect[i]
            feret = min(width,height)
            area = cv2.contourArea(c)
            (p,q), radius = cv2.minEnclosingCircle(c)
            area2 = cv2.contourArea(cv2.convexHull(contours[i]))
            convexity = area/area2
            if convexity >= minconvexity:
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
            color = (128+randint(0,128), 128+randint(0,128), 128+randint(0,128))
            box = cv2.boxPoints(minRect[i])
            (x, y), (width, height), angle = minRect[i]
            feret = min(width,height)
            area = cv2.contourArea(c)
            (p,q), radius = cv2.minEnclosingCircle(c)
            area2 = cv2.contourArea(cv2.convexHull(contours[i]))
            convexity = area/area2
            if convexity >= minconvexity:
                cv2.drawContours(drawing, contours, i, color)
                file = open(directoryout+str(filename[:-4])+".csv", "a")
                file.write(str(area)+","+str(feret)+"\n")    
                cv2.putText(drawing, str(int(feret)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                box = np.intp(box)
                #cv2.drawContours(drawing, [box], 0, color)   
     imwrite(directoryferet+str(filename[:-4])+".tif",drawing)
        

 
