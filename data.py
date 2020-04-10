from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2 as cv
from tifffile import imread, imwrite
import sys
from random import randint

def adjustData(img,mask,flag_multi_class,num_class):
    if(np.max(img) > 1):
        img = img / np.max(img)
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_format='tif',
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_format='tif',
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tif"%i),as_gray = as_gray)
        img = img / np.max(img)
        curimg = np.zeros(shape=img.shape,dtype = np.float32)
        patches = getpatches(img.shape[0],img.shape[1])
        for j in range(len(patches)):
            ranges = patches[j]
            curimg = img[ranges[0]:ranges[2],ranges[1]:ranges[3]]
            curimg = np.reshape(curimg,curimg.shape+(1,))
            curimg = np.reshape(curimg,(1,)+curimg.shape)
            yield curimg

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2,dims = (512,512)):
    counter= 0
    patches = getpatches(dims[0],dims[1])
    nopatch = len(patches)
    img = np.zeros(shape=(dims[0],dims[1]),dtype=np.float32)
    for i,item in enumerate(npyfile):
        currentpatch = i % nopatch
        ranges = patches[currentpatch]
        offsetxlower = 0
        offsetylower = 0
        offsetxhigher = 0
        offsetyhigher = 0
        if ranges[0] != 0:
            offsetxlower = 50
        if ranges[1] != 0:
            offsetylower = 50
        if ranges[2] != dims[0]:
            offsetxhigher = -50
        if ranges[3] != dims[1]:
            offsetyhigher = -50
        if currentpatch != nopatch-1:
            img[ranges[0]+offsetxlower:ranges[2]+offsetxhigher,ranges[1]+offsetylower:ranges[3]+offsetyhigher] = item[offsetxlower:512+offsetxhigher,offsetylower:512+offsetyhigher,0]
        else:
            img[ranges[0]+offsetxlower:ranges[2]+offsetxhigher,ranges[1]+offsetylower:ranges[3]+offsetyhigher] = item[offsetxlower:512+offsetxhigher,offsetylower:512+offsetyhigher,0]
            img = 255*img
            io.imsave(os.path.join(save_path,"%d_mask.tif"%counter),img.astype(np.uint8))
            counter+=1

def getpatches(x,y):
    assert x >= 512
    assert y >= 512
    numberofxtackons = divmod(x-512,400)[0]+2
    numberofytackons = divmod(y-512,400)[0]+2
    patches = list()
    for i in range(numberofxtackons):
        for j in range(numberofytackons):
            upperx = min(400*i+512,x)
            uppery = min(400*j+512,y)
            patches.append((upperx-512,uppery-512,upperx,uppery))
    return patches
    