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

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


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
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
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
        img1 = img[0:512,0:512]
        img2 = img[0:512,752:1264]
        img3 = img[0:512,376:888]
        img4 = img[169:681,0:512]
        img5 = img[169:681,752:1264]
        img6 = img[169:681,376:888]
        img1 = trans.resize(img1,target_size)
        img1 = np.reshape(img1,img1.shape+(1,)) if (not flag_multi_class) else img1
        img1 = np.reshape(img1,(1,)+img1.shape)
        img2 = trans.resize(img2,target_size)
        img2 = np.reshape(img2,img2.shape+(1,)) if (not flag_multi_class) else img2
        img2 = np.reshape(img2,(1,)+img2.shape)
        img3 = trans.resize(img3,target_size)
        img3 = np.reshape(img3,img3.shape+(1,)) if (not flag_multi_class) else img3
        img3 = np.reshape(img3,(1,)+img3.shape)
        img4 = trans.resize(img4,target_size)
        img4 = np.reshape(img4,img4.shape+(1,)) if (not flag_multi_class) else img4
        img4 = np.reshape(img4,(1,)+img4.shape)
        img5 = trans.resize(img5,target_size)
        img5 = np.reshape(img5,img5.shape+(1,)) if (not flag_multi_class) else img5
        img5 = np.reshape(img5,(1,)+img5.shape)
        img6 = trans.resize(img6,target_size)
        img6 = np.reshape(img6,img6.shape+(1,)) if (not flag_multi_class) else img6
        img6 = np.reshape(img6,(1,)+img6.shape)
        yield img1
        yield img2
        yield img3
        yield img4
        yield img5
        yield img6

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



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    counter= 0
    for i,item in enumerate(npyfile):
        img = np.zeros(shape=(681,1264),dtype=np.float32)
        if i%6 == 0:
            img1 = item[:,:,0]
        if i%6 == 1:
            img2 = item[:,:,0]
        if i%6 == 2:
            img3 = item[:,:,0]
        if i%6 == 3:
            img4 = item[:,:,0]
        if i%6 == 4:
            img5 = item[:,:,0]
        if i%6 == 5:
            img6 = item[:,:,0]
            img[0:512,0:462] = img1[:,0:462]
            img[0:512,802:1264] = img2[:,50:512]
            img[0:512,426:838] = img3[:,50:462]
            img[219:681,0:512] = img4[50:512,:]
            img[219:681,802:1264] = img5[50:512:,50:512]
            img[219:681,426:838] = img6[50:512,50:462]
            img = 255*img
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path,"%d_mask.tif"%counter),img)
            counter+=1