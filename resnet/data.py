from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from keras.models import load_model
from tqdm import tqdm
import numpy as np

def adjustImage(img):
    if(np.max(img) > 1):
        img = img/255
    return img


def adjustMask(img):
    if(np.max(img) > 1):
        img = img/255
        img[img>0.5] = 1
        img[img<=0.5] = 0
    return img


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict_image, aug_dict_mask, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict_image)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        shuffle=False)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        shuffle=False)
    train_generator = zip(image_generator, mask_generator)
    return train_generator
  

def validationGenerator(batch_size, val_path, image_folder, mask_folder, aug_dict_image, aug_dict_mask, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix ="image", mask_save_prefix ="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict_image)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)
    
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        shuffle=False)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        shuffle=False)
    val_generator = zip(image_generator, mask_generator)
    return val_generator


def testGenerator(batch_size, test_path, image_folder, aug_dict_image, image_color_mode="grayscale",
                    image_save_prefix ="image",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256,256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict_image)
    
    image_generator = image_datagen.flow_from_directory(
        test_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed,
        shuffle=False)
    
    return image_generator
    
