from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from skimage.io import imread, imshow
from skimage.transform import resize

# Data Generator

# Defining generator functions for train/test samples
def get_input(path):     
    # read the image using skimage
    image = imread(path)
    
    # resize the image
    image = resize(image, (500, 500), mode='constant', preserve_range=True)
    
    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    image = np.expand_dims(image, axis=-1)

    return image

def get_output(path):
    # read the image using skimage
    mask = imread(path)
    
    # resize the image
    mask = resize(mask, (308, 308), mode='constant', preserve_range=True)
    
    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def gen_pairs_train():
    data_paths = df_labeled
    for i in range(train_mount):
        # Get a random image each time
        idx = np.random.randint(0,train_mount)
        
        yield (get_input(data_paths["image_labeled"][idx]), get_output(data_paths["mask_labeled"][idx]))


def gen_pairs_test():
    data_paths = df_labeled
    for i in range(train_mount, val_mount):
        # Get a random image each time
        idx = np.random.randint(train_mount, val_mount)

        # x = tf.convert_to_tensor(get_input(data_paths["mixed_img"][idx]), dtype=tf.float32)
        # y = tf.convert_to_tensor(get_output(data_paths["inf_mask_paths"][idx]), dtype=tf.float32)
        x = get_input(data_paths["image_labeled"][idx])
        y = get_output(data_paths["mask_labeled"][idx])
        yield x, y

# Function to test input pipeline
sample_image, sample_label = next(gen_pairs_train())