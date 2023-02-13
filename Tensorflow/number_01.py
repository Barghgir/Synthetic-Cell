# Download Dataset
import opendatasets as op
op.download("https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1/download?datasetVersionNumber=1")

# Import all the Libraries that we will use 
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from skimage.io import imread, imshow
from skimage.transform import resize

# use the addresses to find all the .tif files 
mask_lst_1 = glob("/content/synthetic-cell-images-and-masks-bbbc005-v1/BBBC005_v1_ground_truth/BBBC005_v1_ground_truth/*.TIF")
img_lst = glob("/content/synthetic-cell-images-and-masks-bbbc005-v1/BBBC005_v1_images/BBBC005_v1_images/*.TIF")

# Get all the path to use them to create DataFrame 
path = img_lst[1].replace(img_lst[1].split("/")[-1], "")

img_labeled = []
for i in mask_lst_1:
    new_path = path + i.split("/")[-1]
    if new_path in img_lst:
        img_labeled.append(new_path)

img_labeled.sort()
mask_lst_1.sort()

# Create DataFrame from building dictionary
dict_labeled = {
    "image_labeled": img_labeled,
    "mask_labeled": mask_lst_1,
}

df_labeled = pd.DataFrame(dict_labeled)

# Add a column showing how many cells are on each image

def get_num_cells(x):
    x = x.split("/")[-1]
    # split on the _
    a = x.split('_')
    # choose the third item
    b = a[2] # e.g. C53
    # choose second item onwards and convert to int
    num_cells = int(b[1:])
    
    return num_cells

# create a new column called 'num_cells'
df_labeled['num_cells'] = df_labeled['image_labeled'].apply(get_num_cells)


# Add a column indicating if an image has a mask.

# Keep in mind images and masks have the same file names.

def check_for_mask(x):
    x = x.split("/")[-1]
    mask_list = [i.split("/")[-1] for i in mask_lst_1]
    if x in mask_list:
        return 'yes'
    else:
        return 'no'
    
# create a new column called 'df_labeled'
df_labeled['has_mask'] = df_labeled['image_labeled'].apply(check_for_mask)


# Add a column showing how much blur was added to each image

def get_blur_amt(x):
    x = x.split("/")[-1]
    # split on the _
    a = x.split('_')
    # choose the third item
    b = a[3] # e.g. F1
    # choose second item onwards and convert to int
    blur_amt = int(b[1:])
    
    return blur_amt

# create a new column called 'blur_amt'
df_labeled['blur_amt'] = df_labeled['image_labeled'].apply(get_blur_amt)
