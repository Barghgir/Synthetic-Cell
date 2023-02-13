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

dict_labeled = {
    "image_labeled": img_labeled,
    "mask_labeled": mask_lst_1,
}

df_labeled = pd.DataFrame(dict_labeled)
