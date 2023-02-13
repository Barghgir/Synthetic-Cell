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
from Generators import gen_pairs_test, gen_pairs_train
from loss_function import DiceLoss

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


# Use Data Generators to set train_dataset and test_dataset
batch_size = 8
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_generator(generator=gen_pairs_train, output_types=(np.float32, np.float32))
train_dataset = train_dataset.batch(batch_size)

# Prepare the validation dataset.
test_dataset = tf.data.Dataset.from_generator(generator=gen_pairs_test, output_types=(np.float32, np.float32))
test_dataset = test_dataset.batch(batch_size)


# creating traning loop
epochs = 10
learning_rate = 0.001
optimizer = tf.optimizers.Adam(learning_rate)

def train_one_batch(x, y):
    y = tf.reshape(y, shape=(batch_size, 308, 308, 1))
    
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = dice_loss(y, pred)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))


def validate_one_batch(x, y):
    y = tf.reshape(y, shape=(batch_size, 308, 308, 1))
    pred = model(x, training=False)
    loss = dice_loss(y, pred)
    return loss


