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

