from keras.models import load_model
from MyMetrics import sensitivity, specificity

import os
import random

import numpy as np
from PIL import Image

def get_n_samples(n=32, dir=None, target_size=(299,299)):
    samples = []
    files = os.listdir(dir)
    for _ in range(n):
        random_file = random.choice(files)
        x = Image.open(random_file)
        if target_size is not 'original':
            x = Image.Image.resize(x,target_size)
        x = np.asarray(x, dtype='float32')
        x /= 255
        samples.append(x)
    return samples

def preprocess(img):
    x = np.asarray(img, dtype='float32')
    x /= 255
    #x = np.expand_dims(x, axis=3)
    return x

if dir == None:
    IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data"
else:
    IMAGES_DIR_PATH = dir

DIR_TRAIN_GLOM = IMAGES_DIR_PATH + "/train/01_glomeruli"
DIR_TEST_GLOM = IMAGES_DIR_PATH + "/test/01_glomeruli"
DIR_TRAIN_NONGLOM = IMAGES_DIR_PATH + "/train/00_nonglomeruli"
DIR_TEST_NONGLOM = IMAGES_DIR_PATH + "/test/00_nonglomeruli"

nb_samples = 32

# Load some images and preprocess them
x_test_glom = get_n_samples(nb_samples, dir=DIR_TEST_GLOM, target_size=(299,299))
x_test_nonglom = get_n_samples(nb_samples, dir=DIR_TEST_NONGLOM, target_size=(299,299))

print(x_test_glom.shape)

model = load_model('./output/model.hdf5', custom_objects={'sensitivitity': sensitivity, 'specificity' : specificity} )