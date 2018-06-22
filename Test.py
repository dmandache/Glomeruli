from keras.models import load_model
from MyMetrics import sensitivity, specificity

import os
import random
import argparse

import numpy as np
from PIL import Image

def get_n_samples(n=32, dir=None, target_size=(299,299)):
    samples = []
    files = os.listdir(dir)
    for _ in range(n):
        random_file = random.choice(files)
        path_to_random_file = dir + '/' + random_file
        x = Image.open(path_to_random_file)
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

def main(dir=None):
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

    model = load_model('./output/model.hdf5', custom_objects={'sensitivity': sensitivity, 'specificity': specificity} )

    y_test_glom = model.predict(x_test_glom)
    print(y_test_glom)

    y_test_nonglom = model.predict(x_test_nonglom)
    print(y_test_nonglom)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    args = parser.parse_args()

    main(**vars(args))