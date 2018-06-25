from keras.models import load_model
from MyMetrics import sensitivity, specificity

import os
import random
import argparse

import numpy as np
from PIL import Image
from scipy.misc import imsave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_n_samples(n=32, dir=None, target_size=(299,299)):
    samples = []
    files = os.listdir(dir)
    random.seed(9001)

    for i in range(n):
        random_file = random.choice(files)
        path_to_random_file = dir + '/' + random_file
        try:
            img = Image.open(path_to_random_file)
        except IOError:
            i = i-1
            continue
        if target_size is not 'original':
            img_new = Image.Image.resize(img,target_size)
        else: img_new = img
        x = np.asarray(img_new, dtype='float32')
        x /= 255
        samples.append(x)

    samples = np.stack(samples, axis=0)
    #samples = np.expand_dims(samples, axis=3)
    return samples

def plot_to_grid(batch, name='images', grid_size=7, random=False):
    # img_width, img_height = patches[0].shape
    img_size = batch[0].shape[0]
    try:
        ch =  batch[0].shape[2]
    except:
        ch = 1
    nt = len(batch)            # nt = total number of images
    n = grid_size * grid_size  # n = images displayed (based on grid size)

    if n <= nt:
        if random:
            kept_idxs = random.sample(range(nt), n)
        else:
            kept_idxs = range(n)
        kept_patches = [batch[i] for i in kept_idxs]
    else:
        n0 = n - nt
        kept_patches = list(batch)
        for i in range(n0):
            kept_patches.append(np.zeros((img_size, img_size, ch)))

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 3
    width = grid_size * img_size + (grid_size - 1) * margin
    height = grid_size * img_size + (grid_size - 1) * margin
    stitched_images = np.zeros((width, height, ch))

    # fill the picture with our saved filters
    for i in range(grid_size):
        for j in range(grid_size):
            img = kept_patches[i * grid_size + j]
            stitched_images[(img_size + margin) * i: (img_size + margin) * i + img_size,
            (img_size + margin) * j: (img_size + margin) * j + img_size, ] = img

    # save the result to disk
    # imsave('./patches_%s_%d-%d.png' % (name, n, nt), stitched_patches)
    imsave('./output/%s.png' % name, stitched_images)


def main(dir=None, n=None):
    if dir is None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data"
    else:
        IMAGES_DIR_PATH = dir

    if n is None:
        NB_SAMPLES = 50
    else:
        NB_SAMPLES = n

    DIR_TRAIN_GLOM = IMAGES_DIR_PATH + "/train/01_glomeruli"
    DIR_TEST_GLOM = IMAGES_DIR_PATH + "/test/01_glomeruli"
    DIR_TRAIN_NONGLOM = IMAGES_DIR_PATH + "/train/00_nonglomeruli"
    DIR_TEST_NONGLOM = IMAGES_DIR_PATH + "/test/00_nonglomeruli"

    # Load some images and preprocess them
    x_test_glom = get_n_samples(NB_SAMPLES, dir=DIR_TEST_GLOM, target_size=(299,299))
    x_test_nonglom = get_n_samples(NB_SAMPLES, dir=DIR_TEST_NONGLOM, target_size=(299,299))

    model = load_model('./output/model.hdf5', custom_objects={'sensitivity': sensitivity, 'specificity': specificity})

    plot_to_grid(x_test_glom, name='glomeruli_examples', grid_size=7)

    plot_to_grid(x_test_nonglom, name='non-glomeruli_examples', grid_size=7)

    y_test_glom = model.predict(x_test_glom)
    print('Those should be all ones - glom 1')
    print(np.around(y_test_glom, decimals=5))
    print('Those should be all zeros - nonglom 0')
    y_test_nonglom = model.predict(x_test_nonglom)
    print(np.around(y_test_nonglom, decimals=5))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--n', help='number of samples')
    args = parser.parse_args()

    main(**vars(args))