import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# doesn't work
def confusion_matrix(y_true, y_pred):
    raw_data = {'actual': y_true,
                'preds': y_pred}
    df = pd.DataFrame(raw_data, columns=['actual', 'preds'])
    tab = pd.crosstab(df.actual, df.preds, margins=True)
    print(tab)
    with open("confusion-matrix.txt", "w") as text_file:
        text_file.write(tab.to_string())


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


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


def plot_to_grid(batch, grid_size=None, random=False):

    img_size = batch[0].shape[0]

    nt = len(batch)  # nt = total number of images

    if grid_size is None:
        grid_size = math.ceil(math.sqrt(nt))

    try:
        ch = batch[0].shape[2]
    except:
        ch = 1

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
            if ch == 1:
                kept_patches.append(np.zeros((img_size, img_size)))
            else:
                kept_patches.append(np.zeros((img_size, img_size, ch)))

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 3
    width = grid_size * img_size + (grid_size - 1) * margin
    height = grid_size * img_size + (grid_size - 1) * margin
    if ch == 1:
        stitched_images = np.zeros((width, height))
    else:
        stitched_images = np.zeros((width, height, ch))

    # fill the picture with our saved filters
    for i in range(grid_size):
        for j in range(grid_size):
            img = kept_patches[i * grid_size + j]
            if ch == 1:
                stitched_images[(img_size + margin) * i: (img_size + margin) * i + img_size,
                                (img_size + margin) * j: (img_size + margin) * j + img_size] = img
            else:
                stitched_images[(img_size + margin) * i: (img_size + margin) * i + img_size,
                                (img_size + margin) * j: (img_size + margin) * j + img_size, ] = img
    return stitched_images


def apply_jet_colormap(img):
    img /= np.max(img)
    img = Image.fromarray(np.uint8(cm.jet(img) * 255))
    return img


def gaussian_blur(img, kernel=2):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=kernel))
    return pil_img

def merge_history_dicts(d1, d2, epoch):
    if isinstance(d1, dict):
        d2 = {key: {key_ + epoch: val_ for key_, val_ in val.items()}
                     for key, val in d2.items()}

    d = {key: (d1[key], d2[key]) for key in d2}

    return d