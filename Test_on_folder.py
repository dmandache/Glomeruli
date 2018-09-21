from keras.models import load_model
from Models.Metrics import precision, recall, sensitivity, specificity, f1_score

import argparse
import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Others import Util
import settings
settings.init()


def main(dir=None, model=None, out=None):
    if dir is None:
        # IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/small_test"
    else:
        IMAGES_DIR_PATH = dir
    if model is None:
        MODEL_PATH = './output/model.hdf5'
        print('You shpuld specify the path to the model file.')
    else:
        MODEL_PATH = model

    if 'inception' in MODEL_PATH:
        settings.MODEL_INPUT_WIDTH = 299
        settings.MODEL_INPUT_HEIGHT = 299
        settings.MODEL_INPUT_DEPTH = 3
    elif 'resnet' or 'vgg' in MODEL_PATH:
        settings.MODEL_INPUT_WIDTH = 224
        settings.MODEL_INPUT_HEIGHT = 224

    # Load all test samples
    x_test = Util.get_all_imgs_from_folder(dir=IMAGES_DIR_PATH, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    # load model
    model = load_model(MODEL_PATH, custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    # model.summary()

    y_proba = model.predict(x_test).flatten()

    y_class = np.around(y_proba) # 0 for glomeruli, 1 for nonglomeruli

    y_max = y_proba.argmax(axis=-1)

    Util.prediction_to_folder(x_test, y_proba, out)

    with open('y_proba', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(y_proba)

    '''

     prob dist
    '''
    num_bins = 100
    n, bins, patches = plt.hist(y_proba, num_bins, facecolor='blue', log=True)  # alpha=0.5)
    plt.savefig('test_nonglom_proba_dist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--model', help='model filepath')
    parser.add_argument('--out', help='output directory')
    args = parser.parse_args()

    main(**vars(args))

