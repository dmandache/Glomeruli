from keras.models import load_model
from MyMetrics import precision, recall, sensitivity, specificity, f1_score

import argparse
import numpy as np
import csv
import pickle
from scipy.misc import imsave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _util
import settings
settings.init()


def main(dir=None, out=None):
    if dir is None:
        # IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/small_test"
    else:
        IMAGES_DIR_PATH = dir


    # Load all test samples
    x_test = _util.get_all_imgs_from_folder(dir=IMAGES_DIR_PATH, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    # load model
    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    # model.summary()

    y_proba = model.predict(x_test).flatten()

    y_class = np.around(y_proba) # 0 for glomeruli, 1 for nonglomeruli

    y_max = y_proba.argmax(axis=-1)

    _util.prediction_to_folder(x_test, y_proba, out)

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
    parser.add_argument('--out', help='output directory')
    args = parser.parse_args()

    main(**vars(args))

