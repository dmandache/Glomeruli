import os
import argparse
import numpy as np
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

from Models.Metrics import precision, recall, sensitivity, specificity, f1_score
from Util import util
import settings


def main(dir=None, model=None, out=None):
    if dir is None:
        # IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        IMAGES_DIR_PATH = "/Volumes/Raid1Data/2018_Glomeruli/data"
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
    elif ('resnet' or 'vgg') in MODEL_PATH:
        settings.MODEL_INPUT_WIDTH = 224
        settings.MODEL_INPUT_HEIGHT = 224
        settings.MODEL_INPUT_DEPTH = 3

    GLOM_DIR_PATH = os.path.join(IMAGES_DIR_PATH,'glomeruli')
    NONGLOM_DIR_PATH = os.path.join(IMAGES_DIR_PATH, 'nonglomeruli')

    # Load all test samples
    x_test_glom, x_filename_glom = util.get_all_imgs_from_folder(
        dir=GLOM_DIR_PATH, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))
    x_test_nonglom, x_filename_nonglom = util.get_all_imgs_from_folder(
        dir=NONGLOM_DIR_PATH, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    # ground truth class
    # 0 for glomeruli, 1 for nonglomeruli
    y_test_glom = np.zeros(x_test_glom.shape[0])
    y_test_nonglom = np.ones(x_test_nonglom.shape[0])

    # put data together
    x_test = []
    x_test.extend(x_test_glom)
    x_test.extend(x_test_nonglom)
    x_test = np.asarray(x_test, dtype='float32')

    x_filename = []
    x_filename.extend(x_filename_glom)
    x_filename.extend(x_filename_nonglom)

    y_test = np.concatenate((y_test_glom, y_test_nonglom), axis=None)

    # load model
    model = load_model(MODEL_PATH, custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    # model.summary()
    print("Predicting...")
    y_proba = model.predict(x_test).flatten()
    y_class = np.around(y_proba)

    util.prediction_to_folder(images=x_test, image_names=x_filename, true_class=y_test, pred_class=y_class, pred_proba=y_proba, path_dir=out)

    util.confusion_matrix(y_test, y_class)


    '''

     prob dist
    '''
    num_bins = 100
    n, bins, patches = plt.hist(y_proba, num_bins, facecolor='blue', log=True)  # alpha=0.5)
    plt.savefig('test_nonglom_proba_dist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on images organized in glomeruli / nonglomeruli subfolders.')
    parser.add_argument('--dir', help='path to parent directory (with glomeruli / nonglomeruli subfolders)')
    parser.add_argument('--model', help='model filepath')
    parser.add_argument('--out', help='output directory')
    args = parser.parse_args()

    main(**vars(args))

