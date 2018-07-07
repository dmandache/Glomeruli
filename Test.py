from keras.models import load_model
from MyMetrics import precision, recall, sensitivity, specificity, f1_score

import argparse

import numpy as np
import math
from scipy.misc import imsave

import _util
import _vis

import settings
settings.init()


def main(dir=None, n=None):
    if dir is None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        #IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data
    else:
        IMAGES_DIR_PATH = dir

    if n is None:
        NB_SAMPLES = 64
    else:
        NB_SAMPLES = n

    DIR_GLOM = IMAGES_DIR_PATH + "/glomeruli"
    DIR_NONGLOM = IMAGES_DIR_PATH + "/nonglomeruli"

    # Load some images and preprocess them
    x_test_glom = _util.get_n_samples(NB_SAMPLES, dir=DIR_GLOM, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))
    x_test_nonglom = _util.get_n_samples(NB_SAMPLES, dir=DIR_NONGLOM, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    x_test_glom /= 255
    x_test_nonglom /= 255

    # load model
    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    model.summary()

    glomeruli_examples = _util.plot_to_grid(x_test_glom)
    imsave('./output/%s.png' % 'glomeruli_examples', glomeruli_examples)

    non_glomeruli_examples = _util.plot_to_grid(x_test_nonglom)
    imsave('./output/%s.png' % 'non_glomeruli_examples', non_glomeruli_examples)

    # print('Those should be all ones - glom 1')
    y_pred_glom = model.predict(x_test_glom)
    y_true_glom = list(np.ones(len(y_pred_glom)))
    TP = np.count_nonzero(y_pred_glom >= 0.51)
    print('True Positives ', TP)
    print(y_pred_glom.flatten())

    # print('Those should be all zeros - nonglom 0')
    y_pred_nonglom = model.predict(x_test_nonglom)
    y_true_nonglom = list(np.zeros(len(y_pred_nonglom)))
    TN = np.count_nonzero(y_pred_nonglom <= 0.5)
    print('True Negatives ', TN)
    print(y_pred_nonglom.flatten())   #print(y_test_nonglom.argmax(axis=-1)) # for categorical

    img = x_test_glom[10, :, :, :]
    imsave('./output/glom.png', img)
    img_input = np.expand_dims(img, axis=0)
    print('Glomeruli probability  = {} ' .format(y_pred_glom[10][0]))

    y_pred = y_pred_glom.flatten() + y_pred_nonglom.flatten()
    y_true = y_true_glom + y_true_nonglom
    _util.confusion_matrix(y_true,y_pred)

    _vis.visualize_model_max_activations(model, grad_step=0.5, grad_iter=300)

    _vis.visualize_model_weights(model)

    _vis.visualize_model_activation_maps(model, img_input)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--n', help='number of samples')
    args = parser.parse_args()

    main(**vars(args))

