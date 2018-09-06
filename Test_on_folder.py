from keras.models import load_model
from MyMetrics import precision, recall, sensitivity, specificity, f1_score

import argparse

import numpy as np
import pickle
from scipy.misc import imsave

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import _util
import _vis

import settings

settings.init()


def main(dir=None, n=None):
    if dir is None:
        # IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data"
    else:
        IMAGES_DIR_PATH = dir

    if n is None:
        NB_SAMPLES = 64
    else:
        NB_SAMPLES = n

    DIR_GLOM = IMAGES_DIR_PATH + "/glomeruli"
    DIR_NONGLOM = IMAGES_DIR_PATH + "/nonglomeruli"

    # Load some images and preprocess them
    # x_test_glom = _util.get_n_samples(NB_SAMPLES, dir=DIR_GLOM, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))
    # x_test_nonglom = _util.get_n_samples(NB_SAMPLES, dir=DIR_NONGLOM, target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    # Load all test samples
    x_test_glom = _util.get_all_imgs_from_folder(dir=DIR_GLOM,
                                                 target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))
    x_test_nonglom = _util.get_all_imgs_from_folder(dir=DIR_NONGLOM, target_size=(
    settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT))

    # load model
    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    # model.summary()

    # print('Those should be all ones - glom 1')
    y_pred_glom = model.predict(x_test_glom).flatten()
    y_true_glom = list(np.zeros(len(y_pred_glom)))

    glomeruli_examples = _util.plot_to_grid_with_proba(x_test_glom, y_pred_glom)
    imsave('./output/%s.png' % 'glomeruli_examples', glomeruli_examples)

    # print('Those should be all zeros - nonglom 0')
    y_pred_nonglom = model.predict(x_test_nonglom).flatten()
    y_true_nonglom = list(np.ones(len(y_pred_nonglom)))
    # print(y_test_nonglom.argmax(axis=-1)) # for categorical

    non_glomeruli_examples = _util.plot_to_grid_with_proba(x_test_nonglom, y_pred_nonglom, grid_size=12, shuffle=True)
    imsave('./output/%s.png' % 'non_glomeruli_examples', non_glomeruli_examples)

    img = x_test_glom[10, :, :, :]
    imsave('./output/glom.png', img)
    img_input_glom = np.expand_dims(img, axis=0)
    print('Glomeruli : probability = {} '.format(y_pred_glom[10]))

    img = x_test_nonglom[35, :, :, :]
    imsave('./output/nonglom.png', img)
    img_input_nonglom = np.expand_dims(img, axis=0)
    print('Non Glomeruli : probability = {} '.format(y_pred_nonglom[10]))

    y_pred = list(np.around(y_pred_glom)) + list(np.around(y_pred_nonglom))
    y_proba = list(y_pred_glom) + list(y_pred_nonglom)
    y_true = y_true_glom + y_true_nonglom
    x = list(x_test_glom) + list(x_test_nonglom)

    with open('y_proba', 'wb') as f:
        pickle.dump(y_proba, f)

    with open('y_proba', 'rb') as f:
        pickle.load(f)

    '''

     prob dist
    '''
    num_bins = 100
    n, bins, patches = plt.hist(y_proba, num_bins, facecolor='blue', log=True)  # alpha=0.5)
    plt.savefig('test_nonglom_proba_dist.png')

    _util.confusion_matrix(y_true, y_pred)

    # _vis.visualize_model_max_activations(model, grad_step=0.5, grad_iter=300)

    # _vis.visualize_model_weights(model)

    # _vis.visualize_model_activation_maps(model, img_input_glom)
    # _vis.visualize_model_activation_maps(model, img_input_nonglom)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    args = parser.parse_args()

    main(**vars(args))

