from keras.models import load_model
from MyMetrics import sensitivity, specificity, f1_score

import argparse

import numpy as np
from scipy.misc import imsave

import _util
import _vis


def main(dir=None, n=None):
    if dir is None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data"
    else:
        IMAGES_DIR_PATH = dir

    if n is None:
        NB_SAMPLES = 50
    else:
        NB_SAMPLES = n

    DIR_GLOM = IMAGES_DIR_PATH + "/glomeruli"
    DIR_NONGLOM = IMAGES_DIR_PATH + "/nonglomeruli"

    # Load some images and preprocess them
    x_test_glom = _util.get_n_samples(NB_SAMPLES, dir=DIR_GLOM, target_size=(299, 299))
    x_test_nonglom = _util.get_n_samples(NB_SAMPLES, dir=DIR_NONGLOM, target_size=(299, 299))

    model = load_model('./output/model.hdf5', custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'f1_score': f1_score})

    model.summary()

    #_util.plot_to_grid(x_test_glom, name='glomeruli_examples', grid_size=7)

    #_util.plot_to_grid(x_test_nonglom, name='non-glomeruli_examples', grid_size=7)

    # print('Those should be all ones - glom 1')
    y_test_glom = model.predict(x_test_glom)

    #print('Those should be all zeros - nonglom 0')
    y_test_nonglom = model.predict(x_test_nonglom)

    # save activation maps
    img = x_test_glom[10, :, :, :]
    imsave('./output/glom.png', img)
    img_input = np.expand_dims(img, axis=0)
    print('Glomeruli predicted as %d', y_test_glom[10][0])
    for layer in model.layers:
        if "conv" in layer.name:
            _vis.visualize_activation_map(model, layer.name, img_input, color_map='jet')

    '''
    # save confusion matrix
    raw_data = {'actual': y_test,
                'preds': y_pred}
    df = pd.DataFrame(raw_data, columns=['actual', 'preds'])
    tab = pd.crosstab(df.actual, df.preds, margins=True)
    print(tab)
    with open("confusion-matrix.txt", "w") as text_file:
        text_file.write(tab.to_string())
    '''

    # save activation map for random patch
    #vis.visualize_class_activation_map(model, img_input, "cam.png")

    # save filters
    #_vis.save_conv_filters_to_file(model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--n', help='number of samples')
    args = parser.parse_args()

    main(**vars(args))