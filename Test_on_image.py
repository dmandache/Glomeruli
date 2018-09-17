from keras.models import load_model
from Models.Metrics import precision, recall, sensitivity, specificity, f1_score

import argparse

import numpy as np

import matplotlib

matplotlib.use('Agg')

from PIL import Image

import settings

settings.init()

IS_GLOMERULI = 0
NOT_GLOMERULI = 1


def main(path):
    if path is None:
        # IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/test"
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data/glomeruli/542862_544892_3782094.png"
    else:
        IMAGES_DIR_PATH = path

    # load image
    try:
        img = Image.open(IMAGES_DIR_PATH)
    except IOError:
        print("Could not open image file ", path)
        exit()

    target_size = (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT)

    original_size = img.size
    print("Resizing image {} from original size {} to target size {}.".format(path, original_size, target_size))
    # zero pad if smaller
    if original_size[0] < target_size[0] and original_size[1] < target_size[1]:
        img_new = Image.new("RGB", target_size)
        img_new.paste(img, ((target_size[0] - original_size[0]) // 2,
                            (target_size[1] - original_size[1]) // 2))
    # resize if bigger
    else:
        img_new = Image.Image.resize(img, target_size)

    x = np.asarray(img_new, dtype='float32')
    x = np.expand_dims(x, axis=0)
    x /= 255

    # load model
    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    # model.summary()

    # print('Those should be all ones - glom 1')
    y_pred = model.predict(x).flatten()

    glom_prob = (1 - y_pred[0])*100

    print(" I AM {0:.4f} % SURE THAT THIS IS A GLOMERULI !".format(glom_prob))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to image')
    args = parser.parse_args()

    main(**vars(args))
