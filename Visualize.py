from keras.models import load_model

import argparse
import os
import numpy as np
from PIL import Image


from Models.Metrics import precision, recall, sensitivity, specificity, f1_score
from Others import visutil
import settings


def main(out=None, model=None, activations=None, img=None, maxinput=None, weights=None):

    if model is None:
        MODEL_PATH = '/Users/diana/Desktop/Glomeruli InceptionV3/model.hdf5'
        print('You should specify the path to the model file.')
    else:
        MODEL_PATH = model

    models = ['inception', 'vgg', 'resnet', 'tiny']
    for m in models:
        if m in MODEL_PATH.lower():
            MODEL_NAME = m

    if out == None:
        settings.OUTPUT_DIR = './output_{}'.format(MODEL_NAME)
    else:
        settings.OUTPUT_DIR = out
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    '''
        Default model input size
    '''
    if MODEL_NAME is 'inception':
        settings.MODEL_INPUT_WIDTH = 299
        settings.MODEL_INPUT_HEIGHT = 299
        settings.MODEL_INPUT_DEPTH = 3
    elif MODEL_NAME is 'resnet' or 'vgg':
        settings.MODEL_INPUT_WIDTH = 224
        settings.MODEL_INPUT_HEIGHT = 224
        settings.MODEL_INPUT_DEPTH = 3

    if not activations and not maxinput and not weights:
        activations = True
        maxinput = True
        weights = True

    '''
        Load input image
    '''
    if activations:
        if img is None:
            print("If you want to visualize the activation on another image you should provide the filepath to the image ! "
                    "\n **** Pass a filepath to --img argument ! ****")
            IMAGE_PATH='./img.png'
        else:
            IMAGE_PATH = img

        # load image
        try:
            im = Image.open(IMAGE_PATH)
        except IOError:
            print("Could not open image file")
            exit()
        target_size = (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT)

        original_size = im.size
        print("Resizing image {} from original size {} to target size {}.".format(IMAGE_PATH, original_size, target_size))
        im_resized = Image.Image.resize(im, target_size)

        im_resized = np.asarray(im_resized, dtype='float32')
        im_resized = np.expand_dims(im_resized, axis=0)
        im_resized /= 255


    # load model
    model = load_model(MODEL_PATH, custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                           'specificity': specificity, 'f1_score': f1_score})

    if weights:
        visutil.visualize_model_weights(model)
    if activations:
        visutil.visualize_model_activation_maps(model, im_resized)
    if maxinput:
        visutil.visualize_model_max_activations(model, (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_DEPTH))

    print("DONE")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model filepath')
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--activations', dest='activations', default=False, action='store_true')
    parser.add_argument('--img', help='path to image file')
    parser.add_argument('--maxinput', dest='maxinput', default=False, action='store_true')
    parser.add_argument('--weights', dest='weights', default=False, action='store_true')
    args = parser.parse_args()

    main(**vars(args))