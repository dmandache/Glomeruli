from keras import backend as K
from PIL import Image
from scipy.misc import imsave
import numpy as np
import math
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import _util
import settings
settings.init()

'''
    Model functions
'''


def visualize_model_max_activations(model, img_shape=(299,299,3),  grad_step=1.0, grad_iter=20, save_all_filters=True, img_placeholder=None):
    """
        Saves to file the maximum activations of all the filters of all convolutional layers in the model.
        Performs gradient ascent in the image space wrt the loss.
        Filters are ordered in descending order by loss (higher loss - better looking filter).

        Parameters
        ----------
        model : keras model
        img_shape : (width, height, h)
        grad_step : float
            step of gradient ascent
        grad_iter : int
            number of steps to perform gradient ascent
        save_all_filters : bool
            if 'False' discards the filters with negative gradient
        img_placeholder : str
            if 'None' the function uses a gray image input
            if an image is given the function produces an output similar to 'Deep Dream'

        Returns
        -------
        void
            creates an image file "filters_[layer_name]_[nb_filters].png" containing the filters of each layer in model

    """
    os.makedirs('./output/filters/', exist_ok=True)
    for layer in model.layers:
        if 'conv' in layer.name:
            print('Plotting maximum activations of layer ', layer.name)
            visualize_layer_max_activations(layer, model.input, img_shape, grad_step, grad_iter, save_all_filters, img_placeholder)


def visualize_model_weights(model):
    os.makedirs('./output/weights/', exist_ok=True)
    for layer in model.layers:
        if 'conv' in layer.name:
            print('Plotting weights of layer ', layer.name)
            visualize_layer_weights(layer)


def visualize_model_activation_maps(model, img, color_map=None):
    os.makedirs('./output/activation_maps/', exist_ok=True)
    for layer in model.layers:
        if 'conv' in layer.name:
            print('Plotting activation maps of layer ', layer.name)
            visualize_layer_activation_maps(model, layer, img, color_map)


'''
    Layer functions
'''


def visualize_layer_max_activations(layer, model_input, img_shape=None, grad_step=1.0, grad_iter=100,
                                    save_all_filters=True, sort_descending_loss=False, img_placeholder=None):

    if img_shape:
        (img_width, img_height, ch) = img_shape
    else:
        img_width = settings.MODEL_INPUT_WIDTH
        img_height = settings.MODEL_INPUT_HEIGHT
        ch = settings.MODEL_INPUT_DEPTH

    weights = layer.get_weights()[0]
    nb_filters = weights.shape[-1]

    input_img = model_input

    kept_filters = []

    for filter_index in range(nb_filters):

        print('Processing filter %d of %d' % (filter_index, nb_filters))
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer.output

        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        # we start from a gray image with some random noise
        if img_placeholder is None:
            input_img_data = np.random.random((1, img_width, img_height, ch))
            input_img_data = (input_img_data - 0.5) * 20 + 128
        else:
            input_img_data = np.expand_dims(img_placeholder, 0)

        # we run gradient ascent for n steps
        for i in range(grad_iter):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * float(grad_step)

            #print('Current loss value:', loss_value)

            if loss_value <= 0. and save_all_filters is False:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if save_all_filters is True or loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        # print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # the filters that have the highest loss are assumed to be better-looking.
    if sort_descending_loss:
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    # unzip filters and their losses
    filters, losses = zip(*kept_filters)
    filters = list(filters)

    file_name = '%s_%d' % (layer.name, nb_filters)

    # save to file
    filters_img = _util.plot_to_grid(filters)
    filters_img = _util.gaussian_blur(filters_img)

    imsave('./output/filters/%s.png' % file_name, filters_img)


def visualize_layer_weights(layer):
    weights = layer.get_weights()[0]
    weights = weights[:, :, 0, :]
    width, height, nb_filters = weights.shape
    # only plot square conv filters
    if width == height:
        filter_weights = np.swapaxes(weights, -1, 0)
        file_name = '%s_%d' % (layer.name, nb_filters)
        filter_weights_img = _util.plot_to_grid(filter_weights)
        imsave('./output/weights/%s.png' % file_name, filter_weights_img)
    else:
        print('Skipping layer %s with conv filters of size %d x %d' (layer.name, width, height))


def visualize_layer_activation_maps(model, layer, img, color_map=False):

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
    activations = get_activations([img, 0])[0]

    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name, activations.ndim))

    print(activations.shape)
    nb_maps = activations.shape[-1]
    feature_maps = activations[0, :, :, :]
    feature_maps_swap = np.swapaxes(feature_maps, -1, 0)
    file_name = '%s_%d' % (layer.name, nb_maps)
    maps_img = _util.plot_to_grid(feature_maps_swap)
    if color_map:
        maps_img = _util.apply_jet_colormap(maps_img)
        file_name += '_jet'
    imsave('./output/activation_maps/%s.png' % file_name, maps_img)


# doesn't work
def visualize_class_activation_map(model, img, output_path):
    _, width, height, ch = img.shape

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-2].get_weights()[0]
    final_conv_layer = model.get_layer("conv2d_266")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-2].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_outputs[i, :, :]
    print("predictions", predictions)
    cam /= np.max(cam)
    cam_new = Image.Image.resize(cam, (height, width))
    cam_new = Image.fromarray(np.uint8(cm. jet(cam_new) * 255))
    cam_new.save(output_path)


'''
    Utility functions
'''


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# utility function to normalize a tensor by its L2 norm
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
