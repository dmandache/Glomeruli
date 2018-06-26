from keras import backend as K
from scipy.misc import imsave
import numpy as np
import math

from util import *

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

def getFilters(model, layer_name, img_width, img_height, input_img=None):

    nb_filters = 64

    if input_img is None:
        input_img = model.input

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    kept_filters = []
    for filter_index in range(0, nb_filters-1):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
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

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise

        input_img_data = np.random.random((1, img_height, img_width, 1)) * 20 + 128.

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

        # decode the resulting input image

        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 5

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    print(len(kept_filters))
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    # save the result to disk
    imsave('filters_%s_%dx%d.png' % (layer_name, n, n), stitched_filters)

def save_conv_filters_to_file (model):
    for layer in model.layers:
        if "convolution" in layer.name:
            #print " === ", layer.name, " === ", len(layer.get_weights()), " === ", layer.get_weights()[0].shape
            weights = layer.get_weights()[0]
            weights = weights[:,:,0,:]
            n = int(math.ceil(math.sqrt(layer.nb_filter)))
            filters = []
            for i in range(layer.nb_filter):
                filters.append(weights[:,:,i])
            plot_to_grid(filters, layer.name, grid_size=n)



def visualize_activation_map(model,layer_name,img):
    layer = model.get_layer(layer_name)

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
    activations = get_activations([img, 0])[0]


    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name, activations.ndim))

    print(activations.shape)
    nb_maps = activations.shape[3]
    grid_size = math.ceil(math.sqrt(nb_maps))
    feature_maps = activations[0, :, :, :]
    feature_maps_swap = np.swapaxes(feature_maps, 2, 0)
    plot_to_grid(feature_maps_swap, "feature_map_" + layer_name, grid_size=grid_size)
    '''
    feature_maps = []
    for i in range(nb_maps):
        feature_maps.append(activations[0,:,:,i])
        plot_to_grid(feature_maps, "feature_map_"+layer_name, grid_size=6)
    '''


def visualize_class_activation_map(model, img, output_path):
    width, height, _ = img.shape

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
    cam_new = Image.fromarray(np.uint8(cm.jet(cam_new) * 255))
    cam_new.save(output_path)