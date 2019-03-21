import os

from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from Models.Metrics import precision, recall, sensitivity, specificity, f1_score
import settings

"""
    Train State of the Art (SoA) models:
        - InceptionV3
        - VGG16
        - ResNet50
    either :    - from scratch
                - fine tuning (from ImageNet)
                - transfer learning (from ImageNet)
    ####
    Note: For fine-tuning: to freeze layers look up this code snippet :
            for layer in model.layers[:TRAINABLE_LAYERS]:
                layer.trainable = True  ## and change to False
"""

'''
    Define model
'''
#ToDo
def get_base_model(model_name, weights='imagenet'):
    global BASE_LAYERS, ALL_LAYERS, TRAINABLE_LAYERS, FC_LAYER_SIZE

    if model_name == 'inception':
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

        wfile = "./imagenet_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
        if weights == 'imagenet' and os.path.isfile(wfile):
            weights = wfile
        base_model = InceptionV3(weights=weights, include_top=False)
        BASE_LAYERS = len(InceptionV3(weights=None, include_top=False).layers)
        ALL_LAYERS = len(InceptionV3(weights=None, include_top=True).layers)
        #TRAINABLE_LAYERS = 172
        FC_LAYER_SIZE = 1024

    elif model_name == 'vgg':
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

        wfile = "./imagenet_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        if weights == 'imagenet' and os.path.isfile(wfile):
            weights = wfile
        base_model = VGG16(weights=weights, include_top=False)
        BASE_LAYERS = len(VGG16(weights=None, include_top=False).layers)    # 19
        ALL_LAYERS = len(VGG16(weights=None, include_top=True).layers)      # 23
        #TRAINABLE_LAYERS = 11
        FC_LAYER_SIZE = 1024

    elif model_name == 'resnet':
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

        wfile = "./imagenet_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        if weights == 'imagenet' and os.path.isfile(wfile):
            weights = wfile
        base_model = ResNet50(weights=weights, include_top=False)           # 174
        BASE_LAYERS = len(ResNet50(weights=None, include_top=False).layers)
        ALL_LAYERS = len(ResNet50(weights=None, include_top=True).layers)
        #TRAINABLE_LAYERS = 8
        FC_LAYER_SIZE = 1024
    print('\t {} model with {} base layers !'.format(model_name, BASE_LAYERS))
    return base_model


# weights = 'imagenet' or None
def get_model(model_name, num_classes, weights='imagenet'):

    base_model = get_base_model(model_name, weights)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(FC_LAYER_SIZE, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=[base_model.input], outputs=[predictions])
    return model


def get_top_layer_model(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:BASE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[BASE_LAYERS:]:
        layer.trainable = True

    # compile the model (should be done after setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, sensitivity, specificity, f1_score])

    return model


def get_mid_layer_model(model):
    """After we fine-tune the dense layers, train deeper."""
    # freeze the first TRAINABLE_LAYER_INDEX layers and unfreeze the rest
    for layer in model.layers[:TRAINABLE_LAYERS]:
        layer.trainable = True  # change here if you want to freeze first layers i.e. keep weights traine don ImageNet
    for layer in model.layers[TRAINABLE_LAYERS:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, sensitivity, specificity, f1_score])

    return model


'''
    Train model
'''

def train_finetune(model_name, train_generator, validation_generator, callback_list):
    global fine_tune_epoch

    print('Finetuning !')

    model = get_model(model_name, settings.NUM_CLASSES, weights='imagenet')

    print("Training dense classifier from scratch")
    # Get and train the top layers.
    model = get_top_layer_model(model)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=settings.NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=settings.NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.DENSE_TRAIN_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=callback_list)

    model = load_model(settings.OUTPUT_DIR+'/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    fine_tune_epoch = len(history.history['loss'])
    print('Epoch when fine-tuning starts: %d' % fine_tune_epoch)

    print("Fine-tune InceptionV3")
    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    history = model.fit_generator(
        train_generator,
        initial_epoch=fine_tune_epoch,
        steps_per_epoch=settings.NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=settings.NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=callback_list)

    model = load_model(settings.OUTPUT_DIR+'/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})
    return model, history, fine_tune_epoch


def train_transfer(model_name, train_generator, validation_generator, callback_list):
    print('Transfer learning !')

    model = get_model(model_name, settings.NUM_CLASSES, weights='imagenet')

    print("Training dense classifier from scratch")
    # Get and train the top layers.
    model = get_top_layer_model(model)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=settings.NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=settings.NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=callback_list)

    model = load_model(settings.OUTPUT_DIR + '/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})
    return model, history


def train_from_scratch(model_name, train_generator, validation_generator, callback_list):
    print('Training from scratch !')

    model = get_model(model_name, settings.NUM_CLASSES, weights=None)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, sensitivity, specificity, f1_score])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=settings.NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=settings.NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=callback_list)

    model = load_model(settings.OUTPUT_DIR+'/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    return model, history
