from keras.models import Model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3

from MyMetrics import precision, recall, sensitivity, specificity, f1_score

TRAINABLE_LAYERS = 172

INCEPTIONV3_BASE_LAYERS = len(InceptionV3(weights=None, include_top=False).layers)
INCEPTIONV3_ALL_LAYERS = len(InceptionV3(weights=None, include_top=True).layers)

FC_LAYER_SIZE = 1024

def get_model(num_classes, weights='imagenet'):

    # create the base pre-trained model
    # , input_tensor=input_tensor

    base_model = InceptionV3(weights='imagenet', include_top=False)
    # base_model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(FC_LAYER_SIZE, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=[base_model.input], outputs=[predictions])
    return model


def get_top_layer_model(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:INCEPTIONV3_BASE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[INCEPTIONV3_BASE_LAYERS:]:
        layer.trainable = True

    # compile the model (should be done after setting layers to non-trainable)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, sensitivity, specificity, f1_score])

    return model


def get_mid_layer_model(model):
    """After we fine-tune the dense layers, train deeper."""
    # freeze the first TRAINABLE_LAYER_INDEX layers and unfreeze the rest
    for layer in model.layers[:TRAINABLE_LAYERS]:
        layer.trainable = True # CHANGED HERE (was False)
    for layer in model.layers[TRAINABLE_LAYERS:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, sensitivity, specificity, f1_score])

    return model

