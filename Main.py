"""Train model using transfer learning."""
import os
import shutil
import argparse

import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from MyMetrics import f1_score

from sklearn import metrics

def f1(y_true, y_pred):
    print(y_true, flush=True)
    print(y_pred, flush=True)
    return metrics.f1_score(y_true, y_pred)

# we chose to train the top 2 inception blocks
BATCH_SIZE = 100
TRAINABLE_LAYERS = 172
INCEPTIONV3_BASE_LAYERS = len(InceptionV3(weights=None, include_top=False).layers)
INCEPTIONV3_ALL_LAYERS = len(InceptionV3(weights=None, include_top=True).layers)


STEPS_PER_EPOCH = 625   # or TRAIN_SAMPLES//BATCH_SIZE
VALIDATION_STEPS = 100  # or TEST_SAMPLES//BATCH_SIZE

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

FC_LAYER_SIZE = 1024

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./output/events')


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
    predictions = Dense(num_classes, activation='softmax')(x)

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
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy',f1_score])

    return model


def get_mid_layer_model(model):
    """After we fine-tune the dense layers, train deeper."""
    # freeze the first TRAINABLE_LAYER_INDEX layers and unfreeze the rest
    for layer in model.layers[:TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[TRAINABLE_LAYERS:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy',f1_score])

    return model

def main(image_dir=None):

    VALIDATION_SPLIT = 0.2
    DATA_IS_SPLIT = True

    if image_dir == None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data"
    else:
        IMAGES_DIR_PATH = image_dir

    os.makedirs('./output/checkpoints/', exist_ok=True)

    DIR_TRAIN_GLOM = IMAGES_DIR_PATH + "/train/glomeruli"
    DIR_TEST_GLOM = IMAGES_DIR_PATH + "/test/glomeruli"
    DIR_TRAIN_NONGLOM = IMAGES_DIR_PATH + "/train/nonglomeruli"
    DIR_TEST_NONGLOM = IMAGES_DIR_PATH + "/test/nonglomeruli"

    if DATA_IS_SPLIT:
        pass
    else:  # Split data into train and validation
        files_glom = os.listdir(DIR_TRAIN_GLOM)
        files_nonglom = os.listdir(DIR_TRAIN_NONGLOM)

        for f in files_glom:
            if np.random.rand(1) < VALIDATION_SPLIT:
                shutil.move(DIR_TRAIN_GLOM + '/' + f, DIR_TEST_GLOM + '/' + f)

        for i in files_nonglom:
            if np.random.rand(1) < VALIDATION_SPLIT:
                shutil.move(DIR_TRAIN_NONGLOM + '/' + i, DIR_TEST_NONGLOM + '/' + i)

    print('Trainset\tglomeruli:\t' + str(len(os.listdir(DIR_TRAIN_GLOM))))
    print('\t\tnon-glomeruli:\t' + str(len(os.listdir(DIR_TRAIN_NONGLOM))))
    print('Testset\tglomeruli:\t' + str(len(os.listdir(DIR_TEST_GLOM))))
    print('\t\tnon-glomeruli:\t' + str(len(os.listdir(DIR_TEST_NONGLOM))))

    TRAIN_DIR_PATH = IMAGES_DIR_PATH + "/train"
    TEST_DIR_PATH = IMAGES_DIR_PATH + "/test"

    TRAIN_SAMPLES = len(os.listdir(TRAIN_DIR_PATH))
    TEST_SAMPLES = len(os.listdir(TEST_DIR_PATH))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR_PATH,
        target_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR_PATH,
        target_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model = get_model(2)

    # Get and train the top layers.
    model = get_top_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=TRAIN_SAMPLES//BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=TEST_SAMPLES//BATCH_SIZE,
        epochs=10,
        callbacks=[])

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=TRAIN_SAMPLES//BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=TEST_SAMPLES//BATCH_SIZE,
        epochs=100,
        callbacks=[checkpointer, early_stopper, tensorboard])

    # save model
    model.save('./output/model.hdf5', overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', help='data directory')
    args = parser.parse_args()

    main(**vars(args))




