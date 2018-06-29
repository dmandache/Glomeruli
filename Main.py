"""Train model using transfer learning."""
import os
import shutil
import argparse

import numpy as np
import pandas as pd

from keras.models import Model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.utils import multi_gpu_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import MyMetrics
import FlyGenerator
import Test
import _util

class_dict = {'nonglomeruli': 0, 'glomeruli': 1}

class_weight = {class_dict['nonglomeruli']:  1,    # 0 : 1
                class_dict['glomeruli']:     25}      # 1 : 25

DENSE_TRAIN_EPOCHS = 30
FINE_TUNE_EPOCHS = 300

NUM_CLASSES = 1
RANDOM_SEED = None
CLASS_MODE = 'binary'

# we chose to train the top 2 inception blocks
BATCH_SIZE = 256
TRAINABLE_LAYERS = 172
INCEPTIONV3_BASE_LAYERS = len(InceptionV3(weights=None, include_top=False).layers)
INCEPTIONV3_ALL_LAYERS = len(InceptionV3(weights=None, include_top=True).layers)

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

FC_LAYER_SIZE = 1024

# Helper: Save the model.
checkpointer_dense = ModelCheckpoint(
    filepath='./output/checkpoints/inception.dense.{epoch:03d}-{val_f1_score:.3f}.hdf5',
    verbose=1,
    monitor='val_f1_score',
    mode='max',
    save_best_only=True)

checkpointer_finetune = ModelCheckpoint(
    filepath='./output/checkpoints/inception.finetune.{epoch:03d}-{val_f1_score:.3f}.hdf5',
    verbose=1,
    monitor='val_f1_score',
    mode='max',
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper_dense = EarlyStopping(patience=10)

early_stopper_finetune = EarlyStopping(patience=30)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./output/events')

# Helper: Keep track of acc and loss during training
history_dense = History()

history_finetune = History()

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
                  metrics=['accuracy', MyMetrics.sensitivity, MyMetrics.specificity, MyMetrics.f1_score])

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
    model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', MyMetrics.sensitivity, MyMetrics.specificity, MyMetrics.f1_score])

    return model


def get_generators(image_dir, validation_pct=None):

    global class_dict, class_weight

    train_data_gen_args = dict( rescale=1. / 255,
                                rotation_range=90)

    test_data_gen_args = dict(rescale=1. / 255)

    if validation_pct is None:
        DIR_TRAIN_GLOM = image_dir + "/train/glomeruli"
        DIR_TEST_GLOM = image_dir + "/test/glomeruli"
        DIR_TRAIN_NONGLOM = image_dir + "/train/nonglomeruli"
        DIR_TEST_NONGLOM = image_dir + "/test/nonglomeruli"

        '''
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
        '''

        print('Trainset\tglomeruli:\t' + str(len(os.listdir(DIR_TRAIN_GLOM))))
        print('\t\tnon-glomeruli:\t' + str(len(os.listdir(DIR_TRAIN_NONGLOM))))
        print('Testset\tglomeruli:\t' + str(len(os.listdir(DIR_TEST_GLOM))))
        print('\t\tnon-glomeruli:\t' + str(len(os.listdir(DIR_TEST_NONGLOM))))

        TRAIN_DIR_PATH = image_dir + "/train"
        TEST_DIR_PATH = image_dir + "/test"

        NUM_TRAIN_SAMPLES = len(os.listdir(DIR_TRAIN_GLOM)) + len(os.listdir(DIR_TRAIN_NONGLOM))
        NUM_TEST_SAMPLES = len(os.listdir(DIR_TEST_GLOM)) + len(os.listdir(DIR_TEST_NONGLOM))

        train_datagen = ImageDataGenerator(**train_data_gen_args)

        test_datagen = ImageDataGenerator(**test_data_gen_args)

        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR_PATH,
            target_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE)

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
            TEST_DIR_PATH,
            target_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE)
    else:

        image_lists = FlyGenerator.create_image_lists(image_dir, validation_pct)

        classes = list(image_lists.keys())
        print(classes)
        num_classes = len(classes)

        NUM_TRAIN_SAMPLES = 0
        NUM_TEST_SAMPLES = 0
        for i in range(num_classes):
            NUM_TRAIN_SAMPLES += len(image_lists[classes[i]]['training'])
            NUM_TEST_SAMPLES += len(image_lists[classes[i]]['validation'])

        train_datagen = FlyGenerator.CustomImageDataGenerator(**train_data_gen_args)

        test_datagen = FlyGenerator.CustomImageDataGenerator(**test_data_gen_args)

        train_generator = train_datagen.flow_from_image_lists(
            image_lists=image_lists,
            category='training',
            image_dir=image_dir,
            target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE,
            seed=RANDOM_SEED)

        validation_generator = test_datagen.flow_from_image_lists(
            image_lists=image_lists,
            category='validation',
            image_dir=image_dir,
            target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE,
            seed=RANDOM_SEED)

        # modify global variables if necessary

        class_dict = train_generator.class2id
        print("label mapping is : ", class_dict)

        class_weight = {class_dict['nonglomeruli']: 1,  # 0 : 1
                        class_dict['glomeruli']: 25}    # 1 : 25
        print("class weighting is : ", class_weight)

    return train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES

def main(dir=None, split=None):

    if dir == None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data/"
    else:
        IMAGES_DIR_PATH = dir

    if split == None:
        VALIDATION_SPLIT = 10
    else:
        VALIDATION_SPLIT = split

    model = get_model(NUM_CLASSES)

    train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES = get_generators(IMAGES_DIR_PATH, VALIDATION_SPLIT)

    os.makedirs('./output/checkpoints/', exist_ok=True)

    print("Training dense classifier from scratch")
    # Get and train the top layers.
    model = get_top_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES//BATCH_SIZE,
        epochs=DENSE_TRAIN_EPOCHS,
        class_weight=class_weight,
        callbacks=[checkpointer_dense, early_stopper_dense, tensorboard, history_dense])

    print("Fine-tune InceptionV3, bottom layers frozen")
    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES//BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weight,
        callbacks=[checkpointer_finetune, early_stopper_finetune, tensorboard, history_finetune])

    # save model
    model.save('./output/model.hdf5', overwrite=True)

    # save metrics during training epochs
    pd.DataFrame(history_dense.history).to_csv("./output/history_classifier.csv")

    pd.DataFrame(history_finetune.history).to_csv("./output/history_finetune.csv")

    # plot  metrics during training epochs
    plt.plot(history_dense.history['loss'], 'ro-', label='Train loss')
    plt.plot(history_dense.history['val_loss'], 'go-', label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./output/training_plot_classifier.png')

    plt.plot(history_finetune.history['loss'], 'ro-', label='Train loss')
    plt.plot(history_finetune.history['val_loss'], 'go-', label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./output/training_plot_finetune.png')

    Test.main(IMAGES_DIR_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--split', help='percentage of validation data')
    args = parser.parse_args()

    main(**vars(args))
