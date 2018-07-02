"""Train model using transfer learning."""
import os
import shutil
import argparse

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.utils import multi_gpu_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import MyModel
import MyInception
from  MyMetrics import sensitivity, specificity, f1_score
import Data
import Test
import _util

import settings
settings.init_globals()

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./output/model.hdf5',
    verbose=1,
    monitor='val_f1_score',
    mode='max',
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=30)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./output/events')

# Helper: Keep track of acc and loss during training
history = History()

def train_inception(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES ):

    model = MyInception.get_model(settings.NUM_CLASSES)

    print("Training dense classifier from scratch")
    # Get and train the top layers.
    model = MyInception.get_top_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.DENSE_TRAIN_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=[checkpointer, early_stopper, tensorboard, history])

    model = load_model('./output/model.hdf5',
                       custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'f1_score': f1_score})

    fine_tune_epoch = len(history.history['loss'])
    print('Epoch when fine-tuning starts: %d' % fine_tune_epoch)

    print("Fine-tune InceptionV3, bottom layers frozen")
    # Get and train the mid layers.
    model = MyInception.get_mid_layer_model(model)
    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=[checkpointer, early_stopper, tensorboard, history])

    model = load_model('./output/model.hdf5',
                       custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'f1_score': f1_score})
    return model


def train_mymodel(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES ):

    model = MyModel.get_model(settings.NUM_CLASSES)

    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=[checkpointer, early_stopper, tensorboard, history])

    model = load_model('./output/model.hdf5',
                       custom_objects={'sensitivity': sensitivity, 'specificity': specificity, 'f1_score': f1_score})

    return model



def main(dir=None, split=None):

    if dir == None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/data/"
    else:
        IMAGES_DIR_PATH = dir

    if split == None:
        VALIDATION_SPLIT = 10
    else:
        VALIDATION_SPLIT = split

    train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES = Data.get_generators(IMAGES_DIR_PATH, VALIDATION_SPLIT)

    #model = train_inception(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    model = train_mymodel(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    model.summary()

    # save metrics during training epochs
    pd.DataFrame(history.history).to_csv("./output/history.csv")

    # plot  metrics during training epochs
    plt.style.use('seaborn-notebook')
    plt.plot(history.history['loss'], 'go-', label='Train loss')
    plt.plot(history.history['val_loss'], 'ro-', label='Test loss')
    plt.plot(history.history['acc'], 'g*-', label='Train acc')
    plt.plot(history.history['val_acc'], 'r*-', label='Test acc')
    #plt.axvline(x=fine_tune_epoch, 'k--')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./output/training_plot.png')

    Test.main(IMAGES_DIR_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--split', help='percentage of validation data')
    args = parser.parse_args()

    main(**vars(args))
