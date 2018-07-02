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
Import MyInception
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

MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

FC_LAYER_SIZE = 1024

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./output/model.hdf5',
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
        callbacks=[checkpointer, early_stopper_dense, tensorboard, history_dense])

    model = load_model('./output/model.hdf5')

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
        callbacks=[checkpointer, early_stopper_finetune, tensorboard, history_finetune])

    # save model
    #model.save('./output/model.hdf5', overwrite=True)

    # save metrics during training epochs
    pd.DataFrame(history_dense.history).to_csv("./output/history_classifier.csv")
    pd.DataFrame(history_finetune.history).to_csv("./output/history_finetune.csv")

    fine_tune_epoch = len(history_dense.history['loss'])
    print('Epoch when fine-tuning starts: %d' % fine_tune_epoch)

    '''
    dict_data = {key: {key_ + fine_tune_epoch : val_ for key_, val_ in val.items()}
                 for key, val in dict_data.items()}
                 
    finaldict = {key:(dict1[key], dict2[key]) for key in dict1}
    '''
    history = _util.merge_history_dicts(history_dense.history, history_finetune.history, fine_tune_epoch)

    '''
    # plot  metrics during training epochs
    plt.style.use('seaborn-notebook')
    plt.plot(history_dense.history['loss'], 'go-', label='Train loss')
    plt.plot(history_dense.history['val_loss'], 'ro-', label='Test loss')
    plt.plot(history_dense.history['acc'], 'g*-', label='Train acc')
    plt.plot(history_dense.history['val_acc'], 'r*-', label='Test acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./output/training_plot_classifier.png')
    '''

    plt.style.use('seaborn-notebook')
    plt.plot(history['loss'], 'go-', label='Train loss')
    plt.plot(history['val_loss'], 'ro-', label='Test loss')
    plt.plot(history['acc'], 'g*-', label='Train acc')
    plt.plot(history['val_acc'], 'r*-', label='Test acc')
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
