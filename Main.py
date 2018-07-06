import argparse
import pandas as pd

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.utils import multi_gpu_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import MyModel
import MyInception
from  MyMetrics import precision, recall, sensitivity, specificity, f1_score
import Data
import Test

import settings
settings.init()

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./output/model.hdf5',
    verbose=1,
    monitor='val_f1_score',
    mode='max',
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./output/events')

# Helper: Keep track of acc and loss during training
history = History()

fine_tune_epoch = 0

def train_inception(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES ):
    global fine_tune_epoch

    model = MyInception.get_model(settings.NUM_CLASSES)

    print("Training dense classifier from scratch")
    # Get and train the top layers.
    model = MyInception.get_top_layer_model(model)

    model.summary()

    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.DENSE_TRAIN_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=[checkpointer, early_stopper, tensorboard, history])

    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})


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
                       custom_objects={'precision': precision, 'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})
    return model


def train_mymodel(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES ):

    model = MyModel.get_model(settings.NUM_CLASSES)

    model.summary()

    model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // settings.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=NUM_TEST_SAMPLES // settings.BATCH_SIZE,
        epochs=settings.FINE_TUNE_EPOCHS,
        class_weight=settings.class_weight,
        callbacks=[checkpointer, early_stopper, tensorboard, history])

    model = load_model('./output/model.hdf5',
                       custom_objects={'precision': precision,'recall': recall, 'sensitivity': sensitivity,
                                       'specificity': specificity, 'f1_score': f1_score})

    return model


def main(dir=None, split=None, model=None, train=None, test=None):
    global fine_tune_epoch

    if dir == None:
        IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/"
    else:
        IMAGES_DIR_PATH = dir

    if split == None:
        '''
            If data is already split in train / test subfolders
        '''
        VALIDATION_SPLIT = None
    else:
        VALIDATION_SPLIT = split

    train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES = Data.get_generators(IMAGES_DIR_PATH, VALIDATION_SPLIT)

    model = train_inception(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    #model = train_mymodel(train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES)

    # save metrics during training epochs
    pd.DataFrame(history.history).to_csv("./output/history.csv")

    # plot  metrics during training epochs
    plt.style.use('seaborn-notebook')
    plt.plot(history.history['loss'], 'go--', label='Train loss')
    plt.plot(history.history['val_loss'], 'ro--', label='Test loss')
    plt.axvline(x=fine_tune_epoch)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./output/training_plot.png')

    Test.main(IMAGES_DIR_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--split', help='percentage of validation data')
    parser.add_argument('--model', help='InceptionV3 / Small CNN')
    parser.add_argument('--train', help='is training')
    parser.add_argument('--test', help='is testing')
    args = parser.parse_args()

    main(**vars(args))
