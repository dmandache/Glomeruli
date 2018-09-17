import argparse
import os
import pandas as pd

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Models.Metrics import precision, recall, sensitivity, specificity, f1_score
import Data.GetData as Data
from Models import SoA, Tiny
import Test

import settings


fine_tune_epoch = 0


def main(dir=None, split=None, out=None, model=None, finetune=False, transfer=False):
    global fine_tune_epoch

    model = 'resnet'
    transfer = True

    if dir == None:
        settings.IMAGES_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/"
    else:
        settings.IMAGES_DIR_PATH = dir

    if split == None:
        '''
            Data is already split in train / test subfolders
        '''
        settings.VALIDATION_SPLIT = None
    else:
        '''
            Randomly split data in train / test 
            A list of filenames with the images chosen for train and test will be displayed
        '''
        settings.VALIDATION_SPLIT = split

    if out == None:
        settings.OUTPUT_DIR = './output_{}'.format(model)
        if finetune == True:
            settings.OUTPUT_DIR += '_finetuned'
    else:
        settings.OUTPUT_DIR = out
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    '''
        Default model input size
    '''
    if model == 'inception':
        settings.MODEL_INPUT_WIDTH = 299
        settings.MODEL_INPUT_HEIGHT = 299
        settings.MODEL_INPUT_DEPTH = 3
    elif model == 'vgg' or model == 'resnet':
        settings.MODEL_INPUT_WIDTH = 224
        settings.MODEL_INPUT_HEIGHT = 224
        settings.MODEL_INPUT_DEPTH = 3


    '''
        Get data generators
    '''

    train_generator, validation_generator = Data.get_generators(settings.IMAGES_DIR_PATH, settings.VALIDATION_SPLIT)

    '''
        Callback functions
    '''

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=settings.OUTPUT_DIR+'/model.hdf5',
        verbose=1,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=100)

    # Helper: TensorBoard
    tensorboard = TensorBoard(log_dir=settings.OUTPUT_DIR+'/tensorboard')

    callback_list = [checkpointer, early_stopper, tensorboard]

    '''
        Train model
    '''

    supported_models = ['inception','vgg','resnet','tiny']
    if model not in supported_models:
        print("Invalid model ! Please specify one of this arguments : {}".format(supported_models))
        exit()

    if model == 'tiny':
        model, history = Tiny.train(train_generator, validation_generator, callback_list)
    else:
        if transfer:
            model, history = SoA.train_transfer(model, train_generator, validation_generator, callback_list)
        elif finetune:
            model, history = SoA.train_finetune(model, train_generator, validation_generator, callback_list)
        else:
            model, history = SoA.train_from_scratch(model, train_generator, validation_generator, callback_list)

    '''
        Save metrics
    '''

    # save metrics during training epochs
    pd.DataFrame(history.history).to_csv(settings.OUTPUT_DIR+'/history.csv')

    # plot  metrics during training epochs
    plt.style.use('seaborn')
    plt.plot(history.history['loss'], 'g.-', label='Train loss')
    plt.plot(history.history['val_loss'], 'r.-', label='Test loss')
    #plt.axvline(x=fine_tune_epoch, color='gray', linestyle='dashed')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(settings.OUTPUT_DIR+'/training_plot.png')

    #Test.main(IMAGES_DIR_PATH + '/test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='data directory')
    parser.add_argument('--split', help='percentage of validation data, usually 20')
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--model', help='inception / vgg / resnet / tiny')
    parser.add_argument('--finetune', dest='finetune', default=False, action='store_true')
    parser.add_argument('--transfer', dest='transfer', default=False, action='store_true')
    args = parser.parse_args()

    main(**vars(args))
