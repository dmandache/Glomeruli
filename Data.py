import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import FlyGenerator
import settings
settings.init()

train_data_gen_args = dict(rescale=1. / 255,
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            zca_whitening=True,
                            rotation_range=180,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            fill_mode='reflect')

test_data_gen_args = dict(rescale=1. / 255,
                        featurewise_center=True,
                        featurewise_std_normalization=True,
                        zca_whitening=True)


# add functionality for splitting folders
def get_generators(image_dir, validation_pct=None):
    return gen_from_image_lists(image_dir, validation_pct)

def gen_from_image_lists(image_dir, validation_pct=20):
    """
        Generate image batches for training and validation without physically splitting it on disk

    :param image_dir:
    :param validation_pct: integer between (0, 100)
    :return:
    """

    global train_data_gen_args, test_data_gen_args

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
        # save_to_dir='./output/augmented_data',
        target_size=(settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH),
        batch_size=settings.BATCH_SIZE,
        class_mode=settings.CLASS_MODE,
        seed=settings.RANDOM_SEED)

    validation_generator = test_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='validation',
        image_dir=image_dir,
        target_size=(settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH),
        batch_size=settings.BATCH_SIZE,
        class_mode=settings.CLASS_MODE,
        seed=settings.RANDOM_SEED)

    '''
        modify global variables if necessary: class label association, class weight matrix respectively
    '''
    settings.class_dict = train_generator.class2id
    print("label mapping is : ", settings.class_dict)
    settings.class_weight = {settings.class_dict['nonglomeruli']: 1,  # 0 : 1
                             settings.class_dict['glomeruli']: 25}  # 1 : 25
    print("class weighting is : ", settings.class_weight)

    return train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES


def gen_from_folders(image_dir):
    """
        Generate image batches from train and test subdirectories

    :param image_dir:
    :return:
    """

    global train_data_gen_args, test_data_gen_args

    # folders with unsplit data
    DIR_GLOM = image_dir + "/glomeruli"
    DIR_NONGLOM = image_dir + "/nonglomeruli"

    # go up one folder
    parent_dir = os.pardir(image_dir)

    # folders with split data
    DIR_TRAIN_GLOM = parent_dir + "split/train/glomeruli"
    DIR_TEST_GLOM = parent_dir + "split/test/glomeruli"
    DIR_TRAIN_NONGLOM = parent_dir + "split/train/nonglomeruli"
    DIR_TEST_NONGLOM = parent_dir + "split/test/nonglomeruli"

    NUM_TRAIN_SAMPLES = len(os.listdir(DIR_TRAIN_GLOM)) + len(os.listdir(DIR_TRAIN_NONGLOM))
    NUM_TEST_SAMPLES = len(os.listdir(DIR_TEST_GLOM)) + len(os.listdir(DIR_TEST_NONGLOM))

    TRAIN_DIR_PATH = image_dir + "split/train"
    TEST_DIR_PATH = image_dir + "split/test"

    train_datagen = ImageDataGenerator(**train_data_gen_args)

    test_datagen = ImageDataGenerator(**test_data_gen_args)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR_PATH,
        target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT),
        batch_size=settings.BATCH_SIZE,
        class_mode=settings.CLASS_MODE)

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR_PATH,
        target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT),
        batch_size=settings.BATCH_SIZE,
        class_mode=settings.CLASS_MODE)

    test_datagen.fit(train_generator, nb_iter=100)

    train_datagen.fit_generator(train_generator, nb_iter=100)

    return train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES


def split_data_train_test_folders(image_dir, validation_pct=20):
    """
        Create train and test subdirectories for every class folder found in image_dir

    :param image_dir: folder with data spit in classes
    :param validation_pct: integer between (0, 100)
    :return:

        +---image_dir
        |   \---glomeruli
        |   \---nonglomeruli
        +---split
        |   +---train
        |       \---glomeruli
        |       \---nonglomeruli
        |   +---test
        |       \---glomeruli
        |       \---nonglomeruli

    """
    validation_pct /= 100           # validation_pct in interval (0, 1)

    # folders with unsplit data
    DIR_GLOM = image_dir + "/glomeruli"
    DIR_NONGLOM = image_dir + "/nonglomeruli"

    # go up one folder
    parent_dir = os.pardir(image_dir)

    # folders with split data
    DIR_TRAIN_GLOM = parent_dir + "split/train/glomeruli"
    DIR_TEST_GLOM = parent_dir + "split/test/glomeruli"
    DIR_TRAIN_NONGLOM = parent_dir + "split/train/nonglomeruli"
    DIR_TEST_NONGLOM = parent_dir + "split/test/nonglomeruli"

    files_glom = os.listdir(DIR_GLOM)
    files_nonglom = os.listdir(DIR_NONGLOM)

    for f in files_glom:
        if np.random.rand(1) < validation_pct:
            shutil.copy(DIR_GLOM + '/' + f, DIR_TEST_GLOM + '/' + f)
        else:
            shutil.copy(DIR_GLOM + '/' + f, DIR_TRAIN_GLOM + '/' + f)

    for f in files_nonglom:
        if np.random.rand(1) < validation_pct:
            shutil.copy(DIR_NONGLOM + '/' + f, DIR_TEST_NONGLOM + '/' + f)
        else:
            shutil.copy(DIR_NONGLOM + '/' + f, DIR_TRAIN_NONGLOM + '/' + f)
