import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import FlyGenerator
import settings
settings.init()

train_data_gen_args = dict(rescale=1. / 255)
'''
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=180,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True,
                            fill_mode='reflect')
'''

test_data_gen_args = dict(rescale=1. / 255)


# add functionality for splitting folders
def get_generators(image_dir, validation_pct=None):
    if validation_pct is None:
        return gen_from_folders(image_dir)
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
                             settings.class_dict['glomeruli']: settings.weight_glom}  # 1 : 25
    print("class weighting is : ", settings.class_weight)

    return train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES


def gen_from_folders(image_dir):
    """
        Generate image batches from train and test subdirectories

    :param image_dir:
    :return:
    """

    global train_data_gen_args, test_data_gen_args

    # folders with split data
    DIR_TRAIN_GLOM = image_dir + "/train/glomeruli"
    DIR_TEST_GLOM = image_dir + "/test/glomeruli"
    DIR_TRAIN_NONGLOM = image_dir + "/train/nonglomeruli"
    DIR_TEST_NONGLOM = image_dir + "/test/nonglomeruli"

    NUM_TRAIN_SAMPLES = len(os.listdir(DIR_TRAIN_GLOM)) + len(os.listdir(DIR_TRAIN_NONGLOM))
    NUM_TEST_SAMPLES = len(os.listdir(DIR_TEST_GLOM)) + len(os.listdir(DIR_TEST_NONGLOM))

    TRAIN_DIR_PATH = image_dir + "/train"
    TEST_DIR_PATH = image_dir + "/test"

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

    if train_generator.class_indices != validation_generator.class_indices:
        raise ValueError

    settings.class_dict = train_generator.class_indices
    print("label mapping is : ", settings.class_dict)
    settings.class_weight = {settings.class_dict['nonglomeruli']: 1,  # 0 : 1
                             settings.class_dict['glomeruli']: settings.weight_glom}  # 1 : 25
    print("class weighting is : ", settings.class_weight)

    #test_datagen.fit(train_generator)
    #train_datagen.fit(train_generator)

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
    parent_dir = os.path.dirname(image_dir)

    # folders with split data
    DIR_TRAIN_GLOM = parent_dir + "/split/train/glomeruli"
    DIR_TEST_GLOM = parent_dir + "/split/test/glomeruli"
    DIR_TRAIN_NONGLOM = parent_dir + "/split/train/nonglomeruli"
    DIR_TEST_NONGLOM = parent_dir + "/split/test/nonglomeruli"

    os.makedirs(DIR_TRAIN_GLOM, exist_ok=True)
    os.makedirs(DIR_TEST_GLOM, exist_ok=True)
    os.makedirs(DIR_TRAIN_NONGLOM, exist_ok=True)
    os.makedirs(DIR_TEST_NONGLOM, exist_ok=True)

    files_glom = os.listdir(DIR_GLOM)
    files_nonglom = os.listdir(DIR_NONGLOM)

    nb_files_glom = len(files_glom)
    nb_files_nonglom = len(files_nonglom)

    for f in files_glom:
        if np.random.rand(1) < validation_pct: # and nb_files_glom_test < nb_files_glom // validation_pct:
            shutil.copy(DIR_GLOM + '/' + f, DIR_TEST_GLOM + '/' + f)
        else:
            shutil.copy(DIR_GLOM + '/' + f, DIR_TRAIN_GLOM + '/' + f)

    for f in files_nonglom:
        if np.random.rand(1) < validation_pct:
            shutil.copy(DIR_NONGLOM + '/' + f, DIR_TEST_NONGLOM + '/' + f)
        else:
            shutil.copy(DIR_NONGLOM + '/' + f, DIR_TRAIN_NONGLOM + '/' + f)


if __name__ == '__main__':

    '''
        Split data in train and test 
    '''
    dir = "/Users/diana/Documents/2018_Glomeruli/data"
    #split_data_train_test_folders(image_dir=dir, validation_pct=10)

    '''
        Augment training data for glomeruli class
    '''
    GLOM_TRAIN_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/train/glomeruli/"
    AUGM_GLOM_TRAIN_DIR_PATH = "/Users/diana/Documents/2018_Glomeruli/split/train/augmented_glomeruli"

    target_size = (512, 512)

    files = os.listdir(GLOM_TRAIN_DIR_PATH)

    x_train_glom = []

    for f in files:
        if f == '.DS_Store':
            continue
        path = GLOM_TRAIN_DIR_PATH + f
        x = Image.open(path)
        x_resize = Image.Image.resize(x, target_size)
        x_resize = np.asarray(x_resize, dtype='float32')
        x_resize /= 255
        x_train_glom.append(x_resize)

    x_train_glom = np.stack(x_train_glom, axis=0)


    train_data_gen_args = dict(
                               rotation_range=90,
                               width_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               fill_mode='reflect')

    train_datagen = ImageDataGenerator(**train_data_gen_args)

    train_generator = train_datagen.flow(
                        x_train_glom,
                        save_to_dir=AUGM_GLOM_TRAIN_DIR_PATH)

    train_datagen.fit(train_generator, augment=True, rounds=1)

