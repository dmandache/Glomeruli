import os

from keras.preprocessing.image import ImageDataGenerator

import FlyGenerator

import settings
settings.init_globals()

def get_generators(image_dir, validation_pct=None):
    #global class_dict, class_weight
    #global MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, BATCH_SIZE, CLASS_MODE, RANDOM_SEED

    train_data_gen_args = dict(rescale=1. / 255,
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
            target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT),
            batch_size=settings.BATCH_SIZE,
            class_mode=settings.CLASS_MODE)

        # this is a similar generator, for validation data
        validation_generator = test_datagen.flow_from_directory(
            TEST_DIR_PATH,
            target_size=(settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT),
            batch_size=settings.BATCH_SIZE,
            class_mode=settings.CLASS_MODE)
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

        # modify global variables if necessary

        settings.class_dict = train_generator.class2id
        print("label mapping is : ", settings.class_dict)

        settings.class_weight = {settings.class_dict['nonglomeruli']: 1,  # 0 : 1
                                 settings.class_dict['glomeruli']: 25}  # 1 : 25
        print("class weighting is : ", settings.class_weight)

    return train_generator, validation_generator, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES