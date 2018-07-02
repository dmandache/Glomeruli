
def init_globals():
    global class_dict, class_weight
    global DENSE_TRAIN_EPOCHS, FINE_TUNE_EPOCHS, NUM_CLASSES, CLASS_MODE, RANDOM_SEED
    global BATCH_SIZE, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_DEPTH

    class_dict = {'nonglomeruli': 0, 'glomeruli': 1}

    class_weight = {class_dict['nonglomeruli']: 1,  # 0 : 1
                    class_dict['glomeruli']: 25}  # 1 : 25

    DENSE_TRAIN_EPOCHS = 30
    FINE_TUNE_EPOCHS = 500

    NUM_CLASSES = 1
    RANDOM_SEED = None
    CLASS_MODE = 'binary'

    # we chose to train the top 2 inception blocks
    BATCH_SIZE = 150

    MODEL_INPUT_WIDTH = 288
    MODEL_INPUT_HEIGHT = 288
    MODEL_INPUT_DEPTH = 3