
def init():
    global class_dict, class_weight, weight_glom
    global DENSE_TRAIN_EPOCHS, FINE_TUNE_EPOCHS, NUM_CLASSES, CLASS_MODE, RANDOM_SEED
    global BATCH_SIZE, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_DEPTH

    class_dict = {'nonglomeruli': 0, 'glomeruli': 1}

    weight_glom = 5

    class_weight = {class_dict['nonglomeruli']: 1,  # 0 : 1
                    class_dict['glomeruli']: weight_glom}  # 1 : 25

    DENSE_TRAIN_EPOCHS = 30
    FINE_TUNE_EPOCHS = 500

    NUM_CLASSES = 1
    RANDOM_SEED = None
    CLASS_MODE = 'binary'

    BATCH_SIZE = 150

    MODEL_INPUT_WIDTH = 288
    MODEL_INPUT_HEIGHT = 288
    MODEL_INPUT_DEPTH = 3