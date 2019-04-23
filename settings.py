from Models import Losses, Metrics

class_dict = {'nonglomeruli': 0, 'glomeruli': 1}

weight_glom = 1

class_weight = {class_dict['nonglomeruli']: 1,          # 0 : 1
                class_dict['glomeruli']: weight_glom}   # 1 : 25

DENSE_TRAIN_EPOCHS = 50
FINE_TUNE_EPOCHS = 200

NUM_CLASSES = 1
RANDOM_SEED = None
CLASS_MODE = 'binary'

BATCH_SIZE = 64

# default size will be changed according to the model chosen
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3

OUTPUT_DIR = './output'

# will be given valuess in code
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES = 0

CUSTOM_OBJECTS = {'precision': Metrics.precision, 'recall': Metrics.recall,
                  'sensitivity': Metrics.sensitivity, 'specificity': Metrics.specificity,
                  'f1_score': Metrics.f1_score}

losses_whitelist = ['crossentropy', 'expectation', 'focal']

LOSS = 'crossentropy'

model_whitelist = ['inception', 'vgg', 'resnet', 'tiny']


def get_loss_function():
    if LOSS == 'crossentropy':
        loss_function = 'binary_crossentropy' if NUM_CLASSES is 1 else 'categorical_crossentropy'
    elif LOSS == 'focal':
        loss_function = Losses.binary_focal_loss() if NUM_CLASSES is 1 else Losses.categorical_focal_loss()
    elif LOSS == 'expectation':
        loss_function = Losses.binary_expectation_loss_normalized() if NUM_CLASSES is 1 else Losses.normalized_categorical_expectation_loss()
    return loss_function