from Models.Metrics import precision, recall, sensitivity, specificity, f1_score

class_dict = {'nonglomeruli': 0, 'glomeruli': 1}

weight_glom = 1

class_weight = {class_dict['nonglomeruli']: 1,  # 0 : 1
                class_dict['glomeruli']: weight_glom}  # 1 : 25

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

CUSTOM_OBJECTS = {'precision': precision, 'recall': recall,
                  'sensitivity': sensitivity, 'specificity': specificity,
                  'f1_score': f1_score}