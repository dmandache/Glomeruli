from keras.models import load_model
import os
import random
import numpy as np


from Globals import *

model = load_model('./output/model.hdf5')

nb_samples = 32

x_test = []

files_test_glom = os.listdir(DIR_TEST_GLOM)

for _ in range(nb_samples):
    # Select random element
    random_file = random.choice(files_test_glom)
    x_test.append()


x_test = x_test.astype('float32')
x_test /= 255
x_test = np.expand_dims(x_test, axis=3)