from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras import backend as K
import MyMetrics

import settings
settings.init_globals()

# with img size (288, 288, 3) it has 7.6M parameters
def get_model(num_classes=1):
    """
        Build a basic CNN with 4 convolutional blocks = 8 convolutonal layers

        Parameters
        ----------
        input_shape : (batch_size, width, height, ch)
            width, height must to be a multiple of 2^4 = 16
        num_classes : number of classes

        Returns
        -------
        model

    """
    if K.image_data_format() == 'channels_first':
        input_shape = (settings.MODEL_INPUT_DEPTH, settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT)
    else:
        input_shape = (settings.MODEL_INPUT_WIDTH, settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_DEPTH)

    model = Sequential()

    '''
            First convolutional block 32 x 2 filters 7x7, maxpooling 2x2, dropout 25%
    '''
    model.add(Conv2D(32, (7, 7), border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '''
            Second convolutional block 32 x 2 filters 5x5, maxpooling 2x2, dropout 25%
    '''
    model.add(Conv2D(32, (5, 5), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '''
            Third convolutional block 64 x 2 filters 3x3, maxpooling 2x2, dropout 25%
    '''
    model.add(Conv2D(64, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '''
            4th convolutional block 64 x 2 filters 3x3, maxpooling 2x2, dropout 25%
    '''
    model.add(Conv2D(128, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '''
            Classifier : fully connected layer 512 neurons, dropout 50%
    '''
    model.add(Flatten())
    model.add(Dense(1024, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, init='glorot_uniform'))
    model.add(Activation('sigmoid'))

    if num_classes is 1:
        loss_function = 'binary_crossentropy'
    else:
        loss_function = 'categorical_crossentropy'

    '''
        Adam optimization algo
            amsgrad=True - ON THE CONVERGENCE OF ADAM AND BEYOND - https://openreview.net/pdf?id=ryQu7f-RZ
    '''
    adam = optimizers.Adam(lr=0.01, decay=1e-4, amsgrad=False)
    rmsprop = optimizers.RMSprop(lr=0.001, decay=1e-5)

    model.compile(optimizer=adam, loss=loss_function,
                  metrics=['accuracy', MyMetrics.sensitivity, MyMetrics.specificity, MyMetrics.f1_score])

    return model