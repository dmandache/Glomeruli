from keras import backend as K
import tensorflow as tf


"""
    FOCAL LOSS
"""


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # Sum the losses in mini_batch
        return K.sum(loss)

    return categorical_focal_loss_fixed


"""
    EXPECTATION LOSS
"""

# Binary
def binary_expectation_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred), axis=-1)     # L1 norm


def binary_expectation_loss_normalized(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred), axis=-1)     # sum of vector elements / number of elements


def binary_regularized_expectation_loss(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=-1)    # L2 norm


def binary_regularized_expectation_loss_normalized(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)    # sum of squares / number of elements

#   Categorical
def categorical_expectation_loss(y_true, y_pred):
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate expectatin loss
    loss = binary_expectation_loss(y_true, y_pred)
    # Sum the losses
    return K.sum(loss, axis=-1)


def normalized_categorical_expectation_loss(y_true, y_pred):
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate Loss
    loss = binary_expectation_loss_normalized(y_true, y_pred)
    # Sum the losses
    return K.mean(loss, axis=-1)

'''
    Test functions
'''
if __name__ == '__main__':
    '''
        Binary case
    '''
    print('*******Binary Case *******\n')
    y_true_vector = [0,    0,    1,    1,    1]
    y_pred_vector = [0.75, 0.01, 0.20, 0.52, 0.63]

    y_true_tensor = K.variable(value=y_true_vector)
    y_pred_tensor = K.variable(value=y_pred_vector)

    print("y true", K.eval(y_true_tensor))
    print("y pred", K.eval(y_pred_tensor))


    print("\n\texpectation loss ", K.eval(binary_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized expectation loss ", K.eval(binary_expectation_loss_normalized(y_true_tensor, y_pred_tensor)))
    print("\tregularized expectation loss ", K.eval(binary_regularized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnorm reg expectation loss ", K.eval(binary_regularized_expectation_loss_normalized(y_true_tensor, y_pred_tensor)))

    print("\n\tfocal loss ", K.eval(binary_focal_loss(alpha=.25, gamma=2)(y_true_tensor, y_pred_tensor)))

    '''
        Categorical case
    '''
    print('\n******* Categorical Case *******\n')
    y_true_vector = [[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]]
    y_pred_vector = [[0.25, 0.75],
                     [0.99, 0.01],
                     [0.80, 0.20],
                     [0.48, 0.52],
                     [0.37, 0.63]]

    y_true_tensor = K.variable(value=y_true_vector)
    y_pred_tensor = K.variable(value=y_pred_vector)

    print("y true", K.eval(y_true_tensor))
    print("y pred", K.eval(y_pred_tensor))

    print("\n\texpectation loss ", K.eval(binary_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized expectation loss ", K.eval(binary_expectation_loss_normalized(y_true_tensor, y_pred_tensor)))
    print("\tregularized expectation loss ", K.eval(binary_regularized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnorm reg expectation loss ", K.eval(binary_regularized_expectation_loss_normalized(y_true_tensor, y_pred_tensor)))

    print("\n\tcategorical expectation loss ", K.eval(categorical_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized categorical expectation loss ", K.eval(normalized_categorical_expectation_loss(y_true_tensor, y_pred_tensor)))

    print("\n\tcategorical focal loss ", K.eval(categorical_focal_loss(alpha=.25, gamma=2)(y_true_tensor, y_pred_tensor)))
