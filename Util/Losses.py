from keras import backend as K
import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    """
        This is the keras implementation of focal loss proposed by
        Lin et. al. in their Focal Loss for Dense Object Detection paper.

        usage : my_model.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)])
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def expectation_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred), axis=-1)     # L1 norm


def normalized_expectation_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred), axis=-1)     # sum of vector elements / number of elements


def regularized_expectation_loss(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=-1)    # L2 norm


def normalized_regularized_expectation_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)    # sum of squares / number of elements


def categorical_expectation_loss(y_true, y_pred):
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate expectatin loss
    loss = expectation_loss(y_true, y_pred)
    # Sum the losses
    return K.sum(loss, axis=-1)


def normalized_categorical_expectation_loss(y_true, y_pred):
    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    # Calculate Loss
    loss = normalized_expectation_loss(y_true, y_pred)
    # Sum the losses
    return K.mean(loss, axis=-1)


if __name__ == '__main__':
    '''
        Binary case
    '''
    print('*******Binary Case')
    y_true_vector = [0,    0,    1,    1,    1]
    y_pred_vector = [0.75, 0.01, 0.20, 0.52, 0.63]

    y_true_tensor = K.variable(value=y_true_vector)
    y_pred_tensor = K.variable(value=y_pred_vector)

    print("y true", K.eval(y_true_tensor))
    print("y pred", K.eval(y_pred_tensor))


    print("\texpectation loss ", K.eval(expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized expectation loss ", K.eval(normalized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tregularized expectation loss ", K.eval(regularized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnorm reg expectation loss ", K.eval(normalized_regularized_expectation_loss(y_true_tensor, y_pred_tensor)))

    '''
        Categorical case
    '''
    print('\n*******Categorical Case')
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


    print("\texpectation loss ", K.eval(expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized expectation loss ", K.eval(normalized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tregularized expectation loss ", K.eval(regularized_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnorm reg expectation loss ", K.eval(normalized_regularized_expectation_loss(y_true_tensor, y_pred_tensor)))

    print("\n\tcategorical expectation loss ", K.eval(categorical_expectation_loss(y_true_tensor, y_pred_tensor)))
    print("\tnormalized categorical expectation loss ", K.eval(normalized_categorical_expectation_loss(y_true_tensor, y_pred_tensor)))







