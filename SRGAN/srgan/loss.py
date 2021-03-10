import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import *


cross_entropy = keras.losses.BinaryCrossentropy(from_logits = True)


def discriminator_loss(y_true, y_pred):
    """
    :param y_true: true images predicted from discriminator
    :param y_pred: fake images generated from generator is through from discriminator

    y_pred == generator(LR) --> discriminator(LR and 1 or 0) --> discriminator_loss
    :return: combined loss
    """
    y_pred_label = tf.zeros_like(y_pred)
    y_true_label = tf.ones_like(y_true)

    real_loss = cross_entropy(y_true_label , y_true)
    fake_loss = cross_entropy(y_pred_label , y_pred)

    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(y_pred , y_true):
    """
    :param y_pred: true images
    :param y_true: generated images
    :return: combined loss , content loss and binary cross entropy
    """

    mse_loss = keras.losses.MSE(y_true , y_pred)
    binary_loss = cross_entropy(y_true , y_pred)

    total_loss = mse_loss + binary_loss

    return total_loss




if __name__ == "__main__":

    img = np.random.rand(1 , 255 , 255 , 3)
    fake = np.random.rand(1 , 255 , 255 , 3)

    loss = discriminator_loss(img , fake)

    print(loss)


