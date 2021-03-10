import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from srgan import layers

#----------------------------------------------------------------------------------------------------------------#
#---------------------------------------------GENERATOR----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
def generator(inputs_shape ,num_residual_layers):
    """
    inputs : tuples of input shape ex ==> (64 , 64 , 3) or something
    :param inputs_shape:
    :return: keras model for generator
    """

    inputs = layers.Input(shape = inputs_shape)

    #----------------------------------------------------------------------------#
    # First conv block with relu , residual                                      #
    #----------------------------------------------------------------------------#
    with tf.compat.v1.variable_scope(name_or_scope = "FirstConvBlockAndPRelu"):
        conv1 = layers.Conv2D(filters = 32,
                          kernel_size = 4,
                          strides = 1 ,
                          padding = "same")(inputs)
        conv1 = layers.PReLU()(conv1)
    with tf.compat.v1.variable_scope(name_or_scope = "SecondConvBlockAndPRelu"):
        conv2 = layers.Conv2D(filters=32,
                          kernel_size=4,
                          strides=1,
                          padding="same")(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.PReLU()(conv2)

    #-----------------------------------------------------------------------------#
    #  B Residuals blocks                                                         #
    #-----------------------------------------------------------------------------#

    for i in range(num_residual_layers):
        with tf.compat.v1.variable_scope(name_or_scope = f"Residual_{i}"):
            x = layers.Residual(filters = 32,
                     kernel_size = (3,3),
                     strides = 1,
                     padding = "same")(conv2)

        conv2 = x

    #-----------------------------------------------------------------------------#
    #  AFTER WORD RESIDUAL BLOCK                                                  #
    #-----------------------------------------------------------------------------#
    with tf.compat.v1.variable_scope(name_or_scope = "FinalConvBlockAndPRelu"):
        conv2 = layers.Conv2D(filters=32,
                          kernel_size=3,
                          strides= 1,
                          padding="same",
                          activation="relu")(conv2)
        conv2 = layers.BatchNormalization()(conv2)

        conv2 = layers.Add()([conv1 , conv2])

    # -----------------------------------------------------------------------------#
    #  FINAL RESIDUAL BLOCK                                                        #
    # -----------------------------------------------------------------------------#

    with tf.compat.v1.variable_scope(name_or_scope = "FirstPixelShufflerAndPrelu"):
        conv2 = layers.Conv2D(filters=64,
                          kernel_size=3,
                          strides=1,
                          padding="same")(conv2)
        conv2 = layers.PixelShuffler2x()(conv2)
        conv2 = layers.PReLU()(conv2)
    #Output ==> (batch , 128 , 128 , 32)
    with tf.compat.v1.variable_scope(name_or_scope="SecondPixelShufflerAndPrelu"):
        conv2 = layers.Conv2D(filters=64,
                          kernel_size=3,
                          strides=1,
                          padding="same")(conv2)
        conv2 = layers.PixelShuffler2x()(conv2)
        conv2 = layers.PReLU()(conv2)

    #Final layer:
    with tf.compat.v1.variable_scope(name_or_scope="FinalConv2D"):
        conv2 = layers.Conv2D(filters=3,
                          kernel_size=3,
                          strides=1,
                          padding="same")(conv2)

    with tf.compat.v1.variable_scope(name_or_scope="generator"):
        model = Model(inputs , conv2 , name = 'generator')

    return model



#----------------------------------------------------------------------------------------------------------------#
#---------------------------------------- DISCRIMINATOR----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

def discriminator(input_shape , num_residual_layers):
    """
    inputs : input shape for model's prediction
    :param input_shape: input image shape
    :return: model discriminator
    """
    inputs = layers.Input(shape=input_shape)

    #------------------------------------------------------------------------------------------------------------#
    #   model's pre part convolution #
    #------------------------------------------------------------------------------------------------------------#

    conv_1 = Conv2D( filters = 64 ,
                     kernel_size = (4,4),
                     padding = "valid" ,
                     strides = 2)(inputs)
    conv_1 = LeakyReLU(alpha = 0.3)(conv_1)


    #-------------------------------------------------------------------------------------------------------------#
    #    model's second convolution parts  #
    #-------------------------------------------------------------------------------------------------------------#

    conv_1 = Conv2D(filters=64,
                    kernel_size=(4, 4),
                    padding="valid",
                    strides=2)(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = LeakyReLU(alpha=0.3)(conv_1)

    #-------------------------------------------------------------------------------------------------------------#
    #   model's residual part
    #-------------------------------------------------------------------------------------------------------------#

    for i in range(num_residual_layers):
        x = layers.Residual(filters = 64,
                     kernel_size = (3,3),
                     strides = 1,
                     padding = "same")(conv_1)

        conv_1 = x


    #--------------------------------------------------------------------------------------------------------------#
    # model's final part #
    #--------------------------------------------------------------------------------------------------------------#
    conv_1 = Conv2D(filters=16,
                    kernel_size=(4, 4),
                    padding="valid",
                    strides=2)(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = LeakyReLU(alpha=0.3)(conv_1)
    conv_1 = AveragePooling2D()(conv_1)

    conv_1 = Flatten()(conv_1)

    conv_1 = Dense(units = 1 , activation = "sigmoid")(conv_1)

    model = Model(inputs , conv_1 , name = "discriminator")

    return model



if __name__ == "__main__":

    model = generator((64 , 64 , 3) , 6)
    model.summary()


