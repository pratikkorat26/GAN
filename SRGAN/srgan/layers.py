import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

print("in layers")

class residual_block(keras.Model):
    def __init__(self ,filters ,  kernel_size,strides , padding = "same"):
        super(residual_block , self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.filters = filters

        self.conv1 = Conv2D(filters = filters,
                            kernel_size = kernel_size,
                            strides = strides ,
                            padding = padding ,
                            kernel_initializer = "glorot_uniform")
        self.conv2 = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer="glorot_uniform")
        self.layernorm = LayerNormalization(epsilon = 1e-5)
        self.activation = PReLU()


    def call(self , x):

        """
        inputs : images
        :param x:
        :return: calculate tensorf from residual block
        """
        inp = self.conv1(x)
        inp = self.layernorm(inp)
        inp = self.activation(inp)
        inp = self.conv2(inp)
        inp = self.layernorm(inp)
        #shape same as input
        #Element wise sum
        skip_connection = Add()([x , inp])

        return skip_connection


class Residual(keras.Model):
    def __init__(self , filters , kernel_size , strides , padding = "same"):
        super(Residual , self).__init__()

        self.residual_block = residual_block(filters , kernel_size , strides , padding)

    def call(self, x):
        """
        inputs : image tensor
        :param x:
        :return: output tenosr calculated from big residual blocks
        """
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)

        return x


class PixelShuffler2x(keras.layers.Layer):
    def __init__(self):
        super(PixelShuffler2x,self).__init__()
        #Suppose input ==> (32 , 64 , 64 , 128) =====> output ==> (32 , 128 , 128 ,32)

    def call(self , x):
        """
        PixelShuffler2x
        :param x: input ==> (batch , height , width , depth)
        :return: output ==> (batch , height*2 , width*2 , depth//4)
        """
        assert len(x.shape) == 4
        reshaped = tf.nn.depth_to_space(x , block_size = 2)
        print(reshaped.shape)
        return reshaped

if __name__ == "__main__":
    pass

