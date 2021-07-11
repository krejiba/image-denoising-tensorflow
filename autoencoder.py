import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D

# Sets the seed of the random number generator
tf.random.set_seed(10)


class AutoEncoder(tf.keras.Model):

    def __init__(self, input_shape):

        super(AutoEncoder, self).__init__(name='Autoencoder')
        self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             activation='relu', name='convolution2d_1', input_shape=input_shape[1:])
        self.pool_1 = MaxPool2D(pool_size=(2, 2), name='maxpooling2d_1')
        self.conv_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             activation='relu', name='convolution2d_2')
        self.pool_2 = MaxPool2D(pool_size=(2, 2), name='maxpooling2d_2')
        self.conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             activation='relu', name='convolution2d_3')
        self.upsampl_1 = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsampling2d_1')
        self.conv_4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             activation='relu', name='convolution2d_4')
        self.upsampl_2 = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsampling2d_2')
        self.conv_5 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             activation='sigmoid', name='convolution2d_5')

    def call(self, inputs):
        # Encoder
        x = self.conv_1(inputs)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        # Decoder
        x = self.upsampl_1(x)
        x = self.conv_4(x)
        x = self.upsampl_2(x)
        y = self.conv_5(x)

        return y
