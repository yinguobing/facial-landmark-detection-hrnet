import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from models.hrnet import HRNetBody, FusionBlock


class HRNetStem(layers.Layer):

    def __init__(self, filters=64, **kwargs):
        super(HRNetStem, self).__init__(**kwargs)

        # The stem of the network.
        self.conv_1 = layers.Conv2D(filters, 3, 2, 'same')
        self.batch_norm_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(filters, 3, 2, 'same')
        self.batch_norm_2 = layers.BatchNormalization()
        self.activation = layers.Activation('relu')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)

        return x


class HRNetTail(layers.Layer):

    def __init__(self, filters=64, **kwargs):
        super(HRNetTail, self).__init__(**kwargs)

        # TODO: Implementation tail part.

    def call(self, inputs):
        # TODO: Implementation tail part.

        return x


class HRNetV2(Model):

    def __init__(self, width=18, **kwargs):
        super(HRNetV2, self).__init__(**kwargs)

        self.stem = HRNetStem(64)
        self.body = HRNetBody(width)
        self.tail = FusionBlock(15*width, 4, 1)

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.body(x)
        x = self.tail(x)

        return x


if __name__ == "__main__":
    model = HRNetV2(18)
    model(keras.Input((256, 256, 3)))
    model.summary()
