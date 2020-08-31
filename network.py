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

    def __init__(self, input_channels=64, output_channels=17, **kwargs):
        super(HRNetTail, self).__init__(**kwargs)

        # Up sampling layers.
        scales = [2, 4, 8]
        self.up_scale_layers = [layers.UpSampling2D((s, s)) for s in scales]
        self.concatenate = layers.Concatenate()
        self.conv_1 = layers.Conv2D(filters=input_channels, kernel_size=(1, 1),
                                    strides=(1, 1), padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.conv_2 = layers.Conv2D(filters=output_channels, kernel_size=(1, 1),
                                    strides=(1, 1), padding='same')

    def call(self, inputs):
        inputs[1:] = [f(x) for f, x in zip(self.up_scale_layers, inputs[1:])]
        x = self.concatenate(inputs)
        x = self.conv_1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv_2(x)

        return x


class HRNetV2(Model):

    def __init__(self, width=18, output_channels=98, **kwargs):
        super(HRNetV2, self).__init__(**kwargs)

        self.stem = HRNetStem(64)
        self.body = HRNetBody(width)
        last_stage_width = sum([width * pow(2, n) for n in range(4)])
        self.tail = HRNetTail(input_channels=last_stage_width,
                              output_channels=output_channels)

    def call(self, inputs):
        x = self.stem(inputs)
        x = self.body(x)
        x = self.tail(x)

        return x


if __name__ == "__main__":
    model = HRNetV2(18)
    model(keras.Input((256, 256, 3)))
    model.summary()
