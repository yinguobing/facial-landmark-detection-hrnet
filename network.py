import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from models.hrnet import HRNetBody, FusionBlock


class HRNetStem(layers.Layer):

    def __init__(self, filters=64, **kwargs):
        super(HRNetStem, self).__init__(**kwargs)

        self.filters = filters

    def build(self, input_shape):
        # The stem of the network.
        self.conv_1 = layers.Conv2D(self.filters, 3, 2, 'same')
        self.batch_norm_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(self.filters, 3, 2, 'same')
        self.batch_norm_2 = layers.BatchNormalization()
        self.activation = layers.Activation('relu')

        self.built = True

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.activation(x)

        return x

    def get_config(self):
        config = super(HRNetStem, self).get_config()
        config.update({"filters": self.filters})

        return config


class HRNetTail(layers.Layer):

    def __init__(self, input_channels=64, output_channels=17, **kwargs):
        super(HRNetTail, self).__init__(**kwargs)

        self.input_channels = input_channels
        self.output_channels = output_channels

    def build(self, input_shape):
        # Up sampling layers.
        scales = [2, 4, 8]
        self.up_scale_layers = [layers.UpSampling2D((s, s)) for s in scales]
        self.concatenate = layers.Concatenate(axis=3)
        self.conv_1 = layers.Conv2D(filters=self.input_channels, kernel_size=(1, 1),
                                    strides=(1, 1), padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.conv_2 = layers.Conv2D(filters=self.output_channels, kernel_size=(1, 1),
                                    strides=(1, 1), padding='same')

        self.built = True

    def call(self, inputs):
        scaled = [f(x) for f, x in zip(self.up_scale_layers, inputs[1:])]
        x = self.concatenate([inputs[0], scaled[0], scaled[1], scaled[2]])
        x = self.conv_1(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv_2(x)

        return x

    def get_config(self):
        config = super(HRNetTail, self).get_config()
        config.update({"input_channels": self.input_channels,
                       "output_channels": self.output_channels})

        return config


def HRNetV2(width=18, output_channels=98):
    """This function returns a functional model of HRNetV2.

    Args:
        width: the hyperparameter width.
        output_channels: number of output channels.

    Returns:
        a functional model.
    """
    # Get the output size of the HRNet body.
    last_stage_width = sum([width * pow(2, n) for n in range(4)])

    # Describe the model.
    inputs = keras.Input((256, 256, 3), dtype=tf.float32)
    x = HRNetStem(64)(inputs)
    x = HRNetBody(width)(x)
    outputs = HRNetTail(input_channels=last_stage_width,
                        output_channels=output_channels)(x)

    # Construct the model and return it.
    model = keras.Model(inputs=inputs, outputs=outputs, name="hrnetv2")

    return model


if __name__ == "__main__":
    model = HRNetV2(18)
    model(keras.Input((256, 256, 3)))
    model.summary()
