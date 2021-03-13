import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import Model, layers


def mobilenet(input_shape, output_size, name="mobilenetv3"):
    """This function returns a keras model of MobileNetv3.

    Args:
        output_size: number of output size.

    Returns:
        a functional model.
    """
    base_model = keras.applications.MobileNetV3Small(
        input_shape, include_top=False, pooling='avg')

    # Describe the model.
    inputs = keras.Input(input_shape, dtype=tf.float32)
    x = base_model(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(output_size)(x)
    outputs = keras.layers.Reshape((output_size // 2, 2))(x)

    # Construct the model and return it.
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model


if __name__ == "__main__":
    model_2 = mobilenet((224, 224, 3), 98*2)
    model_2.summary()
