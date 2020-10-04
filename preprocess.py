import tensorflow as tf


def normalize(inputs):
    """Preprocess the inputs. This function follows the official implementation
    of HRNet.

    Args:
        inputs: a TensorFlow tensor of image.
    """
    img_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std
