"""The training script for HRNet facial landmark detection.
"""
import os

import tensorflow as tf
from tensorflow import keras

from network import HRNetV2


def parse_dataset(dataset):
    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/depth': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'label/marks': tf.io.FixedLenFeature([], tf.string),
        'label/n_marks': tf.io.FixedLenFeature([], tf.int64),
        'heatmap/map': tf.io.FixedLenFeature([], tf.string),
        'heatmap/height': tf.io.FixedLenFeature([], tf.int64),
        'heatmap/width': tf.io.FixedLenFeature([], tf.int64),
        'heatmap/depth': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, feature_description)

        image_decoded = tf.image.decode_jpeg(
            example['image/encoded'], channels=3)
        image_decoded = tf.cast(image_decoded, tf.float32)
        # TODO: infer the tensor shape automatically
        image_decoded = tf.reshape(image_decoded, [256, 256, 3])

        heatmaps = tf.io.parse_tensor(example['heatmap/map'], tf.double)
        heatmaps = tf.cast(heatmaps, tf.float32)
        # TODO: infer the tensor shape automatically
        heatmaps = tf.reshape(heatmaps, (98, 64, 64))
        heatmaps = tf.transpose(heatmaps, [2, 1, 0])

        return image_decoded, heatmaps

    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset


if __name__ == "__main__":
    # Construct training and validation datasets.
    record_file_train = "/home/robin/data/facial-marks/wflw/tfrecord/wflw_train.record"
    record_file_test = "/home/robin/data/facial-marks/wflw/tfrecord/wflw_test.record"

    dataset_train = parse_dataset(tf.data.TFRecordDataset(record_file_train))
    dataset_train = dataset_train.shuffle(1024).batch(16)

    dataset_val = parse_dataset(tf.data.TFRecordDataset(record_file_test))

    # Create the model.
    model = HRNetV2(width=18, output_channels=98)

    # Restore the latest model if checkpoints are available.`
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoints = [checkpoint_dir + "/" +
                   name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        # TODO: filter the latest checkpoint file.
        model.load_weights(checkpoint_dir + "/ckpt-1")

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])

    # Callbacks are used to record the training process.
    callbacks = [
        # Save a SavedModel. This could be used to resume training.
        # TODO: more proper name for checkpoint files.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq=1, verbose=1)
    ]

    # Train the model.
    model.fit(dataset_train, epochs=1, callbacks=callbacks)
