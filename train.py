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

    parsed_dataset = dataset.map(_parse_function,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return parsed_dataset


if __name__ == "__main__":
    # Construct training and validation datasets.
    record_file_train = "/home/robin/data/facial-marks/wflw/tfrecord/wflw_train.record"
    record_file_test = "/home/robin/data/facial-marks/wflw/tfrecord/wflw_test.record"

    dataset_train = parse_dataset(tf.data.TFRecordDataset(record_file_train))
    dataset_train = dataset_train.shuffle(1024).batch(16)

    dataset_val = parse_dataset(tf.data.TFRecordDataset(record_file_test))
    dataset_val = dataset_val.batch(16)

    # Create the model.
    model = HRNetV2(width=18, output_channels=98)

    # Restore the latest model if checkpoints are available.`
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])

    # Callbacks are used to record the training process.

    # Save a checkpoint. This could be used to resume training.
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # Visualization in TensorBoard
    # Graph is not available for now, see tensorflow issue:42133
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir="./log",
                                                       histogram_freq=10,
                                                       write_graph=True,
                                                       update_freq=10,)
    callbacks = [callback_checkpoint, callback_tensorboard]

    # Train the model.
    model.fit(dataset_train, validation_data=dataset_val,
              epochs=3, callbacks=callbacks)
