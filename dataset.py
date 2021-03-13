"""This module provides the training and testing datasets."""
import cv2
import numpy as np
import tensorflow as tf

from fmd.universal import Universal
from preprocessing import (flip_randomly, generate_heatmaps, normalize,
                           rotate_randomly, scale_randomly)


def data_generator(data_dir, name, image_size, number_marks, training):
    """A generator function used to make TensorFlow dataset.

    Currently only `universal` dataset (image + json) of FMD is supported.

    Args:
        data_dir: the direcotry of the raw image and json files. 
        name: the name of the dataset.
        image_size: the width and height of the input images for the network.
        number_marks: how many marks/points does one sample contains.
        training: generated data will be used for training or not.

    Yields:
        preprocessed image and heatmaps.
    """

    # Initialize the dataset with files.
    dataset = Universal(name.decode("utf-8"))
    dataset.populate_dataset(data_dir.decode("utf-8"), key_marks_indices=None)
    dataset.meta.update({"num_marks": number_marks})

    image_size = tuple(image_size)
    width, _ = image_size
    for sample in dataset:
        # Follow the official preprocessing implementation.
        image = sample.read_image("RGB")
        marks = sample.marks

        if training:
            # Rotate the image randomly.
            image, marks = rotate_randomly(image, marks, (-30, 30))

            # Scale the image randomly.
            image, marks = scale_randomly(image, marks, output_size=image_size,
                                          scale_range=(0.1, 0.2))

            # Flip the image randomly.
            image, marks = flip_randomly(image, marks)
        else:
            # Scale the image to output size.
            image, marks = scale_randomly(image, marks, output_size=image_size,
                                          scale_range=(0.1, 0.2))

        # Normalize the image and marks.
        image_float = normalize(image.astype(float))
        marks = marks[:, :2]

        yield image_float, marks


def build_dataset(data_dir,
                  name,
                  number_marks,
                  image_shape=(256, 256, 3),
                  training=True,
                  batch_size=None,
                  shuffle=True):
    """Generate TensorFlow dataset from image and json files.

    Args:
        data_dir: the directory of the images and json files.
        name: dataset name.
        image_shape: the shape of the target output image of the dataset.
        number_marks: how many marks/points does one sample contains.
        training: True if dataset is for training.
        batch_size: batch size.
        shuffle: True if data should be shuffled.

    Returns:
        a tf.data.dataset.
    """
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(image_shape, (number_marks, 2)),
        args=[data_dir, name, image_shape[:2], number_marks, training])

    print("Dataset built from generator: {}".format(name))

    # Shuffle the data.
    if shuffle:
        dataset = dataset.shuffle(1024)

    # Batch the data.
    dataset = dataset.batch(batch_size)

    # Prefetch the data.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    data_dir = "/home/robin/data/facial-marks/wflw_cropped/test"
    batch_size = 1

    # Build dataset from generator.
    dataset_from_generator = build_dataset(data_dir, "wflw_generator",
                                           number_marks=98,
                                           training=False,
                                           batch_size=batch_size,
                                           shuffle=False)

    # Show the result in windows.
    for s in dataset_from_generator:
        image, marks = s
        image = image.numpy()[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        marks = marks.numpy().reshape((98, 2))
        for mark in marks:
            cv2.circle(image, (abs(int(mark[0])), abs(
                int(mark[1]))), 2, (0, 255, 0), -1)

        cv2.imshow("images", image)

        if cv2.waitKey() == 27:
            break
