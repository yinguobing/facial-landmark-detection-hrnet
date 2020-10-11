"""This module provides the training and testing datasets."""
import cv2
import numpy as np
import tensorflow as tf

from fmd.universal import Universal
from preprocess import (flip_randomly, generate_heatmaps, normalize,
                        rotate_randomly, scale_randomly)


def generate_wflw_data(data_dir, name, training):
    """A generator function to make WFLW dataset"""

    # Initialize the dataset with files.
    dataset = Universal(name.decode("utf-8"))
    dataset.populate_dataset(data_dir.decode("utf-8"), key_marks_indices=[
        60, 64, 68, 72, 76, 82])

    for sample in dataset:
        # Follow the official preprocess implementation.
        image = sample.read_image("RGB")
        marks = sample.marks

        if training:
            # Rotate the image randomly.
            image, marks = rotate_randomly(image, marks, (-30, 30))

            # Scale the image randomly.
            image, marks = scale_randomly(image, marks)

            # Flip the image randomly.
            image, marks = flip_randomly(image, marks)
        else:
            # Scale the image to output size.
            marks = marks / image.shape[0] * 256
            image = cv2.resize(image, (256, 256))

        # Normalize the image.
        image_float = normalize(image.astype(float))

        # Generate heatmaps.
        _, img_width, _ = image.shape
        heatmaps = generate_heatmaps(marks, img_width, (64, 64))
        heatmaps = np.transpose(heatmaps, (1, 2, 0))

        yield image_float, heatmaps


class WFLWSequence(tf.keras.utils.Sequence):
    """A Sequence implementation for WFLW dataset generation."""

    def __init__(self, data_dir, name, training, batch_size):
        self.training = training
        self.batch_size = batch_size
        self.filenames = []
        self.marks = []

        # Initialize the dataset with files.
        dataset = Universal(name)
        dataset.populate_dataset(data_dir, key_marks_indices=[
            60, 64, 68, 72, 76, 82])

        for sample in dataset:
            self.filenames.append(sample.image_file)
            self.marks.append(sample.marks)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_files = self.filenames[index *
                                     self.batch_size:(index + 1) * self.batch_size]
        batch_marks = self.marks[index *
                                 self.batch_size:(index + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for filename, marks in zip(batch_files, batch_marks):
            # Follow the official preprocess implementation.
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.training:
                # Rotate the image randomly.
                image, marks = rotate_randomly(image, marks, (-30, 30))

                # Scale the image randomly.
                image, marks = scale_randomly(image, marks)

                # Flip the image randomly.
                image, marks = flip_randomly(image, marks)
            else:
                # Scale the image to output size.
                marks = marks / image.shape[0] * 256
                image = cv2.resize(image, (256, 256))

            # Normalize the image.
            image_float = normalize(image.astype(float))

            # Generate heatmaps.
            _, img_width, _ = image.shape
            heatmaps = generate_heatmaps(marks, img_width, (64, 64))
            heatmaps = np.transpose(heatmaps, (1, 2, 0))

            # Generate the batch data.
            batch_x.append(image_float)
            batch_y.append(heatmaps)

        return np.array(batch_x), np.array(batch_y)


def make_wflw_dataset(data_dir, name, training=True, batch_size=None, mode="sequence"):
    """Generate WFLW dataset from image and json files.

    Args:
        data_dir: the directory of the images and json files.
        name: dataset name.
        mode: keras Sequence or dataset from generator.

    Returns:
        a keras.utils.Sequence or a tf.data.dataset.
    """
    if mode == 'sequence':
        dataset = WFLWSequence(data_dir, name, training, batch_size)
    else:
        dataset = tf.data.Dataset.from_generator(
            generate_wflw_data,
            output_types=(tf.float32, tf.float32),
            output_shapes=((256, 256, 3), (64, 64, 98)),
            args=[data_dir, "name", training])

    return dataset
