"""This module provides the training and testing datasets."""
import numpy as np

from fmd.universal import Universal
from preprocess import (flip_randomly, generate_heatmaps, normalize,
                        rotate_randomly, scale_randomly)


def generate_wflw_data(data_dir, name):
    """A generator function to make WFLW dataset"""

    # Initialize the dataset with files.
    dataset = Universal(name.decode("utf-8"))
    dataset.populate_dataset(data_dir.decode("utf-8"), key_marks_indices=[
        60, 64, 68, 72, 76, 82])

    for sample in dataset:
        # Follow the official preprocess implementation.
        image = sample.read_image("RGB")
        marks = sample.marks

        # Rotate the image randomly.
        image, marks = rotate_randomly(image, marks, (-30, 30))

        # Scale the image randomly.
        image, marks = scale_randomly(image, marks)

        # Flip the image randomly.
        image, marks = flip_randomly(image, marks)

        # Normalize the image.
        image_float = normalize(image.astype(float))

        # Generate heatmaps.
        _, img_width, _ = image.shape
        heatmaps = generate_heatmaps(marks, img_width, (64, 64))
        heatmaps = np.transpose(heatmaps, (1, 2, 0))

        yield image_float, heatmaps
