"""This module provides commonly used image preprocessing functions."""
import cv2
import numpy as np
import tensorflow as tf

from mark_operator import MarkOperator

MO = MarkOperator()


def normalize(inputs):
    """Preprocess the inputs. This function follows the official implementation
    of HRNet.

    Args:
        inputs: a TensorFlow tensor of image.

    Returns:
        a normalized image.
    """
    img_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    img_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Normalization
    return ((inputs / 255.0) - img_mean)/img_std


def rotate_randomly(image, marks, degrees=(-30, 30)):
    """Rotate the image randomly in degree range (-degrees, degrees).

    Args:
        image: an image with face to be processed.
        marks: face marks.
        degrees: degree ranges to rotate.

    Returns:
        a same size image rotated, and the rotated marks.
    """
    degree = np.random.random_sample() * (degrees[1] - degrees[0]) + degrees[0]
    img_height, img_width, _ = image.shape
    rotation_mat = cv2.getRotationMatrix2D(((img_width-1)/2.0,
                                            (img_height-1)/2.0), degree, 1)
    image_rotated = cv2.warpAffine(
        image, rotation_mat, (img_width, img_height))

    marks_rotated = MO.rotate(marks, np.deg2rad(degree),
                              (img_width/2, img_height/2))

    return image_rotated, marks_rotated


def scale_randomly(image, marks, output_size=(256, 256), scale_range=(0, 1)):
    """Scale the image randomly in a valid range defined by factor.

    This function automatically calculates the valid scale range so that the
    marks will always be visible in the image.

    Args:
        image: an image fully covered the face area in which the face is also 
            centered.
        marks: face marks as numpy array in pixels.
        scale_range: a tuple (a, b) defines the min and max values of the scale
            range from the valid range.
        output_size: output image size.

    Returns:
        processed image with target output size and new marks.
    """
    img_height, img_width, _ = image.shape
    face_height, face_width, _ = MO.get_height_width_depth(marks)

    # The input image may not be a square. Choose the min range as valid range.
    valid_range = min(img_height - face_height, img_width - face_width) / 2

    # Get the new range from user input.
    low, high = (np.array(scale_range) * valid_range).astype(int)
    margin = np.random.randint(low, high)

    # Cut the margins to the new image bounding box.
    x_start = y_start = margin
    x_stop, y_stop = (img_width - margin, img_height - margin)

    # Crop and resize the image.
    image_cropped = image[y_start:y_stop, x_start:x_stop]
    image_resized = cv2.resize(image_cropped, output_size)

    # Get the new mark locations.
    marks -= [margin, margin, 0]
    marks = (marks / (img_width - margin * 2) * output_size[0]).astype(int)

    return image_resized, marks


def flip_randomly(image, marks, probability=0.5):
    """Flip the image in horizontal direction.

    Args:
        image: input image.
        marks: face marks.

    Returns:
        flipped image, flipped marks
    """
    if np.random.random_sample() < probability:
        image = cv2.flip(image, 1)
        marks = MO.flip_lr(marks, image.shape[0])

    return image, marks


def generate_heatmaps(marks, img_size, map_size):
    """A convenient function to generate heatmaps from marks."""
    marks_norm = marks / img_size
    heatmaps = MO.generate_heatmaps(marks_norm, map_size=map_size)

    return heatmaps
