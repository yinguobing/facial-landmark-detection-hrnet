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


def rotate(image, degrees):
    """Rotate the image in degrees.

    Args:
        image: an image with face to be processed.
        degrees: degrees to rotate.

    Returns:
        a same size image rotated.
    """
    img_height, img_width, _ = image.shape
    rotation_mat = cv2.getRotationMatrix2D(((img_width-1)/2.0,
                                            (img_height-1)/2.0), degrees, 1)
    image_rotated = cv2.warpAffine(
        image, rotation_mat, (img_width, img_height))

    return image_rotated


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
    marks /= ((img_width - margin * 2) * output_size[0])

    return image_resized, marks
