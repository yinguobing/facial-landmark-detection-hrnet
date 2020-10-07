"""This module provides commonly used image preprocessing functions."""
import cv2
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


if __name__ == "__main__":
    img = cv2.imread("/home/robin/Desktop/sample/face.jpg")
    img_rotated = rotate(img, -30)
    cv2.imshow("rotate", img_rotated)
    cv2.waitKey()
