"""This module constains the implimentation of class DataPair."""
import json

import cv2
import numpy as np


class DataPair(object):
    """A pair of data consists of a single image and coresponding marks."""

    def __init__(self, image_file, marks, key_marks_indices):
        """Construct a facial mark data pair

        Args:
            image_file: a path to the image.
            marks: facial marks stored in a numpy array, as [[x, y, z], [x, y, z]
            ...].
            key_marks_indices: the indices of key marks. Key marks are: left eye
            left corner, left eye right corner, right eye left corner, right eye
            right corner, mouse left corner, mouse right corner.

        Returns:
            a DataPair object.
        """
        self.image_file = image_file
        self.marks = marks
        self.key_marks_indices = key_marks_indices

    def read_image(self, format="BGR"):
        """Read in the image as a Numpy array.

        Args:
            format: Color channel order, "BGR" as default. Set it to "RGB" if you
            want to use it in matplotlib.

        Returns:
            an image as numpy array.
        """
        image_bgr = cv2.imread(self.image_file, cv2.IMREAD_COLOR)
        if format is "RGB":
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_bgr

    def get_marks(self):
        """Return all the marks.

        Args:
            None

        Returns:
            The full marks as a numpy array.
        """
        return self.marks

    def get_key_marks(self):
        """Return the key marks of the current marks, in the order of: left eye
            left corner, left eye right corner, right eye left corner, right eye
            right corner, mouse left corner, mouse right corner.

        Args:
            None

        Returns:
            key marks in form of [[x, y, z],[x, y, z]] as a numpy array.
        """
        key_marks = []
        [key_marks.append(self.marks[i]) for i in self.key_marks_indices]
        return np.array(key_marks)

    def save_mark_to_json(self, file_name):
        """Save the marks to a json file.

        Args:
            file_name: the full path of the json file.

        Returns:
            None
        """
        with open(file_name, "w") as fid:
            json.dump(self.marks.tolist(), fid)
