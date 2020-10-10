"""Dataset toolkit for Universal data format.

In this format the marks are stored in a json file which has same basename of 
the image file.

Example:
    /path/to/sample.jpg
    /path/to/sample.json
"""

import json

import cv2
import numpy as np

from fmd.mark_dataset.dataset import MarkDataset
from fmd.mark_dataset.util import FileListGenerator


class Universal(MarkDataset):
    # To use this class, there are two functions need to be overridden.

    def populate_dataset(self, image_dir, key_marks_indices):
        """Populate the IBUG dataset with essential data.

        Args:
            image_dir: the direcotry of the dataset images.
        """
        # As required by the abstract method, we need to override this function.
        # 1. populate the image file list.
        lg = FileListGenerator()
        self.image_files = lg.generate_list(image_dir)

        # 2. Populate the mark file list. Note the order should be same with the
        # image file list. Since the IBUG dataset had the mark file named after
        # the image file but with different extention name `pts`. We will make
        # use of this.
        self.mark_files = [img_path.split(
            ".")[-2] + ".json" for img_path in self.image_files]

        # 3 Set the key marks indices. Here key marks are: left eye left corner,
        #  left eye right corner, right eye left corner, right eye right corner,
        #  mouse left corner, mouse right corner. For IBUG the indices are 36,
        # 39, 42, 45, 48, 54. Most of the time you need to do this manually.
        # Refer to the mark dataset for details.
        self.key_marks_indices = key_marks_indices

        # Even optional, it is highly recommended to update the meta data.
        self.meta.update({"authors": "YinGuobing",
                          "year": 2020,
                          "num_marks": 98,
                          "num_samples": len(self.image_files)
                          })

    def get_marks_from_file(self, mark_file):
        """This function should read the mark file and return the marks as a 
        numpy array in form of [[x, y, z], [x, y, z]]."""
        marks = []
        with open(mark_file) as fid:
            mark_list = json.load(fid)
            marks = np.reshape(
                mark_list, (self.meta['num_marks'], -1)).astype(float)
        if marks.shape[1] == 2:
            marks = np.pad(marks, ((0, 0), (0, 1)), constant_values=-1)
        assert marks.shape[1] == 3, "Marks should be 3D, check z axis values."
        return marks
