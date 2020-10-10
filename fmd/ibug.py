"""Dataset toolkit for IBUG

Useage: https://github.com/yinguobing/facial-landmark-dataset/issues/3
"""

import cv2
import numpy as np

from fmd.mark_dataset.dataset import MarkDataset
from fmd.mark_dataset.util import FileListGenerator


class IBUG(MarkDataset):
    # To use this class, there are two functions need to be overridden.

    def populate_dataset(self, image_dir):
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
            ".")[-2] + ".pts" for img_path in self.image_files]

        # 3 Set the key marks indices. Here key marks are: left eye left corner,
        #  left eye right corner, right eye left corner, right eye right corner,
        #  mouse left corner, mouse right corner. For IBUG the indices are 36,
        # 39, 42, 45, 48, 54. Most of the time you need to do this manually.
        # Refer to the mark dataset for details.
        self.key_marks_indices = [36, 39, 42, 45, 48, 54]

        # Even optional, it is highly recommended to update the meta data.
        self.meta.update({"authors": "Imperial College London",
                          "year": 2013,
                          "num_marks": 68,
                          "num_samples": len(self.image_files)
                          })

    def get_marks_from_file(self, mark_file):
        """This function should read the mark file and return the marks as a 
        numpy array in form of [[x, y, z], [x, y, z]]."""
        marks = []
        with open(mark_file) as fid:
            for line in fid:
                if "version" in line or "points" in line or "{" in line or "}" in line:
                    continue
                else:
                    loc_x, loc_y = line.strip().split(sep=" ")
                    marks.append([float(loc_x), float(loc_y), 0.0])
        marks = np.array(marks, dtype=np.float)
        assert marks.shape[1] == 3, "Marks should be 3D, check z axis values."
        return marks
