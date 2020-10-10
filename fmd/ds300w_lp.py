import cv2
import numpy as np
from scipy.io import loadmat

from mark_dataset.dataset import MarkDataset
from mark_dataset.util import FileListGenerator


class DS300W_LP(MarkDataset):
    # To use this class, there are two functions should be overridden.

    def populate_dataset(self, image_dir):
        """Populate the 300W_LP dataset with essential data.

        Args:
            image_dir: the direcotry of the dataset images.
        """
        # As required by the abstract method, we need to override this function.
        # 1. populate the image file list.
        lg = FileListGenerator()
        self.image_files = lg.generate_list(image_dir)

        # 2. Populate the mark file list. Note the order should be same with the
        # image file list. Since the 300W_LP dataset had the mark file named
        # after the image file but with different extention name `mat`. We will
        # make use of this.
        self.mark_files = [img_path.split(
            ".")[-2] + ".mat" for img_path in self.image_files]

        # 3 Set the key marks indices. Here key marks are: left eye left corner,
        #  left eye right corner, right eye left corner, right eye right corner,
        #  mouse left corner, mouse right corner. For 300W_LP the indices 
        # are 36, 39, 42, 45, 48, 54. Most of the time you need to do this 
        # manually. Refer to the mark dataset for details.
        self.key_marks_indices = [36, 39, 42, 45, 48, 54]

        # Even optional, it is highly recommended to update the meta data.
        self.meta.update({"authors": "Chinese Academy of Sciences",
                          "year": 2015,
                          "num_marks": 68,
                          "num_samples": len(self.image_files)
                          })

    def get_marks_from_file(self, mark_file):
        """This function should read the mark file and return the marks as a 
        numpy array in form of [[x, y, z], [x, y, z]]."""
        marks = []
        mat_data = loadmat(mark_file)
        marks = np.pad(mat_data['pt2d'].T, (0, 1))
        assert marks.shape[1] == 3, "Marks should be 3D, check z axis values."
        return marks
