"""A module provids common operations for point marks.

All marks, or points are numpy arrays of format like:
    mark = [x, y, z]
    marks = [[x, y, z],
             [x, y, z],
             ...,
             [x, y, z]]

Vectors are also numpy arrays:
    vector = [x, y, z]
    vectors = [[x, y, z],
               [x, y, z],
               ...,
               [x, y, z]]

"""
import numpy as np


class MarkOperator(object):
    """Operator instances are used to transform the marks."""

    def __init__(self):
        pass

    def get_distance(self, mark1, mark2):
        """Calculate the distance between two marks."""
        return np.linalg.norm(mark2 - mark1)

    def get_angle(self, vector1, vector2, in_radian=False):
        """Return the angel between two vectors."""
        d = np.dot(vector1, vector2)
        cos_angle = d / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if cos_angle > 1.0:
            radian = 0
        elif cos_angle < -1.0:
            radian = np.pi
        else:
            radian = np.arccos(cos_angle)

        c = np.cross(vector1, vector2)
        if (c.ndim == 0 and c < 0) or (c.ndim == 1 and c[2] < 0):
            radian = 2*np.pi - radian

        return radian if in_radian is True else np.rad2deg(radian)

    def pad_to_3d(self, marks_2d, pad_value=-1):
        """Pad the 2D marks with zeros in z axis."""
        marks_3d = np.pad(marks_2d, ((0, 0), (0, 1)),
                          mode='constant', constant_values=pad_value)

        return marks_3d

    def get_center(self, marks):
        """Return the center point of the mark group."""
        x, y, z = (np.amax(marks, 0) + np.amin(marks, 0)) / 2

        return np.array([x, y, z])

    def get_height_width_depth(self, marks):
        """Return the height and width of the marks bounding box."""
        height, width, depth = np.amax(marks, 0) - np.amin(marks, 0)

        return height, width, depth

    def rotate(self, marks, radian, center=(0, 0)):
        """Rotate the marks around center by angle"""
        _points = marks[:, :2] - np.array(center, np.float)
        cos_angle = np.cos(-radian)
        sin_angle = np.sin(-radian)
        rotaion_matrix = np.array([[cos_angle, sin_angle],
                                   [-sin_angle, cos_angle]])
        marks[:, :2] = np.dot(_points, rotaion_matrix) + center

        return marks

    def flip_lr(self, marks, width):
        """Flip the marks in horizontal direction."""
        marks[:, 0] = width - marks[:, 0]

        # Reset the order of the marks. The HRNet authors had provided this
        # information in the official repository.
        num_marks = marks.shape[0]
        if num_marks == 98:     # WFLW
            mirrored_pairs = np.array([
                [0,  32], [1,  31], [2,  30], [3,  29], [4,  28], [5,  27],
                [6,  26], [7,  25], [8,  24], [9,  23], [10, 22], [11, 21],
                [12, 20], [13, 19], [14, 18], [15, 17], [33, 46], [34, 45],
                [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48],
                [41, 47], [60, 72], [61, 71], [62, 70], [63, 69], [64, 68],
                [65, 75], [66, 74], [67, 73], [55, 59], [56, 58], [76, 82],
                [77, 81], [78, 80], [87, 83], [86, 84], [88, 92], [89, 91],
                [95, 93], [96, 97]
            ])
        elif num_marks == 68:   # IBUG, etc.
            mirrored_pairs = np.array([
                [1,  17], [2,  16], [3,  15], [4,  14], [5,  13], [6,  12],
                [7,  11], [8,  10], [18, 27], [19, 26], [20, 25], [21, 24],
                [22, 23], [32, 36], [33, 35], [37, 46], [38, 45], [39, 44],
                [40, 43], [41, 48], [42, 47], [49, 55], [50, 54], [51, 53],
                [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]]) - 1
        else:
            raise ValueError(
                "Number of points {} not supported, please check the dataset.".format(num_marks))

        tmp = marks[mirrored_pairs[:, 0]]
        marks[mirrored_pairs[:, 0]] = marks[mirrored_pairs[:, 1]]
        marks[mirrored_pairs[:, 1]] = tmp

        return marks

    def generate_heatmaps(self, norm_marks, map_size=(64, 64), sigma=3):
        """Generate heatmaps for all the marks."""

        def _generate_heatmap(heatmap, point, sigma, label_type='Gaussian'):
            """This function is borrowed from the official implementation. We
            will use the method whatever the HRNet authors used. But some
            variables are re-named to make it easier to read. Maybe someday I
            will re-write this.
            """
            # Check that any part of the gaussian is in-bounds
            tmp_size = sigma * 3
            ul = [int(point[0] - tmp_size), int(point[1] - tmp_size)]
            br = [int(point[0] + tmp_size + 1), int(point[1] + tmp_size + 1)]
            if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
                    br[0] < 0 or br[1] < 0):
                # If not, just return the image as is
                return heatmap

            # Generate gaussian
            size_heat = 2 * tmp_size + 1
            x = np.arange(0, size_heat, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size_heat // 2

            # The gaussian is not normalized, we want the center value to equal 1
            if label_type == 'Gaussian':
                heat = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                              (2 * sigma ** 2))
            else:
                heat = sigma / (((x - x0) ** 2 + (y - y0)
                                 ** 2 + sigma ** 2) ** 1.5)

            # Usable gaussian range
            x_heat = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
            y_heat = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]

            # Image range
            x_map = max(0, ul[0]), min(br[0], heatmap.shape[1])
            y_map = max(0, ul[1]), min(br[1], heatmap.shape[0])

            heatmap[y_map[0]:y_map[1], x_map[0]:x_map[1]
                    ] = heat[y_heat[0]: y_heat[1], x_heat[0]: x_heat[1]]

            return heatmap

        maps = []
        width, height = map_size
        for norm_mark in norm_marks:
            heatmap = np.zeros(map_size, dtype=float)
            x = width * norm_mark[0]
            y = height * norm_mark[1]
            heatmap = _generate_heatmap(heatmap, (x, y), sigma)
            maps.append(heatmap)

        return np.array(maps, dtype=np.float)
