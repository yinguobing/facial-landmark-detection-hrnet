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
        _points = marks - np.array(center, np.float)
        cos_angle = np.cos(radian)
        sin_angle = np.sin(radian)
        rotaion_matrix = np.array([[cos_angle, sin_angle],
                                   [-sin_angle, cos_angle]])

        return np.dot(_points, rotaion_matrix) + center

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
                    ] = heat[y_heat[0]:y_heat[1], x_heat[0]:x_heat[1]]

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
