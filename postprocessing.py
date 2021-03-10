"""The post processing module for HRNet facial landmark detection."""
import cv2
import numpy as np


def top_k_indices(x, k):
    """Returns the k largest element indices from a numpy array. You can find
    the original code here: https://stackoverflow.com/q/6910641
    """
    flat = x.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, x.shape)


def get_peak_location(heatmap, image_size=(256, 256)):
    """Return the interpreted location of the top 2 predictions."""
    h_height, h_width = heatmap.shape
    [y1, y2], [x1, x2] = top_k_indices(heatmap, 2)
    x = (x1 + (x2 - x1)/4) / h_width * image_size[0]
    y = (y1 + (y2 - y1)/4) / h_height * image_size[1]

    return int(x), int(y)


def parse_heatmaps(heatmaps, image_size):
    # Parse the heatmaps to get mark locations.
    marks = []
    heatmaps = np.transpose(heatmaps, (2, 0, 1))
    for heatmap in heatmaps:
        marks.append(get_peak_location(heatmap, image_size))

    # Show individual heatmaps stacked.
    heatmap_grid = np.hstack(heatmaps[:8])
    for row in range(1, 12, 1):
        heatmap_grid = np.vstack(
            [heatmap_grid, np.hstack(heatmaps[row:row+8])])

    return np.array(marks), heatmap_grid


def draw_marks(image, marks):
    for m in marks:
        for mark in m:
            cv2.circle(image, tuple(mark.astype(int)), 2, (0, 255, 0), -1)
