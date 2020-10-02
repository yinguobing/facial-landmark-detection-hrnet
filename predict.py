"""Sample module for predicting face marks with HRNetV2."""
import cv2
import numpy as np
import tensorflow as tf


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
    [x1, x2], [y1, y2] = top_k_indices(heatmap, 2)
    x = (x1 + (x2 - x1)/4) / h_height * image_size[0]
    y = (y1 + (y2 - y1)/4) / h_width * image_size[1]

    return int(x), int(y)


if __name__ == "__main__":
    img = cv2.imread("/home/robin/Desktop/sample/face.jpg")
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = (np.array(img_rgb, dtype=np.float32) - 127.5)/127.5

    imported = tf.saved_model.load("./exported")
    heatmaps = imported([img_input]).numpy()[0]
    heatmaps = np.rollaxis(heatmaps, 2)

    heatmap_idvs = np.hstack(heatmaps[:8])
    for row in range(1, 12, 1):
        heatmap_idvs = np.vstack(
            [heatmap_idvs, np.hstack(heatmaps[row:row+8])])

    for heatmap in heatmaps:
        mark = get_peak_location(heatmap)
        cv2.circle(img, mark, 2, (0, 255, 0), -1)

    cv2.imshow('image', img)
    cv2.imshow("Heatmap_idvs", heatmap_idvs)
    cv2.waitKey()
