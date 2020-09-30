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


if __name__ == "__main__":
    img = cv2.imread("/home/robin/Desktop/sample/face.jpg")
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imported = tf.saved_model.load("./exported")
    heatmaps = imported.serve([img]).numpy()[0]
    heatmaps = np.rollaxis(heatmaps, 2)

    heatmap_idvs = np.hstack(heatmaps[:8])
    for row in range(1, 12, 1):
        heatmap_idvs = np.vstack(
            [heatmap_idvs, np.hstack(heatmaps[row:row+8])])

    cv2.imshow("Heatmap_idvs", heatmap_idvs)
    cv2.waitKey()
