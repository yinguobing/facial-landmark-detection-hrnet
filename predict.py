"""Sample module for predicting face marks with HRNetV2."""
import cv2
import numpy as np
import tensorflow as tf

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
