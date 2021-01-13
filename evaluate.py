"""Evaluation of the HRNet model on public facial mark datasets."""

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import fmd
from postprocessing import parse_heatmaps
from preprocessing import crop_face, normalize
from quantization import TFLiteModelPredictor


def compute_nme(prediction, ground_truth):
    """This function is based on the official HRNet implementation."""

    interocular = np.linalg.norm(ground_truth[60, ] - ground_truth[72, ])
    rmse = np.sum(np.linalg.norm(
        prediction - ground_truth, axis=1)) / (interocular)

    return rmse


def evaluate(dataset: fmd.mark_dataset.dataset, model):
    """Evaluate the model on the dataset. The evaluation method should be the 
    same with the official code.

    Args:
        dataset: a FMD dataset
        model: any model having `predict` method.
    """
    # For NME
    nme_count = 0
    nme_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0

    # Loop though the dataset samples.
    for sample in tqdm(dataset):
        # Get image and marks.
        image = sample.read_image()
        marks = sample.marks

        # Crop the face out of the image.
        image_cropped, border, bbox = crop_face(image, marks, scale=1.2)
        image_size = image_cropped.shape[:2]

        # Get the prediction from the model.
        image_cropped = cv2.resize(image_cropped, (256, 256))
        img_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        img_input = normalize(np.array(img_rgb, dtype=np.float32))

        # Do prediction.
        heatmaps = model.predict(tf.expand_dims(img_input, 0))[0]

        # Parse the heatmaps to get mark locations.
        marks_prediction, _ = parse_heatmaps(heatmaps, image_size)

        # Transform the marks back to the original image dimensions.
        x0 = bbox[0] - border
        y0 = bbox[1] - border
        marks_prediction[:, 0] += x0
        marks_prediction[:, 1] += y0

        # Compute NME.
        nme_temp = compute_nme(marks_prediction, marks[:, :2])

        if nme_temp > 0.08:
            count_failure_008 += 1
        if nme_temp > 0.10:
            count_failure_010 += 1

        nme_sum += nme_temp
        nme_count = nme_count + 1

        # # Visualize the result.
        # for mark in marks_prediction:
        #     cv2.circle(image, tuple(mark.astype(int)), 2, (0, 255, 0), -1)

        # cv2.imshow("cropped", image_cropped)
        # cv2.imshow("image", image)
        # if cv2.waitKey(1) == 27:
        #     break

    # NME
    nme = nme_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = "NME:{:.4f}, [008]:{:.4f}, [010]:{:.4f}".format(
        nme, failure_008_rate, failure_010_rate)

    return msg


def make_dataset():
    wflw_dir = "/home/robin/data/facial-marks/wflw/WFLW_images"
    ds_wflw = fmd.wflw.WFLW(False, "wflw_test")
    ds_wflw.populate_dataset(wflw_dir)

    return ds_wflw


if __name__ == "__main__":

    # Evaluate with FP32 model.
    model = tf.keras.models.load_model("exported/hrnetv2")
    print("FP32: ", evaluate(make_dataset(), model))

    # # Evaluate with FP16 model.
    # model_qdr = TFLiteModelPredictor(
    #     "./optimized/hrnet_quant_fp16.tflite")
    # print("FP16 quantized:", evaluate(make_dataset(), model_qdr))
