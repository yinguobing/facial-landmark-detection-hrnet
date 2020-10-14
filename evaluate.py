"""Evaluation of the HRNet model on public facial mark datasets."""

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import fmd
from mark_operator import MarkOperator
from preprocessing import normalize
from postprocessing import parse_heatmaps

MO = MarkOperator()


def crop_face(image, marks, scale=1.8, shift_ratios=(0, 0)):
    """Crop the face area from the input image.

    Args:
        image: input image.
        marks: the facial marks of the face to be cropped.
        scale: how much to scale the face box.
        shift_ratios: shift the face box to (right, down) by facebox size * ratios

    Returns:
        Cropped image, new marks, padding_width and bounding box.
    """
    # How large the bounding box is?
    x_min, y_min, _ = np.amin(marks, 0)
    x_max, y_max, _ = np.amax(marks, 0)
    side_length = max((x_max - x_min, y_max - y_min)) * scale

    # Where is the center point of the bounding box?
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Face box is scaled, get the new corners, shifted.
    img_height, img_width, _ = image.shape
    x_shift, y_shift = np.array(shift_ratios) * side_length

    x_start = int(x_center - side_length / 2 + x_shift)
    y_start = int(y_center - side_length / 2 + y_shift)
    x_end = int(x_center + side_length / 2 + x_shift)
    y_end = int(y_center + side_length / 2 + y_shift)

    # In case the new bbox is out of image bounding.
    border_width = 0
    border_x = min(x_start, y_start)
    border_y = max(x_end - img_width, y_end - img_height)
    if border_x < 0 or border_y > 0:
        border_width = max(abs(border_x), abs(border_y))
        x_start += border_width
        y_start += border_width
        x_end += border_width
        y_end += border_width
        image_with_border = cv2.copyMakeBorder(image, border_width,
                                               border_width,
                                               border_width,
                                               border_width,
                                               cv2.BORDER_CONSTANT,
                                               value=[0, 0, 0])
        image_cropped = image_with_border[y_start:y_end,
                                          x_start:x_end]
    else:
        image_cropped = image[y_start:y_end, x_start:x_end]

    return image_cropped, border_width, (x_start, y_start, x_end, y_end)


def compute_nme(prediction, ground_truth):
    """This function is based on the official HRNet implementation."""

    interocular = np.linalg.norm(ground_truth[60, ] - ground_truth[72, ])
    rmse = np.sum(np.linalg.norm(
        prediction - ground_truth, axis=1)) / (interocular)

    return rmse


def evaluate(dataset: fmd.mark_dataset.dataset):
    """Evaluate the model on the dataset. The evaluation method should be the 
    same with the official code."""

    # Load the target model.
    model = tf.keras.models.load_model("./exported")

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

    msg = 'Test Results: nme:{:.4f} \n[008]:{:.4f} ' \
          '\n[010]:{:.4f}'.format(nme, failure_008_rate, failure_010_rate)
    print(msg)


if __name__ == "__main__":
    # WFLW
    wflw_dir = "/home/robin/data/facial-marks/wflw/WFLW_images"
    ds_wflw = fmd.wflw.WFLW(False, "wflw_test")
    ds_wflw.populate_dataset(wflw_dir)
    evaluate(ds_wflw)
