import os
import cv2
import numpy as np
import tensorflow as tf

import fmd
from mark_operator import MarkOperator
from preprocessing import crop_face, normalize

MODE = {"DynamicRangeQuantization": None,
        "IntegerWithFloatFallback": None,
        "IntergerOnly": None,
        "FP16": None,
        "16x8": None}


def representative_dataset_gen():
    wflw_dir = "/home/robin/data/facial-marks/wflw/WFLW_images"
    ds_wflw = fmd.wflw.WFLW(False, "wflw_test")
    ds_wflw.populate_dataset(wflw_dir)

    for _ in range(100):
        sample = ds_wflw.pick_one()

        # Get image and marks.
        image = sample.read_image()
        marks = sample.marks

        # Crop the face out of the image.
        image_cropped, _, _ = crop_face(image, marks, scale=1.2)

        # Get the prediction from the model.
        image_cropped = cv2.resize(image_cropped, (256, 256))
        img_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        img_input = normalize(np.array(img_rgb, dtype=np.float32))

        yield [np.expand_dims(img_input, axis=0)]


def quantize(saved_model, mode=None, representative_dataset=None):
    """TensorFlow model quantization by TFLite.

    Args:
        saved_model: the model's directory.
        mode: the quantization mode.

    Returns:
        a tflite model quantized.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model("./exported")

    # By default, do Dynamic Range Quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Integer With Float Fallback
    if mode["IntegerWithFloatFallback"]:
        converter.representative_dataset = representative_dataset

    # Integer only.
    if mode["IntergerOnly"]:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8

    # Float16 only.
    if mode["FP16"]:
        converter.target_spec.supported_types = [tf.float16]

    # [experimental] 16-bit activations with 8-bit weights
    if mode["16x8"]:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

    # Finally, convert the model.
    tflite_model = converter.convert()

    return tflite_model


class TFLiteModelPredictor(object):
    """A light weight class for TFLite model prediction."""

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict(self, inputs):
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_index)

        return predictions


if __name__ == "__main__":
    # The directory to save quantized models.
    export_dir = "./optimized"

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # The model to be quantized.
    saved_model = "./exported"

    # Dynamic range quantization
    mode = MODE.copy()
    mode.update({"DynamicRangeQuantization": True})
    tflite_model = quantize(saved_model, mode)
    open("./optimized/hrnet_quant_dynamic_range.tflite", "wb").write(tflite_model)

    # Full integer quantization - Integer with float fallback.
    mode = MODE.copy()
    mode.update({"IntegerWithFloatFallback": True})
    tflite_model = quantize(saved_model, mode, representative_dataset_gen)
    open("./optimized/hrnet_quant_int_fp_fallback.tflite", "wb").write(tflite_model)

    # Full integer quantization - Integer only
    mode = MODE.copy()
    mode.update({"IntegerOnly": True})
    tflite_model = quantize(saved_model, mode,  representative_dataset_gen)
    open("./optimized/hrnet_quant_int_only.tflite", "wb").write(tflite_model)

    # Float16 quantization
    mode = MODE.copy()
    mode.update({"FP16": True})
    tflite_model = quantize(saved_model, mode)
    open("./optimized/hrnet_quant_fp16.tflite", "wb").write(tflite_model)

    # 16x8 quantization
    mode = MODE.copy()
    mode.update({"16x8": True})
    tflite_model = quantize(saved_model, mode)
    open("./optimized/hrnet_quant_16x8.tflite", "wb").write(tflite_model)
