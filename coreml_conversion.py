"""Convert the TensorFlow model to CoreML model supported by Apple devices.

MacOS is REQUIRED for quantization.
"""

import os

import coremltools as ct
import tensorflow as tf
from coremltools.models.neural_network import quantization_utils

if __name__ == "__main__":
    # Converted model will be exported here.
    export_dir = "./mlmodels"
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)

    # Restore the model.
    model = tf.keras.models.load_model("./exported")

    # Do the conversion.
    mlmodel = ct.convert(model)
    mlmodel.save("./mlmodels/hrnetv2_fp32.mlmodel")

    # Quantization: FP16
    model_fp16 = quantization_utils.quantize_weights(mlmodel, nbits=16)
    model_fp16.save("./mlmodels/hrnetv2_fp16.mlmodel")

    # Quantization: INT8
    model_int8 = quantization_utils.quantize_weights(mlmodel, nbits=8)
    model_int8.save("./mlmodels/model_int8.mlmodel")

