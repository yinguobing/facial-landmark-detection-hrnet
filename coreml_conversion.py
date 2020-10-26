"""Convert the TensorFlow model to CoreML model supported by Apple devices."""

import coremltools as ct
import tensorflow as tf
from coremltools.models.neural_network import quantization_utils

if __name__ == "__main__":
    # Restore the model.
    model = tf.keras.models.load_model("./exported")

    # Do the conversion.
    mlmodel = ct.convert(model)

    # Quantization: FP16
    model_fp16 = quantization_utils.quantize_weights(mlmodel, nbits=16)

    # Save them.
    mlmodel.save("./mlmodels/hrnetv2.mlmodel")
    model_fp16.save("./mlmodels/hrnetv2_fp16.mlmodel")
