"""Optimize the model with pruning."""
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from dataset import build_dataset_from_wflw
from network import hrnet_v2


parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Training batch size.")
args = parser.parse_args()


if __name__ == "__main__":
    # There are 3 steps for model pruning.
    #   1. Load the model with pretrained weights.
    #   2. Prune the model during training.
    #   3. Export the model.

    # Where are the pretrained weights.
    checkpoint_dir = "./checkpoints"

    # Where the pruned model will be exported
    pruned_model_path = "./optimized/pruned"

    if not os.path.exists(pruned_model_path):
        os.makedirs(pruned_model_path)

    # First, create the model and restore it with pretrained weights.
    model = hrnet_v2((256, 256, 3), width=18, output_channels=98)

    # Restore the latest model from checkpoint.
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_checkpoint)
    print("Checkpoint restored: {}".format(latest_checkpoint))

    # Second, Setup the pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.5,
            final_sparsity=0.8,
            begin_step=0,
            end_step=500
        )
    }
    model_pruned = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params)

    # Hyper parameters for training.
    epochs = args.epochs
    batch_size = args.batch_size

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir="./logs"),
    ]

    # Construct training datasets.
    train_files_dir = "/home/robin/data/facial-marks/wflw_cropped/train"
    dataset_train = build_dataset_from_wflw(train_files_dir, "wflw_train",
                                            training=True,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            prefetch=tf.data.experimental.AUTOTUNE,
                                            mode="generator")

    # Construct dataset for validation & testing.
    test_files_dir = "/home/robin/data/facial-marks/wflw_cropped/test"
    dataset_val = build_dataset_from_wflw(test_files_dir, "wflw_test",
                                          training=False,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          prefetch=tf.data.experimental.AUTOTUNE,
                                          mode="generator")

    # Compile the model for pruning.
    model_pruned.compile(optimizer=keras.optimizers.Adam(0.0001),
                         loss=keras.losses.MeanSquaredError(),
                         metrics=[keras.metrics.MeanSquaredError()])
    model_pruned.summary()

    # Start training loop.
    model_pruned.fit(dataset_train, validation_data=dataset_val,
                     epochs=epochs, callbacks=callbacks,
                     initial_epoch=args.initial_epoch)

    # At last, Export the pruned model.
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_pruned)
    model_for_export.save(pruned_model_path, include_optimizer=False)
    print("Pruned model saved to: {}".format(pruned_model_path))
