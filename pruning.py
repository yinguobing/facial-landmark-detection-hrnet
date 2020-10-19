import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from dataset import make_wflw_dataset
from network import HRNetV2


parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Training batch size.")
args = parser.parse_args()


if __name__ == "__main__":
    # Create the model.
    model = HRNetV2(width=18, output_channels=98)

    # Restore the latest model if checkpoints are available.`
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])

    # Setup the pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.5,
            final_sparsity=0.8,
            begin_step=0,
            end_step=700
        )
    }
    model_pruned = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params)

    model_pruned.summary()

    # Hyper parameters for training.
    epochs = args.epochs
    batch_size = args.batch_size

    # Save a checkpoint. This could be used to resume training.
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        verbose=1,
        save_best_only=True)

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir="./log"),
    ]

    # Construct training datasets.
    train_files_dir = "/home/robin/data/facial-marks/wflw_cropped/train"
    dataset_train = make_wflw_dataset(train_files_dir, "wflw_train",
                                      training=True,
                                      batch_size=args.batch_size,
                                      mode="generator")
    if not isinstance(dataset_train, keras.utils.Sequence):
        dataset_train = dataset_train.batch(args.batch_size)

    # Construct dataset for validation & testing.
    test_files_dir = "/home/robin/data/facial-marks/wflw_cropped/test"
    dataset_val = make_wflw_dataset(test_files_dir, "wflw_test",
                                    training=False,
                                    batch_size=args.batch_size,
                                    mode="generator")
    if not isinstance(dataset_val, keras.utils.Sequence):
        dataset_val = dataset_val.batch(args.batch_size)

    model_pruned.compile(optimizer=keras.optimizers.Adam(0.0001),
                         loss=keras.losses.MeanSquaredError(),
                         metrics=[keras.metrics.MeanSquaredError()])

    # Start training loop.
    model_pruned.fit(dataset_train, validation_data=dataset_val,
                     epochs=epochs, callbacks=callbacks,
                     initial_epoch=args.initial_epoch)
