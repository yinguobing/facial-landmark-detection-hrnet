"""The training script for HRNet facial landmark detection.
"""
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from callbacks import EpochBasedLearningRateSchedule
from dataset import build_dataset_from_wflw
from network import hrnet_v2

parser = ArgumentParser()
parser.add_argument("--epochs", default=60, type=int,
                    help="Number of training epochs.")
parser.add_argument("--initial_epoch", default=0, type=int,
                    help="From which epochs to resume training.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--eval_only", default=False, type=bool,
                    help="Evaluate the model without training.")
parser.add_argument("--quantization", default=False, type=bool,
                    help="Excute quantization aware training.")
args = parser.parse_args()


if __name__ == "__main__":
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    # Datasets
    train_files_dir = "/home/robin/data/facial-marks/wflw_cropped/train"
    test_files_dir = "/home/robin/data/facial-marks/wflw_cropped/test"

    # Checkpoint is used to resume training.
    checkpoint_dir = "./checkpoints"

    # Save the model for inference later.
    export_dir = "./exported"

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = "./logs"

    # All sets. Now it's time to build the model. This model is defined in the
    # `network` module with TensorFlow's functional API.
    model = hrnet_v2(input_shape=(256, 256, 3), width=18, output_channels=98)

    # Check the model summary.
    model.summary()

    # Model created. Restore the latest model if checkpoints are available.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # If the restored model is ready for inference, save it and quit training.
    if args.export_only:
        if latest_checkpoint is None:
            print("Warning: Model not restored from any checkpoint.")
        print("Saving model to {} ...".format(export_dir))
        model.save(export_dir)
        print("Model saved at: {}".format(export_dir))
        quit()

    # Now is a good time to set up quantization aware training with TensorFlow
    # Model Optimization Toolkits.
    if args.quantization:
        model = tfmot.quantization.keras.quantize_model(model)
        model.summary()

    # The model should be compiled before training.
    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])

    # Construct a dataset for evaluation.
    dataset_test = build_dataset_from_wflw(test_files_dir, "wflw_test",
                                           training=False,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           prefetch=tf.data.experimental.AUTOTUNE,
                                           mode="generator")

    # If only evaluation is required.
    if args.eval_only:
        model.evaluate(dataset_test)
        quit()

    # Construct dataset for validation. The loss value from this dataset will be
    # used to decide which checkpoint should be preserved.
    dataset_val = build_dataset_from_wflw(test_files_dir, "wflw_test",
                                          training=False,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          prefetch=tf.data.experimental.AUTOTUNE,
                                          mode="generator").take(320)

    # Finally, it's time to train the model.

    # Set hyper parameters for training.
    epochs = args.epochs
    batch_size = args.batch_size

    # Schedule the learning rate with (epoch to start, learning rate) tuples
    schedule = [(1, 0.001),
                (30, 0.0001),
                (50, 0.00001)]

    # All done. The following code will setup and start the trainign.

    # Save a checkpoint. This could be used to resume training.
    checkpoint_path = os.path.join(checkpoint_dir, "hrnetv2")
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=True)

    # Visualization in TensorBoard
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1024,
                                                       write_graph=True,
                                                       update_freq='epoch')
    # Learning rate decay.
    callback_lr = EpochBasedLearningRateSchedule(schedule)

    # List all the callbacks.
    callbacks = [callback_checkpoint, callback_tensorboard, callback_lr]

    # Construct training datasets.
    dataset_train = build_dataset_from_wflw(train_files_dir, "wflw_train",
                                            training=True,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            prefetch=tf.data.experimental.AUTOTUNE,
                                            mode="generator")

    # Start training loop.
    model.fit(dataset_train, validation_data=dataset_val,
              epochs=epochs, callbacks=callbacks,
              initial_epoch=args.initial_epoch)

    # Make a full evaluation after training.
    model.evaluate(dataset_test)
