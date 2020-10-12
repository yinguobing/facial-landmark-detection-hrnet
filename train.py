"""The training script for HRNet facial landmark detection.
"""
import os
from argparse import ArgumentParser

import tensorflow as tf
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
parser.add_argument("--export_only", default=False, type=bool,
                    help="Save the model without training.")
parser.add_argument("--eval_only", default=False, type=bool,
                    help="Evaluate the model without training.")
args = parser.parse_args()


class EpochBasedLearningRateSchedule(keras.callbacks.Callback):
    """Sets the learning rate according to epoch schedule."""

    def __init__(self, schedule):
        """
        Args:
            schedule: a tuple that takes an epoch index (integer, indexed from 0)
            and current learning rate.
        """
        super(EpochBasedLearningRateSchedule, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(
            self.model.optimizer.learning_rate))

        # Get the scheduled learning rate.
        def _lr_schedule(epoch, lr, schedule):
            """Helper function to retrieve the scheduled learning rate based on
             epoch."""
            if epoch < schedule[0][0] or epoch > schedule[-1][0]:
                return lr
            for i in range(len(schedule)):
                if epoch == schedule[i][0]:
                    return schedule[i][1]
            return lr

        scheduled_lr = _lr_schedule(epoch, lr, self.schedule)

        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.6f." % (epoch, scheduled_lr))


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

    # Construct dataset for validation & testing.
    test_files_dir = "/home/robin/data/facial-marks/wflw_cropped/test"
    dataset_val = make_wflw_dataset(test_files_dir, "wflw_test",
                                    training=False,
                                    batch_size=args.batch_size,
                                    mode="generator")
    if not isinstance(dataset_val, keras.utils.Sequence):
        dataset_val = dataset_val.batch(args.batch_size)

    # Train the model.
    if not (args.eval_only or args.export_only):
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

        # Visualization in TensorBoard
        # Graph is not available for now, see tensorflow issue:42133
        callback_tensorboard = keras.callbacks.TensorBoard(log_dir="./log",
                                                           histogram_freq=1024,
                                                           write_graph=True,
                                                           update_freq='epoch')

        # Schedule the learning rate with (epoch to start, learning rate) tuples
        schedule = [(1, 0.001),
                    (30, 0.0001),
                    (50, 0.00001)]
        callback_lr = EpochBasedLearningRateSchedule(schedule)

        # List all the callbacks.
        callbacks = [callback_checkpoint, callback_tensorboard, callback_lr]

        # Construct training datasets.
        train_files_dir = "/home/robin/data/facial-marks/wflw_cropped/train"
        dataset_train = make_wflw_dataset(train_files_dir, "wflw_train",
                                          training=True,
                                          batch_size=args.batch_size,
                                          mode="generator")
        if not isinstance(dataset_train, keras.utils.Sequence):
            dataset_train = dataset_train.batch(args.batch_size)

        # Start training loop.
        model.fit(dataset_train, validation_data=dataset_val,
                  epochs=epochs, callbacks=callbacks,
                  initial_epoch=args.initial_epoch)

    # Evaluate the model.
    if not args.export_only:
        model.evaluate(dataset_val)

    # Save the model for inference.
    if args.export_only:
        model.predict(tf.zeros((1, 256, 256, 3)))
        model.save("./exported")
