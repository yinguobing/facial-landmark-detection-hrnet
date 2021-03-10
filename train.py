"""The training script for HRNet facial landmark detection.
"""
import os
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras

from callbacks import EpochBasedLearningRateSchedule, LogImages
from dataset import build_dataset
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
args = parser.parse_args()


if __name__ == "__main__":
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    # What is the model's name?
    name = "hrnetv2"

    # How many marks are there for a single face sample?
    number_marks = 98

    # Where are the training files?
    train_files_dir = "/home/robin/data/facial-marks/wflw_cropped/train"

    # Where are the testing files?
    test_files_dir = "/home/robin/data/facial-marks/wflw_cropped/test"

    # Where are the validation files? Set `None` if no files available. Then 10%
    # of the training files will be used as validation samples.
    val_files_dir = None

    # Do you have a sample image which will be logged into tensorboard for
    # testing purpose?
    sample_image = "docs/face.jpg"

    # That should be sufficient for training. However if you want more
    # customization, please keep going.

    # Checkpoint is used to resume training.
    checkpoint_dir = os.path.join("checkpoints", name)

    # Save the model for inference later.
    export_dir = os.path.join("exported", name)

    # Log directory will keep training logs like loss/accuracy curves.
    log_dir = os.path.join("logs", name)

    # All sets. Now it's time to build the model. This model is defined in the
    # `network` module with TensorFlow's functional API.
    input_shape = (256, 256, 3)
    model = hrnet_v2(input_shape=input_shape, output_channels=number_marks,
                     width=18, name=name)

    # Model built. Restore the latest model if checkpoints are available.
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

    # Construct a dataset for evaluation.
    dataset_test = build_dataset(test_files_dir, "test",
                                 number_marks=number_marks,
                                 image_shape=input_shape,
                                 training=False,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 prefetch=tf.data.experimental.AUTOTUNE)

    # If only evaluation is required.
    if args.eval_only:
        model.evaluate(dataset_test)
        quit()

    # Finally, it's time to train the model.

    # Compile the model and print the model summary.
    model.compile(optimizer=keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])
    # model.summary()

    # Schedule the learning rate with (epoch to start, learning rate) tuples
    schedule = [(1, 0.001),
                (30, 0.0001),
                (50, 0.00001)]

    # All done. The following code will setup and start the trainign.

    # Save a checkpoint. This could be used to resume training.
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, name),
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

    # Log a sample image to tensorboard.
    callback_image = LogImages(log_dir, sample_image)

    # List all the callbacks.
    callbacks = [callback_checkpoint, callback_tensorboard, #callback_lr,
                 callback_image]

    # Construct training datasets.
    dataset_train = build_dataset(train_files_dir, "train",
                                  number_marks=number_marks,
                                  image_shape=input_shape,
                                  training=True,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  prefetch=tf.data.experimental.AUTOTUNE)

    # Construct dataset for validation. The loss value from this dataset will be
    # used to decide which checkpoint should be preserved.
    if val_files_dir:
        dataset_val = build_dataset(val_files_dir, "validation",
                                    number_marks=number_marks,
                                    image_shape=input_shape,
                                    training=False,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    prefetch=tf.data.experimental.AUTOTUNE)
    else:
        dataset_val = dataset_train.take(int(512/args.batch_size))
        dataset_train = dataset_train.skip(int(512/args.batch_size))

    # Start training loop.
    model.fit(dataset_train,
              validation_data=dataset_val,
              epochs=args.epochs,
              callbacks=callbacks,
              initial_epoch=args.initial_epoch)

    # Run a full evaluation after training.
    model.evaluate(dataset_test)
