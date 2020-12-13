"""A module containing custom callbacks."""
import cv2
import tensorflow as tf
from tensorflow import keras

from preprocessing import normalize
from postprocessing import parse_heatmaps


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


class LogImages(keras.callbacks.Callback):
    def __init__(self, logdir, sample_image):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.sample_image = sample_image

    def on_epoch_end(self, epoch, logs={}):
        # Read in the image file.
        image = cv2.imread(self.sample_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (256, 256))
        img = normalize(img)

        # Do prediction.
        heatmaps = self.model.predict(tf.expand_dims(img, 0))[0]

        # Parse the heatmaps to get mark locations.
        marks, _ = parse_heatmaps(heatmaps, image.shape[:2])
        for mark in marks:
            cv2.circle(image, tuple(mark.astype(int)), 2, (0, 255, 0), -1)

        with self.file_writer.as_default():
            # tf.summary needs a 4D tensor
            img_tensor = tf.expand_dims(image, 0)
            tf.summary.image("test-sample", img_tensor, step=epoch)
