"""A module containing custom callbacks."""

import tensorflow as tf
from tensorflow import keras


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
