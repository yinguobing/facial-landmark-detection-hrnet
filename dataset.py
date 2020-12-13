"""This module provides the training and testing datasets."""
import cv2
import numpy as np
import tensorflow as tf

from fmd.universal import Universal
from preprocessing import (flip_randomly, generate_heatmaps, normalize,
                           rotate_randomly, scale_randomly)


def data_generator(data_dir, name, image_size, number_marks, training):
    """A generator function used to make TensorFlow dataset.

    Currently only `universal` dataset (image + json) of FMD is supported.

    Args:
        data_dir: the direcotry of the raw image and json files. 
        name: the name of the dataset.
        image_size: the width and height of the input images for the network.
        number_marks: how many marks/points does one sample contains.
        training: generated data will be used for training or not.

    Yields:
        preprocessed image and heatmaps.
    """

    # Initialize the dataset with files.
    dataset = Universal(name.decode("utf-8"))
    dataset.populate_dataset(data_dir.decode("utf-8"), key_marks_indices=None)
    dataset.meta.update({"num_marks": number_marks})

    image_size = tuple(image_size)
    width, _ = image_size
    for sample in dataset:
        # Follow the official preprocessing implementation.
        image = sample.read_image("RGB")
        marks = sample.marks

        if training:
            # Rotate the image randomly.
            image, marks = rotate_randomly(image, marks, (-30, 30))

            # Scale the image randomly.
            image, marks = scale_randomly(image, marks, output_size=image_size)

            # Flip the image randomly.
            image, marks = flip_randomly(image, marks)
        else:
            # Scale the image to output size.
            marks = marks / image.shape[0] * width
            image = cv2.resize(image, image_size)

        # Normalize the image.
        image_float = normalize(image.astype(float))

        # Generate heatmaps.
        heatmaps = generate_heatmaps(marks, width, (64, 64))
        heatmaps = np.transpose(heatmaps, (1, 2, 0))

        yield image_float, heatmaps


class WFLWSequence(tf.keras.utils.Sequence):
    """A Sequence implementation for WFLW dataset generation.

    This class is not used in training. It simply demonstrates how to generate
    a TensorFlow dataset by using Keras `Sequence`.
    """

    def __init__(self, data_dir, name, training, batch_size):
        self.training = training
        self.batch_size = batch_size
        self.filenames = []
        self.marks = []

        # Initialize the dataset with files.
        dataset = Universal(name)
        dataset.populate_dataset(data_dir, key_marks_indices=[
            60, 64, 68, 72, 76, 82])

        for sample in dataset:
            self.filenames.append(sample.image_file)
            self.marks.append(sample.marks)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_files = self.filenames[index *
                                     self.batch_size:(index + 1) * self.batch_size]
        batch_marks = self.marks[index *
                                 self.batch_size:(index + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for filename, marks in zip(batch_files, batch_marks):
            # Follow the official preprocessing implementation.
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.training:
                # Rotate the image randomly.
                image, marks = rotate_randomly(image, marks, (-30, 30))

                # Scale the image randomly.
                image, marks = scale_randomly(image, marks)

                # Flip the image randomly.
                image, marks = flip_randomly(image, marks)
            else:
                # Scale the image to output size.
                marks = marks / image.shape[0] * 256
                image = cv2.resize(image, (256, 256))

            # Normalize the image.
            image_float = normalize(image.astype(float))

            # Generate heatmaps.
            _, img_width, _ = image.shape
            heatmaps = generate_heatmaps(marks, img_width, (64, 64))
            heatmaps = np.transpose(heatmaps, (1, 2, 0))

            # Generate the batch data.
            batch_x.append(image_float)
            batch_y.append(heatmaps)

        return np.array(batch_x), np.array(batch_y)


def build_dataset(data_dir,
                  name,
                  number_marks,
                  image_shape=(256, 256, 3),
                  training=True,
                  batch_size=None,
                  shuffle=True,
                  prefetch=None):
    """Generate TensorFlow dataset from image and json files.

    Args:
        data_dir: the directory of the images and json files.
        name: dataset name.
        image_shape: the shape of the target output image of the dataset.
        number_marks: how many marks/points does one sample contains.
        training: True if dataset is for training.
        batch_size: batch size.
        shuffle: True if data should be shuffled.
        prefetch: Set to True to prefetch data.

    Returns:
        a tf.data.dataset.
    """
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(image_shape, (64, 64, number_marks)),
        args=[data_dir, name, image_shape[:2], number_marks, training])

    print("Dataset built from generator: {}".format(name))

    # Shuffle the data.
    if shuffle:
        dataset = dataset.shuffle(1024)

    # Batch the data.
    dataset = dataset.batch(batch_size)

    # Prefetch the data.
    if prefetch is not None:
        dataset = dataset.prefetch(prefetch)

    return dataset


if __name__ == "__main__":
    def top_k_indices(x, k):
        """Returns the k largest element indices from a numpy array. You can find
        the original code here: https://stackoverflow.com/q/6910641
        """
        flat = x.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, x.shape)

    def get_peak_location(heatmap, image_size=(256, 256)):
        """Return the interpreted location of the top 2 predictions."""
        h_height, h_width = heatmap.shape
        [y1, y2], [x1, x2] = top_k_indices(heatmap, 2)
        x = (x1 + (x2 - x1)/4) / h_width * image_size[0]
        y = (y1 + (y2 - y1)/4) / h_height * image_size[1]

        return int(x), int(y)

    def _parse_heatmaps(img, heatmaps):
        # Parse the heatmaps to get mark locations.
        heatmaps = np.transpose(heatmaps, (2, 0, 1))
        for heatmap in heatmaps:
            mark = get_peak_location(heatmap)
            cv2.circle(img, mark, 2, (0, 255, 0), -1)

        # Show individual heatmaps stacked.
        heatmap_idvs = np.hstack(heatmaps[:8])
        for row in range(1, 12, 1):
            heatmap_idvs = np.vstack(
                [heatmap_idvs, np.hstack(heatmaps[row:row+8])])

        return img, heatmap_idvs

    data_dir = "/home/robin/data/facial-marks/wflw_cropped/train"
    batch_size = 1

    # Build a sequence dataset.
    dataset_sequence = make_wflw_dataset(data_dir, "wflw_sequence",
                                         training=True,
                                         batch_size=batch_size,
                                         mode="sequence")

    # Build dataset from generator.
    dataset_from_generator = make_wflw_dataset(data_dir, "wflw_generator",
                                               training=True,
                                               batch_size=batch_size,
                                               mode="generator")
    if not isinstance(dataset_from_generator, tf.keras.utils.Sequence):
        dataset_from_generator = dataset_from_generator.batch(batch_size)

    for sample_s, sample_g in zip(dataset_sequence, dataset_from_generator):
        img_s, heatmap_s = sample_s
        img_g, heatmap_g = sample_g

        img_s, heatmaps_s = _parse_heatmaps(img_s[0], heatmap_s[0])
        img_g, heatmaps_g = _parse_heatmaps(
            img_g[0].numpy(), heatmap_g[0].numpy())

        # Show the result in windows.
        cv2.imshow("images", np.hstack((img_s, img_g)))
        cv2.imshow("heatmaps", np.hstack((heatmaps_s, heatmaps_g)))

        if cv2.waitKey() == 27:
            break
