"""Sample module for predicting face marks with HRNetV2."""
import cv2
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from preprocess import normalize

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


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


def parse_heatmaps(heatmaps):
    # Parse the heatmaps to get mark locations.
    marks = []
    heatmaps = np.transpose(heatmaps, (2, 0, 1))
    for heatmap in heatmaps:
        marks.append(get_peak_location(heatmap))

    # Show individual heatmaps stacked.
    heatmap_grid = np.hstack(heatmaps[:8])
    for row in range(1, 12, 1):
        heatmap_grid = np.vstack(
            [heatmap_grid, np.hstack(heatmaps[row:row+8])])

    return np.array(marks), heatmap_grid


if __name__ == "__main__":
    # Restore the model.
    model = tf.keras.models.load_model("./exported")

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        frame = frame[:480, :480]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Read in and preprocess the sample image
        frame = cv2.resize(frame, (256, 256))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = normalize(np.array(img_rgb, dtype=np.float32))

        # Do prediction.
        heatmaps = model.predict(tf.expand_dims(img_input, 0))[0]

        # Parse the heatmaps to get mark locations.
        marks, heatmap_grid = parse_heatmaps(heatmaps)
        for mark in marks:
            cv2.circle(frame, tuple(mark.astype(int)), 2, (0, 255, 0), -1)

        # Show the result in windows.
        cv2.imshow('image', frame)
        cv2.imshow("heatmap_grid", heatmap_grid)
        if cv2.waitKey(27) == 27:
            break
