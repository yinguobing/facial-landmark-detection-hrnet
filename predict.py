"""Sample module for predicting face marks with HRNetV2."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from postprocessing import parse_heatmaps, draw_marks
from preprocessing import normalize
from face_detector.detector import Detector

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--write_video", type=bool, default=False,
                    help="Write output video.")
args = parser.parse_args()

# Allow GPU memory growth.
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

if __name__ == "__main__":
    """Run human head pose estimation from video files."""

    # What is the threshold value for face detection.
    threshold = 0.7

    # Construct a face detector.
    detector_face = Detector('assets/face_model')

    # Restore the model.
    model = tf.keras.models.load_model("./exported/hrnetv2")

    # Setup the video source. If no video file provided, the default webcam will
    # be used.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # If reading frames from a webcam, try setting the camera resolution.
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Get the real frame resolution.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Video output by video writer.
    if args.write_video:
        video_writer = cv2.VideoWriter(
            'output.avi', cv2.VideoWriter_fourcc(*'avc1'), frame_rate, (frame_width, frame_height))

    # Introduce a metter to measure the FPS.
    tm = cv2.TickMeter()

    # Loop through the video frames.
    while True:
        tm.start()

        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Preprocess the input image.
        _image = detector_face.preprocess(frame)

        # Run the model
        boxes, scores, _ = detector_face.predict(_image, threshold)

        # Transform the boxes into squares.
        boxes = detector_face.transform_to_square(
            boxes, scale=1.22, offset=(0, 0.13))

        # Clip the boxes if they cross the image boundaries.
        boxes, _ = detector_face.clip_boxes(
            boxes, (0, 0, frame_height, frame_width))
        boxes = boxes.astype(np.int32)

        if boxes.size > 0:
            faces = []
            for facebox in boxes:
                # Crop the face image
                top, left, bottom, right = facebox
                face_image = frame[top:bottom, left:right]

                # Preprocess it.
                face_image = cv2.resize(face_image, (256, 256))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_image = normalize(np.array(face_image, dtype=np.float32))
                faces.append(face_image)

            faces = np.array(faces, dtype=np.float32)

            # Do prediction.
            heatmap_group = model.predict(faces)

            # Parse the heatmaps to get mark locations.
            mark_group = []
            heatmap_grids = []
            for facebox, heatmaps in zip(boxes, heatmap_group):
                top, left, bottom, right = facebox
                width = height = (bottom - top)

                marks, heatmap_grid = parse_heatmaps(heatmaps, (width, height))

                # Convert the marks locations from local CNN to global image.
                marks[:, 0] += left
                marks[:, 1] += top

                mark_group.append(marks)
                heatmap_grids.append(heatmap_grid)

            # Draw the marks and the facebox in the original frame.
            draw_marks(frame, mark_group)
            detector_face.draw_boxes(frame, boxes, scores)

            # Show the first heatmap.
            cv2.imshow("heatmap_grid", heatmap_grid[0])

        # Show the result in windows.
        cv2.imshow('image', frame)

        # Write video file.
        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
