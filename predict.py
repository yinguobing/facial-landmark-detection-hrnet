"""Sample module for predicting face marks with HRNetV2."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from postprocessing import FaceDetector, parse_heatmaps, draw_marks
from preprocessing import normalize

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--write_video", type=bool, default=False,
                    help="Write output video.")
args = parser.parse_args()


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
    _, sample_frame = cap.read()

    # Video output by video writer.
    if args.write_video:
        height, width = sample_frame.shape[:2]
        video_writer = cv2.VideoWriter(
            'output.avi', cv2.VideoWriter_fourcc(*'x264'), 30, (width, height))

    # Construct a face detector.
    face_detector = FaceDetector()

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[:480, :640]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Get face area images.
        faceboxes = face_detector.extract_cnn_faceboxes(frame, 0.6)

        if faceboxes:
            faces = []
            for facebox in faceboxes:
                # Preprocess the sample image
                face_img = frame[facebox[1]: facebox[3],
                                 facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (256, 256))
                img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                img_input = normalize(np.array(img_rgb, dtype=np.float32))
                faces.append(img_input)

            faces = np.array(faces, dtype=np.float32)

            # Do prediction.
            heatmap_group = model.predict(faces)

            # Parse the heatmaps to get mark locations.
            mark_group = []
            heatmap_grids = []
            for facebox, heatmaps in zip(faceboxes, heatmap_group):
                width = height = (facebox[2] - facebox[0])
                marks, heatmap_grid = parse_heatmaps(heatmaps, (width, height))

                # Convert the marks locations from local CNN to global image.
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                mark_group.append(marks)
                heatmap_grids.append(heatmap_grid)

            # Draw the marks and the facebox in the original frame.
            draw_marks(frame, mark_group)
            face_detector.draw_box(frame, faceboxes)

            # Show the first heatmap.
            cv2.imshow("heatmap_grid", heatmap_grid[0])

        # Show the result in windows.
        cv2.imshow('image', frame)

        # Write video file.
        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
