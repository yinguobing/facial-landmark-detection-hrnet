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
parser.add_argument("--write_video", type=bool, default=False,
                    help="Write output video.")
args = parser.parse_args()


class FaceDetector:
    """Detect human face from image"""

    def __init__(self,
                 dnn_proto_text='./assets/deploy.prototxt',
                 dnn_model='./assets/res10_300x300_ssd_iter_140000.caffemodel'):
        """Initialization"""
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        """
        Get the bounding box of faces in image using dnn.
        """
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append(
                    [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

        self.detection_result = [faceboxes, confidences]

        return confidences, faceboxes

    def draw_all_result(self, image):
        """Draw the detection result on image"""
        for facebox, conf in self.detection_result:
            cv2.rectangle(image, (facebox[0], facebox[1]),
                          (facebox[2], facebox[3]), (0, 255, 0))
            label = "face: %.4f" % conf
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                          (facebox[0] + label_size[0],
                           facebox[1] + base_line),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (facebox[0], facebox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color, 2)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.get_faceboxes(
            image=image, threshold=0.9)

        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                return facebox

        return None


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

        # Get face area image.
        facebox = face_detector.extract_cnn_facebox(frame)

        if facebox is not None:
            # Detect landmarks from image.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]

            # Preprocess the sample image
            face_img = cv2.resize(face_img, (256, 256))
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_input = normalize(np.array(img_rgb, dtype=np.float32))

            # Do prediction.
            heatmaps = model.predict(tf.expand_dims(img_input, 0))[0]

            # Parse the heatmaps to get mark locations.
            marks, heatmap_grid = parse_heatmaps(heatmaps)

            # Convert the marks locations from local CNN to global image.
            marks = marks / 256 * (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Draw the marks in the original frame.
            for mark in marks:
                cv2.circle(frame, tuple(mark.astype(int)), 2, (0, 255, 0), -1)
            face_detector.draw_box(frame, [facebox])

            cv2.imshow("heatmap_grid", heatmap_grid)

        # Show the result in windows.
        cv2.imshow('image', frame)

        # Write video file.
        if args.write_video:
            video_writer.write(frame)

        if cv2.waitKey(1) == 27:
            break
