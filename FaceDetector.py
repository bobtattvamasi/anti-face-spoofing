import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class BlazeFaceDetector:
    def __init__(self, path2model, path2anchors, threshold=0.75):
        self.anchors = self.load_anchors(path2anchors)
        # self.engine = BasicEngine(path2model)
        self.engine = tf.lite.Interpreter(path2model)
        self.engine.allocate_tensors()
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        # score_clipping_thresh 100 -> 50 due to 32float format
        self.score_clipping_thresh = 50.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = threshold
        self.min_suppression_threshold = 0.85

    @staticmethod
    def load_anchors(path):
        return np.float32(np.load(path))

    # preprocess image int -> float [-1,1]
    @staticmethod
    def _preprocess(x):
        return x / 127.5 - 1.0

    def set_input_tensor(self, image):
        """Sets the input tensor."""
        input_data = np.array(image, dtype=np.float32)
        input_data = self._preprocess(input_data)
        self.engine.set_tensor(self.engine.get_input_details()[0]['index'], [input_data])

    def get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.engine.get_output_details()[index]
        tensor = np.squeeze(self.engine.get_tensor(output_details['index']))
        return tensor

    def predict_on_image(self, img):
        resized_img, ratio = resampling_with_original_ratio(Image.fromarray(img),
                                                            [int(self.w_scale), int(self.h_scale)],
                                                            Image.NEAREST)
        self.set_input_tensor(resized_img)
        self.engine.invoke()
        raw_box_tensor = np.reshape(self.get_output_tensor(0),
                                    (1, self.num_anchors, self.num_coords))
        raw_score_tensor = np.reshape(self.get_output_tensor(1),
                                      (1, self.num_anchors, self.num_classes))
        detections = self._tensors_to_detections(raw_box_tensor, raw_score_tensor)

        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, 17))
            filtered_detections.append(faces)

        detection_object = []
        if filtered_detections[0].size != 0:
            max_number = 0
            max_area = 0
            for i, face in enumerate(filtered_detections[0]):
                area = (face[3] - face[1]) * (face[2] - face[0])
                if area > max_area:
                    max_area = area
                    max_number = i

            single_object = {}
            ymin = filtered_detections[0][max_number, 0] * self.h_scale * ratio
            xmin = filtered_detections[0][max_number, 1] * self.w_scale * ratio
            ymax = filtered_detections[0][max_number, 2] * self.h_scale * ratio
            xmax = filtered_detections[0][max_number, 3] * self.w_scale * ratio
            single_object['score'] = filtered_detections[0][max_number, 16]
            single_object['bbox'] = (int(xmin), int(ymin), int(xmax), int(ymax))
            detection_object.append(single_object)

        return detection_object

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.
        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.
        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        detection_boxes = self._decode_boxes(raw_box_tensor)
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clip(-thresh, thresh)
        detection_scores = sigmoid(raw_score_tensor).squeeze(-1)
        mask = detection_scores >= self.min_score_thresh
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = np.expand_dims(detection_scores[i, mask[i]], 1)
            output_detections.append(np.concatenate((boxes, scores), axis=1))

        return output_detections

    def _decode_boxes(self, raw_boxes):
        """Converts the predictions into actual coordinates using
         the anchor boxes. Processes the entire batch at once.
         """
        boxes = np.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * self.anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * self.anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    @staticmethod
    def _weighted_non_max_suppression(detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:
        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."
        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.
        The input detections should be a Tensor of shape (count, 17).
        Returns a list of PyTorch tensors, one for each detected face.
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0:
            return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = np.argsort(-detections[:, 16])

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > 0.5
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = np.copy(detection)
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = sum(scores)
                weighted = np.sum((coordinates * scores), axis=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)
        return output_detections


def resampling_with_original_ratio(img, required_size, sample):
    """Resizes the image to maintain the original aspect ratio by adding pixel padding where needed.
    For example, if your model's input tensor requires a square image but your image is landscape (and
    you don't want to reshape the image to fit), pass this function your image and the required square
    dimensions, and it returns a square version by adding the necessary amount of black pixels on the
    bottom-side only. If the original image is portrait, it adds black pixels on the right-side
    only.
    Args:
      img (:obj:`PIL.Image`): The image to resize.
      required_size (list): The pixel width and height [x, y] that your model requires for input.
      sample (int): A resampling filter for image resizing.
        This can be one of :attr:`PIL.Image.NEAREST` (recommended), :attr:`PIL.Image.BOX`,
        :attr:`PIL.Image.BILINEAR`, :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC`,
        or :attr:`PIL.Image.LANCZOS`. See `Pillow filters
        <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.
    Returns:
      A 2-tuple with a :obj:`PIL.Image` object for the resized image, and a tuple of floats
      representing the aspect ratio difference between the original image and the returned image
      (x delta-ratio, y delta-ratio).
    """
    old_size = img.size
    # Resizing image with original ratio.
    resampling_ratio = min(
        required_size[0] / old_size[0],
        required_size[1] / old_size[1]
    )
    new_size = (
        int(old_size[0] * resampling_ratio),
        int(old_size[1] * resampling_ratio)
    )
    new_img = img.resize(new_size, sample)
    # Expand it to required size.
    delta_w = required_size[0] - new_size[0]
    delta_h = required_size[1] - new_size[1]
    padding = (0, 0, delta_w, delta_h)
    return (ImageOps.expand(new_img, padding), 1 / resampling_ratio)


def sigmoid(x_m):
    return 1. / (1 + np.exp(-x_m))


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, 2:], axis=1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, 2:], axis=0), (A, B, 2)))
    min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :2], axis=1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(box_b[:, :2], axis=0), (A, B, 2)))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ? B / A ? B = A ? B / (area(A) + area(B) - A ? B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    shape = inter.shape
    area_a = np.broadcast_to(np.expand_dims((box_a[:, 2] - box_a[:, 0]) *
                                            (box_a[:, 3] - box_a[:, 1]), axis=1), shape)  # [A,B]
    area_b = np.broadcast_to(np.expand_dims((box_b[:, 2] - box_b[:, 0]) *
                                            (box_b[:, 3] - box_b[:, 1]), axis=0), shape)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return np.squeeze(jaccard(np.expand_dims(box, axis=0), other_boxes), axis=0)