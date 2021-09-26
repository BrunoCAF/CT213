from keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid


class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # Todo: implement object detection logic
        image = self.preprocess_image(image)
        output = self.network.predict(image)
        return self.process_yolo_output(output)

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        # Todo: implement image preprocessing logic
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = (1.0*image)/255.0
        image = np.reshape(image, (1, 120, 160, 3))

        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        # Todo: implement YOLO logic
        features = np.reshape(output, (300, 10))
        features = np.array_split(features, 10, 1)
        t_ball, t_post = features[0], features[5]
        t_ball, t_post = np.reshape(t_ball, (15, 20)), np.reshape(t_post, (15, 20))

        i_b, j_b = np.argwhere(t_ball == t_ball.max())[0]

        post_order = np.reshape(t_post, 300)
        post_1, post_2 = post_order[post_order.argsort()[::-1][0:2]]
        i_p_1, j_p_1 = np.argwhere(t_post == post_1)[0]
        i_p_2, j_p_2 = np.argwhere(t_post == post_2)[0]

        feat_ball = output[i_b][j_b]
        feat_post_1 = output[i_p_1][j_p_1]
        feat_post_2 = output[i_p_2][j_p_2]

        # Todo: change this line
        ball_detection = (sigmoid(feat_ball[0]),
                          (j_b + sigmoid(feat_ball[1]))*coord_scale,
                          (i_b + sigmoid(feat_ball[2]))*coord_scale,
                          bb_scale*self.anchor_box_ball[0]*np.exp(feat_ball[3]),
                          bb_scale*self.anchor_box_ball[1]*np.exp(feat_ball[4]))

        # Todo: change this line
        post1_detection = (sigmoid(feat_post_1[5]),
                           (j_p_1 + sigmoid(feat_post_1[6]))*coord_scale,
                           (i_p_1 + sigmoid(feat_post_1[7]))*coord_scale,
                           bb_scale*self.anchor_box_post[0]*np.exp(feat_post_1[8]),
                           bb_scale*self.anchor_box_post[1]*np.exp(feat_post_1[9]))

        # Todo: change this line
        post2_detection = (sigmoid(feat_post_2[5]),
                           (j_p_2 + sigmoid(feat_post_2[6]))*coord_scale,
                           (i_p_2 + sigmoid(feat_post_2[7]))*coord_scale,
                           bb_scale*self.anchor_box_post[0]*np.exp(feat_post_2[8]),
                           bb_scale*self.anchor_box_post[1]*np.exp(feat_post_2[9]))

        return ball_detection, post1_detection, post2_detection
