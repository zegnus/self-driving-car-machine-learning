from styx_msgs.msg import TrafficLight
import tensorflow as tf
import rospy
import os


class TLClassifier(object):

    def __init__(self):

        self.threshold = rospy.get_param('~classifier_threshold')

        # start tensorflow session
        model_dir = './models'
        model_filename = rospy.get_param('~model')
        model_path = os.path.join(model_dir, model_filename)

        with tf.gfile.GFile(model_path, 'rb') as fid:
            od_graph_def = tf.GraphDef()
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(od_graph_def, name='')
            self.session = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Definite input and output Tensors for detection_graph
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was
        # detected.
        detection_boxes_tensor = self.graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores_tensor = self.graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes_tensor = self.graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections_tensor = self.graph.get_tensor_by_name(
            'num_detections:0')

        _, detection_scores, detection_classes, _ = self.session.run(
            [detection_boxes_tensor, detection_scores_tensor,
             detection_classes_tensor, num_detections_tensor],
            {image_tensor: [image]}
        )

        highest_score = detection_scores[0][0]
        if highest_score < self.threshold:
            return TrafficLight.UNKNOWN

        # Off=4, GREEN=1, YELLOW=3, RED=2 -- Classifier
        # UNKNOWN=4, GREEN=2, YELLOW=1, RED=0 -- Styx
        classifierToStyxLabelMap = {
            1: TrafficLight.GREEN,
            2: TrafficLight.RED,
            3: TrafficLight.YELLOW,
            4: TrafficLight.UNKNOWN
        }
        classfierLabel = detection_classes[0][0]
        return classifierToStyxLabelMap.get(classfierLabel)
