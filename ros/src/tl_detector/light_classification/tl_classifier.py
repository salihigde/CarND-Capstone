from styx_msgs.msg import TrafficLight

import numpy as np
import keras
from keras.models import load_model
import cv2

class TLClassifier(object):
    def __init__(self):
	self.sign_classes = ['Red', 'Green', 'Yellow']

	self.model = load_model('model.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img_resize = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img_resize = np.expand_dims(img_resize, axis=0).astype('float32')

	img_resize = (img_resize / 255 - 0.5)

	predict = self.model.predict(img_resize)

	tl_color = self.sign_classes[np.argmax(predict)]

	if tl_color == 'Red':
	      return TrafficLight.RED
	elif tl_color == 'Green':
	      return TrafficLight.GREEN
	elif tl_color == 'Yellow':
	      return TrafficLight.YELLOW
	
        return TrafficLight.UNKNOWN
