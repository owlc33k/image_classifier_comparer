from pprint import pprint

from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB0
from keras import utils
import numpy as np


class ImageClassifierComparer:
    def __init__(self):
        self.model_dict = {
            'VGG16': VGG16(weights='imagenet'),
            'ResNet50': ResNet50(weights='imagenet'),
            'InceptionV3': InceptionV3(weights='imagenet'),
            'EfficientNetB0': EfficientNetB0(weights='imagenet')
        }
        self.post_processed_preds_dict = {}

    def compare(self, img_path):
        for model_name in self.model_dict:
            self.post_processed_preds_dict[model_name] = self.predict(
                img_path, model_name)

    def predict(self, img_path, model_name):
        self.get_image_from_path(img_path, model_name)

        X = self.preprocess_image()
        self.preds = self.model_dict[model_name].predict([X])
        return self.postprocess_result()

    def get_image_from_path(self, img_path, model_name):
        self.img = utils.load_img(
            img_path, target_size=self.model_dict[model_name].input_shape[1:3])

    def preprocess_image(self):
        x = utils.img_to_array(self.img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def postprocess_result(self):
        return decode_predictions(self.preds, top=10)
