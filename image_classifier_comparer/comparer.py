from pprint import pprint
import time

from keras.applications import vgg16, resnet, inception_v3, mobilenet, efficientnet, imagenet_utils
from keras import utils
import numpy as np


class ImageClassifierComparer:
    def __init__(self):
        self.model_dict = {
            'VGG16': vgg16.VGG16(weights='imagenet'),
            'ResNet50': resnet.ResNet50(weights='imagenet'),
            'MobileNet': mobilenet.MobileNet(weights='imagenet'),
            'InceptionV3': inception_v3.InceptionV3(weights='imagenet'),
            'EfficientNetB0': efficientnet.EfficientNetB0(weights='imagenet')
        }
        self.preprocessor_dict = {
            'VGG16': vgg16.preprocess_input,
            'ResNet50': resnet.preprocess_input,
            'MobileNet': mobilenet.preprocess_input,
            'InceptionV3': inception_v3.preprocess_input,
            'EfficientNetB0': efficientnet.preprocess_input
        }

    def compare(self, img_path):
        result_dict = {}
        for model_name in self.model_dict:
            result_dict[model_name] = {}
            start = time.time()
            result_dict[model_name]['preds'] = self.predict(
                img_path, model_name)
            result_dict[model_name]['process time'] = time.time() - start
        return result_dict

    def predict(self, img_path, model_name):
        self.get_image_from_path(img_path, model_name)
        X = self.preprocess_image(model_name)
        self.preds = self.model_dict[model_name].predict([X])
        return self.postprocess_result()

    def get_image_from_path(self, img_path, model_name):
        self.img = utils.load_img(
            img_path, target_size=self.model_dict[model_name].input_shape[1:3])

    def preprocess_image(self, model_name):
        x = utils.img_to_array(self.img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocessor_dict[model_name](x)
        return x

    def postprocess_result(self):
        return imagenet_utils.decode_predictions(self.preds, top=10)
