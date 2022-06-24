import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from image_classifier_comparer.comparer import ImageClassifierComparer  # noqa


def test_get_image_from_path():
    icc = ImageClassifierComparer()
    model_name = 'VGG16'
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)


def test_preprocess_image():
    icc = ImageClassifierComparer()
    model_name = 'VGG16'
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    assert X.shape == (
        1, icc.model_dict[model_name].input_shape[1], icc.model_dict[model_name].input_shape[2], 3)


def test_predict_VGG16():
    icc = ImageClassifierComparer()
    model_name = 'VGG16'
    preds = []
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'lena.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'fruits.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])
    assert preds == ['baboon', 'bonnet', 'lemon']


def test_predict_ResNet50():
    icc = ImageClassifierComparer()
    model_name = 'ResNet50'
    preds = []
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'lena.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'fruits.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])
    assert preds == ['baboon', 'brassiere', 'lemon']


def test_predict_InceptionV3():
    icc = ImageClassifierComparer()
    model_name = 'InceptionV3'
    preds = []
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'lena.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'fruits.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])
    assert preds == ['clog', 'leatherback_turtle', 'web_site']


def test_predict_EfficientNetB0():
    icc = ImageClassifierComparer()
    model_name = 'EfficientNetB0'
    preds = []
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'lena.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])

    path = os.path.join('tests', 'image', 'fruits.jpg')
    icc.get_image_from_path(path, model_name)
    X = icc.preprocess_image()
    icc.preds = icc.model_dict[model_name].predict([X])
    preds.append(icc.postprocess_result()[0][0][1])
    assert preds == ['lion', 'seat_belt', 'jellyfish']


def test_compare():
    icc = ImageClassifierComparer()
    model_name = 'EfficientNetB0'
    path = os.path.join('tests', 'image', 'baboon.jpg')
    icc.compare(path)
    assert icc.post_processed_preds_dict[model_name][0][0][1] == 'lion'