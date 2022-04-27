import pytest
import io
import pickle
import numpy as np

from PIL import Image
from images.image_preprocessors import (
    DefaultPreprocessor,
    ImagePreprocessor,
    ResNetPreprocessor,
)
from images.image_handler import ImageHandler

FIXTURES_PATH = "test_images/fixtures/"
IMAGE_PATH = f"{FIXTURES_PATH}normal_image.png"


@pytest.mark.parametrize(
    "args",
    [
        5,
        "assf",
        [1, 2, 3],
        DefaultPreprocessor(),
        [DefaultPreprocessor(), ImagePreprocessor(), 2],
    ],
)
def test_incorrect_init_args(args):
    with pytest.raises(ValueError):
        ImageHandler(args)


@pytest.mark.parametrize(
    "args",
    [
        [DefaultPreprocessor()],
        [ImagePreprocessor()],
        [DefaultPreprocessor(), ResNetPreprocessor(), ImagePreprocessor()],
    ],
)
def test_correct_init_args(args):
    _ = ImageHandler(args)


@pytest.mark.parametrize(
    "preprocessors,correct_result_path",
    [
        (
            [DefaultPreprocessor(), ResNetPreprocessor()],
            f"{FIXTURES_PATH}default_resnet.bin",
        ),
        (
            [ResNetPreprocessor(), DefaultPreprocessor()],
            f"{FIXTURES_PATH}resnet_default.bin",
        ),
        (
            [DefaultPreprocessor(), DefaultPreprocessor()],
            f"{FIXTURES_PATH}default_default.bin",
        ),
        (
            [DefaultPreprocessor()],
            f"{FIXTURES_PATH}normal_default_preprocessed.bin",
        ),
    ],
)
def test_preprocessing_pipeline(preprocessors, correct_result_path):
    with open(IMAGE_PATH, "rb") as image_file:
        image_data = io.BytesIO(image_file.read())
    with open(correct_result_path, "rb") as file:
        correct_result = pickle.load(file)
    image_handler = ImageHandler(preprocessors)
    result = image_handler.preprocess_image(image_data)
    if len(preprocessors) == 1:
        assert (result == correct_result).all()
    else:
        for result_element, correct_result_element in zip(
            result, correct_result
        ):
            assert (result_element == correct_result_element).all()


@pytest.mark.parametrize("value", ["315a", 5, 3.14, Image.open(IMAGE_PATH)])
def test_incorrect_preprocess_input(value):
    image_handler = ImageHandler([DefaultPreprocessor()])
    with pytest.raises(ValueError):
        image_handler.preprocess_image(value)
