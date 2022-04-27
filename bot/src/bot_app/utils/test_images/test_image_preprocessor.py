import pytest
import pickle
import numpy as np

from PIL import Image
from images.image_preprocessors import (
    ImagePreprocessor,
    DefaultPreprocessor,
    ResNetPreprocessor,
)

small_image_path = "test_images/fixtures/small_image.png"


def test_not_implemented_crop():
    base = ImagePreprocessor()
    image = Image.open(small_image_path)
    with pytest.raises(NotImplementedError):
        base.crop(image)


def test_not_implemented_to_numpy():
    base = ImagePreprocessor()
    image = Image.open(small_image_path)
    with pytest.raises(NotImplementedError):
        base.to_numpy(image)


names = ["default", "resnet"]
preprocessors = dict(zip(names, [DefaultPreprocessor(), ResNetPreprocessor()]))


@pytest.mark.parametrize(
    "preprocessor_name, image_path, correct_result_path",
    [
        (
            name,
            f"test_images/fixtures/{size}_image.png",
            f"test_images/fixtures/{size}_{name}_preprocessed.bin",
        )
        for name in names
        for size in ["small", "normal", "big"]
    ],
)
def test_correct_input(preprocessor_name, image_path, correct_result_path):
    preprocessor = preprocessors[preprocessor_name]
    image = Image.open(image_path)
    with open(correct_result_path, "rb") as correct_result_file:
        correct_result = pickle.load(correct_result_file)
        result = preprocessor.preprocess(image)

        assert np.all(result == correct_result)


@pytest.mark.parametrize(
    "preprocessor_name, value",
    [(name, value) for name in names for value in ["sfa", 1, 5.0, [1, 5, 6]]],
)
def test_wrong_input(preprocessor_name, value):
    preprocessor = preprocessors[preprocessor_name]
    with pytest.raises(ValueError):
        preprocessor.preprocess(value)
