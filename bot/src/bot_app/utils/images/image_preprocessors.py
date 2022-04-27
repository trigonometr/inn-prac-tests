import numpy as np
from PIL.Image import Image


class ImagePreprocessor:
    """
    Base class for ImagePreprocessors, which preprocesses
    the image by cropping it and then converting to numpy.
    """

    def preprocess(self, image: Image):
        if isinstance(image, Image):
            cropped_image = self.crop(image)
            return self.to_numpy(cropped_image)
        else:
            raise ValueError(f"Expected PIL.Image.Image, got: {type(image)}")

    @staticmethod
    def crop(image: Image) -> Image:
        raise NotImplementedError

    @staticmethod
    def to_numpy(image: Image) -> np.ndarray:
        raise NotImplementedError


class DefaultPreprocessor(ImagePreprocessor):
    """
    Default image preprocessor
    """

    @staticmethod
    def crop(image: Image) -> Image:
        return image.crop((0, 0, 20, 20))

    @staticmethod
    def to_numpy(image: Image) -> np.ndarray:
        return np.array(image, dtype=np.float32)


class ResNetPreprocessor(ImagePreprocessor):
    """
    Preprocesses image for ResNet
    """

    @staticmethod
    def crop(image: Image) -> Image:
        return image.crop((0, 0, 1280, 720))

    @staticmethod
    def to_numpy(image: Image) -> np.ndarray:
        image_np = np.array(image, dtype=np.float32)
        old_shape = image_np.shape
        return image_np.reshape((old_shape[-1::-1]))
