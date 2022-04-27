import io
import numpy as np
from typing import Iterable, List
from PIL import Image
from .image_preprocessors import ImagePreprocessor


class ImageHandler:
    """
    Handles images)
    """

    def __init__(self, image_preprocessors: Iterable[ImagePreprocessor]):
        self.__verify_image_preprocessors(image_preprocessors)
        self.image_preprocessors = image_preprocessors

    def __verify_image_preprocessors(
        self,
        image_preprocessors: Iterable[ImagePreprocessor],
    ):
        if isinstance(image_preprocessors, Iterable):
            for image_preprocessor in image_preprocessors:
                if not isinstance(image_preprocessor, ImagePreprocessor):
                    raise ValueError(
                        "Expected every image_preprocessor within"
                        "image_preprocessors to be of"
                        "ImagePreprocessor type, got:"
                        f"{type(image_preprocessor)}"
                    )
        else:
            raise ValueError("Expected image_preprocessors to be Iterable")

    def preprocess_image(self, image_data: io.BytesIO) -> List[np.ndarray]:
        """
        Preprocesses the image from image_data with self.image_preprocessors.

        Parameters
        ----------
        image_data : io.BytesIO of an image to preprocess

        Returns
        -------
        preprocessed_images : List[np.ndarray] recieved by applying each
            of self.image_preprocessors to image.
            len(preprocessed_images) equals to the amount
            of self.image_preprocessors.
        """
        if not isinstance(image_data, io.BytesIO):
            raise ValueError(
                "Expected image_data to be of io.BytesIO type,"
                f"instead got: {type(image_data)}"
            )

        image = Image.open(image_data)
        preprocessed_images = []

        for image_preprocessor in self.image_preprocessors:
            preprocessed_images.append(image_preprocessor.preprocess(image))

        return preprocessed_images
