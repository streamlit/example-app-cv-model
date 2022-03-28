import io
from typing import ByteString, Callable

import numpy as np
import numpy.typing as npt
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet import (
    decode_predictions as mobilenet_decode_predictions,
)
from tensorflow.keras.applications.mobilenet import (
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import (
    decode_predictions as vgg16_decode_predictions,
)
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.preprocessing import image

IMAGENET_INPUT_SIZE = (224, 224)
IMAGENET_INPUT_SHAPE = [224, 224, 3]


def bytes_to_array(image_bytes: ByteString) -> npt.ArrayLike:
    """Converts image stored in bytes into a Numpy array

    Args:
        image_bytes (ByteString): Image stored as bytes

    Returns:
        npt.ArrayLike: Image stored as Numpy array
    """
    return np.array(Image.open(io.BytesIO(image_bytes)))


@st.experimental_singleton
def load_vgg16() -> tf.keras.Model:
    """Loads pre-trained VGG16 Keras model

    Returns:
        tf.keras.Model: VGG-16 model
    """
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    return model


@st.experimental_singleton
def load_mobilenet() -> tf.keras.Model:
    """Loads pre-trained MobileNet Keras model

    Returns:
        tf.keras.Model: MobileNet model
    """
    model = tf.keras.applications.MobileNet(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    return model


SUPPORTED_MODELS = {
    "VGG-16": {
        "load_model": load_vgg16,
        "preprocess_input": vgg16_preprocess_input,
        "decode_predictions": vgg16_decode_predictions,
    },
    "MobileNet": {
        "load_model": load_mobilenet,
        "preprocess_input": mobilenet_preprocess_input,
        "decode_predictions": mobilenet_decode_predictions,
    },
}


@st.experimental_memo
def prepare_image(img_array: npt.ArrayLike, _model_preprocess: Callable) -> npt.ArrayLike:
    """Prepare any image so that it can be fed into a model predict() function.
    This includes: 
    - converting to RGB channels
    - resizing to the appropriate image size expected by the model
    - reshaping to have the proper ordering of dimensions
    - preprocess the image (essentially normalize pixel intensities) 
      according to the model's weights and original using _model_preprocess parameter

    Args:
        img_array (npt.ArrayLike): Input image
        _model_preprocess (Callable): Model preprocessing function

    Returns:
        npt.ArrayLike: Image ready to be fed into predict()
    """
    img = Image.fromarray(img_array)
    img = img.convert("RGB")
    img = img.resize(IMAGENET_INPUT_SIZE, Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = _model_preprocess(img)
    img = img.reshape(*([1] + IMAGENET_INPUT_SHAPE))
    return img
