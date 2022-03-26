import io
from typing import Callable

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


def bytes_to_array(image_bytes):
    return np.array(Image.open(io.BytesIO(image_bytes)))


@st.experimental_singleton
def load_vgg16():
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
def load_mobilenet():
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
def prepare_image(img_array: npt.ArrayLike, _model_preprocess: Callable):
    img = Image.fromarray(img_array)
    img = img.convert("RGB")
    img = img.resize(IMAGENET_INPUT_SIZE, Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = _model_preprocess(img)
    img = img.reshape(*([1] + IMAGENET_INPUT_SHAPE))
    return img
