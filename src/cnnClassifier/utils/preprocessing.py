import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input


def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """turns an RGB image array with 0-255 pixel values into model input

    VGG16's ImageNet weights expect caffe-style input (RGB->BGR plus mean
    subtraction), not a 1/255 rescale. Training, evaluation and prediction
    all go through this function so the model always sees the same input.

    Args:
        image_array (np.ndarray): image(s) with 0-255 pixel values, channels last

    Returns:
        np.ndarray: preprocessed array (float arrays are modified in place)
    """
    return preprocess_input(image_array)
