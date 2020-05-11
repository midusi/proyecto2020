import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import time
import functools


MAGENTA_PATH = 'https://tfhub.dev/google/lite-model/magenta/'
BASE_PATH = MAGENTA_PATH + 'arbitrary-image-stylization-v1-256/int8/'

STYLE_PREDICT_PATH = tf.keras.utils.get_file(
                        'style_predict.tflite',
                        BASE_PATH + 'prediction/1?lite-format=tflite')

STYLE_TRANSFORM_PATH = tf.keras.utils.get_file(
                           'style_transform.tflite',
                           BASE_PATH + 'transfer/1?lite-format=tflite')


class StyleTransfer:

    def __init__(self, styles={}):
        self.style_bottlenecks = {name: self.bottleneck(path)
                                  for name, path in styles.items()}

        self.style_predict_interpreter = \
            tf.lite.Interpreter(model_path=STYLE_PREDICT_PATH)

        self.style_transform_interpreter = \
            tf.lite.Interpreter(model_path=STYLE_TRANSFORM_PATH)

    # Function to load an image from a file, and add a batch dimension.
    def load_img(self, path_to_img):
        img = tf.io.read_file(path_to_img)

        print(f'Imagen Precargada: {type(img)}')

        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]

        print(f'Imagen Cargada: {type(img)}')
        print(f'Image size: {img.shape}')

        return img

    # Function to pre-process by resizing an central cropping it.
    def preprocess_image(self, image, target_dim):
        # Resize the image so that the shorter dimension becomes 256px.
        shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
        short_dim = min(shape)
        scale = target_dim / short_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        image = tf.image.resize(image, new_shape)

        # Central crop the image.
        image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

        # print(f'Imagen Preprocesada: {type(image)}')

        return image

    def predict_style(self, preprocessed_style_image):
        # Set model input.
        self.style_predict_interpreter.allocate_tensors()
        input_details = self.style_predict_interpreter.get_input_details()
        self.style_predict_interpreter.set_tensor(input_details[0]["index"],
                                                  preprocessed_style_image)

        # Calculate style bottleneck.
        self.style_predict_interpreter.invoke()
        style_bottleneck = self.style_predict_interpreter.tensor(
            self.style_predict_interpreter.get_output_details()[0]["index"]
        )()

        return style_bottleneck

    def bottleneck(self, image_path, target_dim=256):
        image = self.load_img(image_path)
        image = self.preprocess_image(image, target_dim)
        return self.predict_style(image)

    def content(self, path_to_img):
        image = self.load_img(path_to_img)
        return self.preprocess_image(image, 384)

    # Run style transform on preprocessed style image
    def transfer_style(self, style_bottleneck, preprocessed_content_image):
        # Set model input.
        input_details = self.style_transform_interpreter.get_input_details()

        self.style_transform_interpreter.allocate_tensors()

        # Set model inputs.
        self.style_transform_interpreter.set_tensor(input_details[0]["index"],
                                                    preprocessed_content_image)

        self.style_transform_interpreter.set_tensor(input_details[1]["index"],
                                                    style_bottleneck)
        self.style_transform_interpreter.invoke()

        # Transform content image.
        stylized_image = self.style_transform_interpreter.tensor(
            self.style_transform_interpreter.get_output_details()[0]["index"]
        )()

        return stylized_image[0, :, :, ]

    def random_img(self, width=300, height=300):
        array = np.random.rand(1, width, height, 3)
        return tf.convert_to_tensor(array)

