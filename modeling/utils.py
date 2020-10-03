from typing import List

import tensorflow as tf


def get_shape_list(x: tf.Tensor) -> List[int]:
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def create_initializer(initializer_range):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
