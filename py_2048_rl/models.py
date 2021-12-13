import tensorflow as tf


DEFAULT_MODEL = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation=None),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(4, activation=None)
])
