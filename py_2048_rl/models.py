"""Tensorflow model definitions

Models available for use in 2048 RL modeling.
Common characteristics: 16 inputs (representation of game state), 4 labels (representation of potential actions)
"""

import tensorflow as tf

model_16_64_64_64_64_64_64_4_relu = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation=None)
])

model_16_64_4_relu = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation=None)
])

model_16_256_256_4_relu = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation=None)
])

model_16_512_512_512_512_4_sigmoid = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(4, activation=None)
])

DEFAULT_MODEL = model_16_64_64_64_64_64_64_4_relu