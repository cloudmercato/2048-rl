"""Tests for Model."""

from py_2048_rl.learning.model import build_loss

import tensorflow.compat.v1 as tf1

# pylint: disable=missing-docstring

def test_loss():
  q_values = tf1.constant([[1, 2, 3, 0],
                          [4, 5, 6, 0],
                          [7, 8, 9, 0]], dtype=tf1.float32)
  targets = tf1.constant([4, 7, 12], dtype=tf1.float32)
  actions = tf1.constant([2, 1, 0], dtype=tf1.int32)

  loss_tensor = build_loss(q_values, targets, actions)
  with tf1.Session() as session:
    loss_value = session.run(loss_tensor)

  # The relevant q values are [3, 5, 7], the differences from the targets are
  # [1, 2, 5], squared differences are [1, 4, 25], the sum of this is 30,
  # so the loss should be 30 / 3 = 10
  assert loss_value == 10
